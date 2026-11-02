"""Co-Pilot router: Hosted Meta Muse Wonder REST client + data-grounded fallback.

Calls Hosted Meta Muse Wonder directly via ``httpx.AsyncClient``
(``MUSE_WONDER_API_URL`` / ``META_API_KEY``, model ``muse-wonder-1.0``).
Never logs the API key and never imports LiteLLM.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app import sample_trainer

# Fix 6: load agentic-setup/.env automatically (never edit the file itself).
def _load_central_env() -> None:
    path = Path("/Users/dixitpatel/Documents/Personal-Dev-Workspace/Meta_hackathon/agentic-setup/.env")
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    except OSError:
        pass


_load_central_env()

router = APIRouter()

WONDER_MODEL = "muse-wonder-1.0"
ACTION_TYPES = (
    "map_oncotree",
    "cross_reference_genes",
    "show_batch_failures",
    "prep_literature_context",
)


class VariantRef(BaseModel):
    # Fix 2: accept snake_case and camelCase aliases.
    model_config = ConfigDict(populate_by_name=True)

    variant_id: str = Field(default="", validation_alias=AliasChoices("variant_id", "variantId"))
    hugo_symbol: str = Field(default="UNKNOWN", validation_alias=AliasChoices("hugo_symbol", "hugoGeneSymbol", "gene"))
    protein_change: str | None = Field(default=None, validation_alias=AliasChoices("protein_change", "proteinChange"))


class IndexSampleRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    study_id: str
    cancer_type: str
    case_id: str
    variants: list[VariantRef] = Field(default_factory=list)


class ChatRequest(BaseModel):
    # Fix 1: accept both action_type and action aliases.
    model_config = ConfigDict(populate_by_name=True)

    message: str = ""
    action_type: str | None = Field(default=None, validation_alias=AliasChoices("action_type", "action"))
    action: str | None = None
    study_id: str | None = None
    cancer_type: str | None = None
    case_id: str | None = None
    variants: list[VariantRef] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_action(self):
        action = self.action_type or self.action
        if action is None or str(action).strip() in ("", "chat"):
            action = None
        self.action_type = action
        self.action = None
        return self


class SaveMappingRequest(BaseModel):
    oncotree_code: str
    oncotree_name: str = ""
    default_curie: str
    default_label: str = ""
    alternates: list[str] = Field(default_factory=list)


def _wonder_config() -> tuple[str | None, str | None]:
    return os.environ.get("MUSE_WONDER_API_URL"), os.environ.get("META_API_KEY")


def _mongo_cached_count() -> int:
    # Fix 4: pymongo first, `docker exec gn-mongo mongosh` fallback.
    try:
        return int(sample_trainer.mongo_cached_count())
    except Exception:
        return 0


@router.get("/health")
def copilot_health() -> dict:
    counts = sample_trainer.kb_counts()
    return {
        "status": "ok",
        "mongo_cached_count": _mongo_cached_count(),
        **counts,
    }


@router.post("/index-sample")
def copilot_index_sample(request: IndexSampleRequest) -> dict:
    result = sample_trainer.index_patient_sample(
        request.study_id,
        request.cancer_type,
        request.case_id,
        [v.model_dump() for v in request.variants],
    )
    failed = sample_trainer.inspect_batch_failures(
        [v.variant_id for v in request.variants if v.variant_id]
    )
    return {
        "indexed_count": result["indexed_count"],
        "failed_count": len(failed),
        "failed_variants": failed,
        "genes_indexed": result["genes_indexed"],
        "kb_stats": result["kb_stats"],
    }


def _resolve_oncotree(cancer_type: str | None, message: str = "") -> tuple[str, dict]:
    """Resolve an OncoTree code + mapping entry (Fix 3).

    Matches by exact code key OR case-insensitive ``oncotree_name``, and also
    scans the free-text message for any known OncoTree code token.
    """
    mappings = sample_trainer.load_oncotree_map().get("mappings", {})
    # An explicit OncoTree code token in the message wins over the study type.
    tokens = set(re.findall(r"[A-Z]{2,8}", str(message or "").upper()))
    for token in tokens:
        if token in mappings and isinstance(mappings[token], dict):
            return token, mappings[token]
    for candidate in [str(cancer_type or "").strip(), str(message or "")]:
        if not candidate:
            continue
        upper = candidate.upper()
        if upper in mappings and isinstance(mappings[upper], dict):
            return upper, mappings[upper]
        for code, entry in mappings.items():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("oncotree_name") or "")
            if name and name.lower() in candidate.lower():
                return code, entry
    direct = str(cancer_type or "").strip().upper()
    return direct, mappings.get(direct, {}) if isinstance(mappings.get(direct), dict) else {}


def _format_track(track: dict) -> str:
    curie = track.get("ontology_curie") or "n/a"
    name = track.get("biosample_name") or track.get("track_name") or "unknown tissue"
    raw = track.get("raw_score")
    quant = track.get("quantile_score")
    return f"{curie} ({name}; raw={raw}, quantile={quant})"


def _rich_gene_bullets(genes: list[str]) -> str:
    """Multi-bullet KB summary per gene (Fix 6)."""
    try:
        with open(sample_trainer.store_path()) as f:
            store = json.load(f)
    except (OSError, ValueError):
        store = {}
    genes_map = store.get("genes", {}) if isinstance(store, dict) else {}
    lines: list[str] = []
    for gene in genes:
        entry = genes_map.get(gene)
        if not isinstance(entry, dict):
            lines.append(f"- {gene}: not yet indexed in study_gene_knowledge_store.json.")
            continue
        studies = entry.get("studies", []) or []
        study_labels = [f"{s.get('study_id')} ({s.get('cancer_type')})" for s in studies if isinstance(s, dict)]
        variants = entry.get("variants", []) or []
        hg38_seen: list[str] = []
        splicing: list[str] = []
        for v in variants:
            if isinstance(v, dict):
                ms = v.get("merged_splicing_score")
                if ms is not None and str(ms) not in splicing:
                    splicing.append(str(ms))
        # hg38 ids come from the sample variant records; fall back to variant_id.
        hg38_seen = [str(v.get("variant_id")) for v in variants if isinstance(v, dict) and v.get("variant_id")][:5]
        top_tracks = entry.get("top_tracks", {}) if isinstance(entry.get("top_tracks"), dict) else {}
        track_bits: list[str] = []
        for output_type in sample_trainer.OUTPUT_TYPES:
            tracks = top_tracks.get(output_type) or []
            if tracks:
                shown = "; ".join(_format_track(t) for t in tracks[:3] if isinstance(t, dict))
                track_bits.append(f"{output_type}: {shown}")
        lines.append(
            f"- {gene}: studies [{', '.join(study_labels) or 'n/a'}]; "
            f"{len(variants)} indexed variant(s) [{', '.join(hg38_seen) or 'n/a'}]; "
            f"merged_splicing_score(s) [{', '.join(splicing) or 'n/a'}]; "
            f"top tracks {{{' | '.join(track_bits) or 'no tracks yet'}}}"
        )
    return "\n".join(lines)


def _rich_sample_brief(study_id: str, cancer_type: str | None, case_id: str | None, genes: list[str]) -> str:
    """Study + Sample + Gene + Modality literature foundation brief (Fix 6)."""
    try:
        with open(sample_trainer.store_path()) as f:
            store = json.load(f)
    except (OSError, ValueError):
        store = {}
    samples = store.get("samples", {}) if isinstance(store, dict) else {}
    key = f"{study_id}::{case_id}"
    sample = samples.get(key) if isinstance(samples, dict) else None
    code, entry = _resolve_oncotree(cancer_type)
    lines = [
        f"Study: {study_id} (OncoTree {code or 'n/a'}"
        + (f" — {entry.get('oncotree_name')}" if entry.get("oncotree_name") else "")
        + f"; default tissue {entry.get('default_curie', 'n/a')})",
        f"Sample: {case_id or 'n/a'} with {len(genes)} gene(s): {', '.join(genes) or 'n/a'}",
    ]
    if isinstance(sample, dict):
        for record in sample.get("variants", []) or []:
            if not isinstance(record, dict):
                continue
            top = record.get("top_tracks", {}) if isinstance(record.get("top_tracks"), dict) else {}
            dominant = "n/a"
            best = None
            for output_type in sample_trainer.OUTPUT_TYPES:
                for track in top.get(output_type, []) or []:
                    if isinstance(track, dict) and track.get("raw_score") is not None:
                        try:
                            score = abs(float(track["raw_score"]))
                        except (TypeError, ValueError):
                            continue
                        if best is None or score > best[0]:
                            best = (score, output_type)
            if best is not None:
                dominant = f"{best[1]} expression/splicing" if best[1] == "RNA_SEQ" else f"{best[1]} splicing"
            gene = record.get("hugo_symbol") or record.get("canonical_gene") or "?"
            hg38 = record.get("hg38_variant") or record.get("variant_id")
            ms = record.get("merged_splicing_score")
            lines.append(
                f"- {gene} ({hg38}): dominant modality {dominant}; "
                f"merged_splicing_score={ms}; PubMed search: \"{gene}\" {code or ''} AlphaGenome {dominant}"
            )
    else:
        lines.append(f"- Sample {key} not yet indexed; showing request genes:")
        lines.append(_rich_gene_bullets(genes))
    return "\n".join(lines)


def _fallback_reply(request: ChatRequest) -> dict:
    """Deterministic data-grounded reply used offline / in CI."""
    action = (request.action_type or "").strip()
    study = request.study_id or "unknown-study"
    code, entry = _resolve_oncotree(request.cancer_type, request.message)
    genes = [v.hugo_symbol for v in request.variants] or ["(no variants)"]
    suggested_mapping = None
    if action == "map_oncotree":
        if entry:
            suggested_mapping = {
                "oncotree_code": code,
                "oncotree_name": entry.get("oncotree_name", ""),
                "default_curie": entry.get("default_curie"),
                "default_label": entry.get("default_label"),
                "alternates": entry.get("alternates", []),
            }
            text = (
                f"Study {study} ({code}) maps to {entry.get('default_curie')} "
                f"({entry.get('default_label')}). 1-click save payload included."
            )
        else:
            text = (
                f"No curated OncoTree mapping for '{request.cancer_type}'. "
                "Suggest one via save-mapping with a UBERON/CL curie."
            )
    elif action == "cross_reference_genes":
        text = f"Cross-reference for {study} ({len(genes)} gene(s)):\n{_rich_gene_bullets(genes)}"
    elif action == "show_batch_failures":
        failed = sample_trainer.inspect_batch_failures(
            [v.variant_id for v in request.variants if v.variant_id]
        )
        text = (
            f"{len(failed)} failed variant(s) in this batch: "
            + (
                ", ".join(f"{f['variant_id']} ({f['cause']})" for f in failed)
                if failed
                else "none found in log tail or Mongo."
            )
        )
        return {"reply": text, "action_type": action, "failed_variants": failed, "source": "fallback"}
    elif action == "prep_literature_context":
        text = (
            f"Study + Sample + Gene + Modality Literature Foundation Brief "
            f"for {study} ({request.cancer_type}, case {request.case_id}):\n"
            + _rich_sample_brief(study, request.cancer_type, request.case_id, genes)
        )
    else:
        text = (
            f"Co-Pilot (offline) context for {study}: received "
            f"'{request.message[:500]}' with {len(request.variants)} variant(s) "
            f"and {len(request.history)} history turn(s)."
        )
    payload: dict = {"reply": text, "action_type": action or None, "source": "fallback"}
    if suggested_mapping is not None:
        payload["suggested_mapping"] = suggested_mapping
    return payload


async def _wonder_reply(request: ChatRequest) -> dict | None:
    """Call Hosted Muse Wonder; return None when unconfigured or on any error."""
    base_url, api_key = _wonder_config()
    if not base_url or not api_key:
        return None
    # Fix 6: ground the hosted call with the KB + OncoTree slice.
    code, entry = _resolve_oncotree(request.cancer_type, request.message)
    genes = [v.hugo_symbol for v in request.variants]
    kb_context = {
        "oncotree_code": code,
        "oncotree_entry": entry,
        "gene_summary": _rich_gene_bullets(genes) if genes else "",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                base_url.rstrip("/") + "/chat",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": WONDER_MODEL,
                    "message": request.message,
                    "action_type": request.action_type,
                    "study_id": request.study_id,
                    "cancer_type": request.cancer_type,
                    "case_id": request.case_id,
                    "variants": [v.model_dump() for v in request.variants],
                    "history": request.history,
                    "kb_context": kb_context,
                },
            )
            response.raise_for_status()
            data = response.json()
        if isinstance(data, dict):
            data.setdefault("source", "muse-wonder")
            return data
        return {"reply": str(data), "source": "muse-wonder"}
    except Exception:
        return None


@router.post("/chat")
async def copilot_chat(request: ChatRequest) -> dict:
    if request.action_type and request.action_type not in ACTION_TYPES:
        return {
            "reply": f"Unknown action_type '{request.action_type}'. "
            f"Supported: {', '.join(ACTION_TYPES)}.",
            "action_type": request.action_type,
            "source": "fallback",
        }
    live = await _wonder_reply(request)
    if live is not None:
        return live
    return _fallback_reply(request)


@router.post("/save-mapping")
def copilot_save_mapping(request: SaveMappingRequest) -> dict:
    code = request.oncotree_code.strip().upper()
    if not code:
        return {"saved": False, "error": "oncotree_code is required"}
    if not sample_trainer.CURIE_PATTERN.match(request.default_curie.strip()):
        return {
            "saved": False,
            "error": "default_curie must match UBERON:NNNNNNN or CL:NNNNNNN",
        }
    for curie in request.alternates:
        if not sample_trainer.CURIE_PATTERN.match(curie.strip()):
            return {"saved": False, "error": f"invalid alternate curie '{curie}'"}
    path = sample_trainer.oncotree_map_path()
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {"mappings": {}}
    data.setdefault("mappings", {})[code] = {
        "oncotree_name": request.oncotree_name,
        "default_curie": request.default_curie.strip(),
        "default_label": request.default_label,
        "alternates": [c.strip() for c in request.alternates],
        "availability": {
            "RNA_SEQ": True,
            "SPLICE_SITE_USAGE": True,
            "SPLICE_JUNCTIONS": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return {"saved": True, "oncotree_code": code, "entry": data["mappings"][code]}
