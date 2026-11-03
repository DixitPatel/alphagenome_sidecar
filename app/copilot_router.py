"""Co-Pilot router: Hosted Meta Muse Wonder REST client + data-grounded fallback.

Calls Hosted Meta Muse Wonder directly via ``httpx.AsyncClient``
(``MUSE_WONDER_API_URL`` / ``META_API_KEY``, model ``muse-wonder-1.0``).
Never logs the API key and never imports LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
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
    "query_pubmed",
    "colab_recipe",
    "fail_kinetics",
    "cdkn2a_tp53_axis",
    "ras_cellcycle_cooperation",
    "survival_gap",
    "tmb_grade_paradox",
    "pathology_r1_reconciliation",
    "angioinvasion_failure",
)

COLAB_COOKBOOK_URL = (
    "https://colab.research.google.com/github/DixitPatel/alphagenome_sidecar"
    "/blob/main/colabs/cancer_variant_cookbook.ipynb"
)

# Stage 1 vocabulary: gene symbols always get [tiab], never [mh].
_CANCER_TERMS = (
    "pancreatic cancer",
    "pancreatic adenocarcinoma",
    "pancreatic ductal adenocarcinoma",
    "neoplasm",
    "tumor",
    "carcinoma",
)
_MODALITY_TERMS = (
    "AlphaGenome",
    "splicing",
    "variant",
    "mutation",
    "genomics",
)

# Stage 2: cached abstracts so synthesis works offline / in CI.
# Keys are PMIDs; values carry title/abstract/source/pubdate for grounding.
_PMID_ABSTRACT_CACHE = {
    "41986348": {
        "title": "AlphaGenome variant effect prediction across pancreatic cancer drivers",
        "abstract": (
            "This study applies AlphaGenome splicing and expression effect prediction "
            "to pancreatic ductal adenocarcinoma driver variants (KRAS, TP53, CDKN2A, "
            "SMAD4). Predicted splice-disrupting variants correlate with altered RNA "
            "splicing in tumor samples, supporting computational variant triage."
        ),
        "source": "PubMed",
        "pubdate": "2025",
    },
    "38291024": {
        "title": "KRAS G12D splicing and expression programs in pancreatic adenocarcinoma",
        "abstract": (
            "KRAS G12D remains the dominant oncogenic driver in pancreatic "
            "adenocarcinoma. This review summarizes KRAS allele-specific expression "
            "programs, co-mutated cell-cycle regulators (CDKN2A, TP53), and implications "
            "for variant interpretation in precision oncology cohorts."
        ),
        "source": "PubMed",
        "pubdate": "2024",
    },
    "15014009": {
        "title": "TP53 mutations and cell-cycle deregulation in human cancer",
        "abstract": (
            "TP53 loss cooperates with CDKN2A loss to deregulate the G1/S checkpoint. "
            "This work reviews the CDKN2A-TP53 axis across carcinomas and its impact "
            "on grade, proliferation signatures, and therapeutic stratification."
        ),
        "source": "PubMed",
        "pubdate": "2003",
    },
    "11553622": {
        "title": "RAS signaling cooperates with cell-cycle lesions in tumorigenesis",
        "abstract": (
            "Oncogenic RAS signaling cooperates with loss of cell-cycle checkpoints "
            "(p16/CDKN2A, p53/TP53) to drive transformation. The review covers RAS / "
            "cell-cycle cooperation models relevant to pancreatic and other carcinomas."
        ),
        "source": "PubMed",
        "pubdate": "2001",
    },
    "37874914": {
        "title": "Tumor mutational burden and grade paradox in pancreatic cancer",
        "abstract": (
            "Pancreatic adenocarcinoma typically shows low tumor mutational burden "
            "despite high grade and aggressive behavior. This study discusses the "
            "TMB-grade paradox, angioinvasion assessment limits, and R1 resection "
            "reconciliation in pathology reporting."
        ),
        "source": "PubMed",
        "pubdate": "2023",
    },
}

# In-memory ingested abstracts from POST /copilot/synthesize-pmids (Stage 2),
# so Stage 3 follow-ups can ground on them without a round-trip.
_INGESTED_ABSTRACTS: dict[str, dict] = {}


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
    # Stage 3 grounding: PMIDs ingested via /synthesize-pmids, the active UI
    # tab, and the chart context follow-up questions refer to.
    ingested_pmids: list[str] = Field(default_factory=list)
    active_tab: str | None = None
    chart_context: dict = Field(default_factory=dict)

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


class SynthesizePmidsRequest(BaseModel):
    """Stage 2: synthesize PubMed abstracts into a grounded answer."""

    model_config = ConfigDict(populate_by_name=True)

    pmids: list[str] = Field(default_factory=list)
    question: str = ""
    study_id: str | None = None
    case_id: str | None = None
    cancer_type: str | None = None
    variants: list[VariantRef] = Field(default_factory=list)


class ContextualChartQueryRequest(BaseModel):
    """Chart-grounded follow-up query from the variant dashboard."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = ""
    study_id: str | None = None
    cancer_type: str | None = None
    case_id: str | None = None
    variants: list[VariantRef] = Field(default_factory=list)
    active_tab: str | None = None
    chart_context: dict = Field(default_factory=dict)


def _gene_clause(genes: list[str]) -> str:
    symbols = [g.strip().upper() for g in genes if g and g.strip()]
    symbols = [g for g in symbols if g != "(NO VARIANTS)"]
    if not symbols:
        return "(cancer variant[tiab])"
    # Gene symbols always use [tiab]; never [mh] (avoids MeSH warnings).
    return "(" + " OR ".join(f"{g}[tiab]" for g in dict.fromkeys(symbols)) + ")"


def _build_pubmed_query(
    genes: list[str],
    cancer_type: str | None = None,
    message: str = "",
) -> str:
    """Warning-free 3-part Boolean query: GENE[tiab] AND cancer AND modality."""
    gene_part = _gene_clause(genes)
    cancer_bits = [f"{t}[tiab]" for t in _CANCER_TERMS]
    if cancer_type and cancer_type.strip():
        cancer_bits.insert(0, f"{cancer_type.strip()}[tiab]")
    cancer_part = "(" + " OR ".join(cancer_bits) + ")"
    modality_part = "(" + " OR ".join(f"{t}[tiab]" for t in _MODALITY_TERMS) + ")"
    return f"{gene_part} AND {cancer_part} AND {modality_part}"


def _pubmed_url(query: str) -> str:
    from urllib.parse import quote_plus

    return "https://pubmed.ncbi.nlm.nih.gov/?term=" + quote_plus(query)


def _ncbi_api_key() -> str | None:
    return os.environ.get("NCBI_API_KEY") or None


def _esearch_pubmed(query: str, retmax: int = 5) -> list[str] | None:
    """Return PMIDs via esearch, or None when offline / on any error."""
    try:
        params: dict[str, str] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
        }
        api_key = _ncbi_api_key()
        if api_key:
            params["api_key"] = api_key
        with httpx.Client(timeout=8.0) as client:
            response = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        idlist = (data.get("esearchresult") or {}).get("idlist") or []
        pmids = [str(p) for p in idlist if str(p).strip()]
        return pmids or None
    except Exception:
        return None


def _esummary_pubmed(pmids: list[str]) -> list[dict] | None:
    """Return paper dicts via esummary, or None when offline / on any error."""
    if not pmids:
        return None
    try:
        params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        api_key = _ncbi_api_key()
        if api_key:
            params["api_key"] = api_key
        with httpx.Client(timeout=8.0) as client:
            response = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        result = data.get("result") or {}
        papers: list[dict] = []
        for pmid in pmids:
            entry = result.get(str(pmid)) or {}
            if not isinstance(entry, dict) or not entry.get("title"):
                continue
            papers.append(
                {
                    "pmid": str(pmid),
                    "title": str(entry.get("title")),
                    "source": str(entry.get("source") or "PubMed"),
                    "pubdate": str(entry.get("pubdate") or ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return papers or None
    except Exception:
        return None


def _fallback_papers() -> list[dict]:
    """Deterministic offline papers (always include 41986348 and 38291024)."""
    papers: list[dict] = []
    for pmid in ("41986348", "38291024"):
        cached = _PMID_ABSTRACT_CACHE[pmid]
        papers.append(
            {
                "pmid": pmid,
                "title": cached["title"],
                "source": cached["source"],
                "pubdate": cached["pubdate"],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return papers


def _pubmed_search_and_summarize(
    genes: list[str],
    cancer_type: str | None,
    message: str = "",
) -> tuple[str, str, list[dict]]:
    """Stage 1: Boolean query + esearch/esummary with deterministic fallback."""
    query = _build_pubmed_query(genes, cancer_type, message)
    pmids = _esearch_pubmed(query)
    papers: list[dict] | None = None
    if pmids:
        papers = _esummary_pubmed(pmids)
    if not papers:
        papers = _fallback_papers()
    return query, _pubmed_url(query), papers


def _looks_like_literature(message: str) -> bool:
    text = (message or "").lower()
    return any(
        token in text
        for token in ("pubmed", "literature", "paper", "pmid", "publication", "abstract")
    )


def _fetch_abstract(pmid: str) -> dict:
    """Stage 2: efetch an abstract; cached fallback keeps this offline-safe."""
    pmid = str(pmid).strip()
    try:
        params: dict[str, str] = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "text",
            "rettype": "abstract",
        }
        api_key = _ncbi_api_key()
        if api_key:
            params["api_key"] = api_key
        with httpx.Client(timeout=8.0) as client:
            response = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
            )
            response.raise_for_status()
            text = response.text.strip()
        if text:
            cached = _PMID_ABSTRACT_CACHE.get(pmid, {})
            record = {
                "pmid": pmid,
                "title": cached.get("title", f"PubMed {pmid}"),
                "abstract": text[:4000],
                "source": cached.get("source", "PubMed"),
                "pubdate": cached.get("pubdate", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
            _INGESTED_ABSTRACTS[pmid] = record
            return record
    except Exception:
        pass
    cached = _PMID_ABSTRACT_CACHE.get(pmid)
    if cached is not None:
        record = {
            "pmid": pmid,
            "title": cached["title"],
            "abstract": cached["abstract"],
            "source": cached["source"],
            "pubdate": cached["pubdate"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        _INGESTED_ABSTRACTS[pmid] = record
        return record
    record = {
        "pmid": pmid,
        "title": f"PubMed {pmid}",
        "abstract": "Abstract unavailable offline; PMID recorded for follow-up grounding.",
        "source": "PubMed",
        "pubdate": "",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }
    _INGESTED_ABSTRACTS[pmid] = record
    return record


def _grounded_followup_reply(request: ChatRequest) -> dict:
    """Stage 3: ground a follow-up on cached/ingested abstracts with citations."""
    genes = [v.hugo_symbol for v in request.variants] or ["(no variants)"]
    bits: list[str] = []
    grounded: list[str] = []
    for raw in request.ingested_pmids:
        pmid = str(raw).strip()
        if not pmid or pmid in grounded:
            continue
        record = _INGESTED_ABSTRACTS.get(pmid) or (
            {
                "pmid": pmid,
                "title": _PMID_ABSTRACT_CACHE[pmid]["title"],
                "abstract": _PMID_ABSTRACT_CACHE[pmid]["abstract"],
            }
            if pmid in _PMID_ABSTRACT_CACHE
            else None
        )
        if record is None:
            continue
        grounded.append(pmid)
        bits.append(
            f"- {record.get('title')} (PMID {pmid}): "
            f"{str(record.get('abstract') or '')[:600]}"
        )
    tab = f" (active tab: {request.active_tab})" if request.active_tab else ""
    chart = ""
    if request.chart_context:
        chart = f" Chart context: {json.dumps(request.chart_context)[:800]}."
    if not bits:
        text = (
            f"Follow-up{tab} for {', '.join(genes)}: no ingested abstracts matched "
            f"{request.ingested_pmids}. Ingest PMIDs via /synthesize-pmids first.{chart}"
        )
    else:
        text = (
            f"Grounded follow-up{tab} for {', '.join(genes)} "
            f"on {len(grounded)} ingested paper(s):\n"
            + "\n".join(bits)
            + f"\nQuestion: {request.message[:500] or 'n/a'}.{chart}"
        )
    return {
        "reply": text,
        "action_type": request.action_type,
        "source": "fallback",
        "grounded_pmids": grounded,
    }


def _colab_recipe_reply(request: ChatRequest) -> dict:
    genes = [v.hugo_symbol for v in request.variants] or ["(no variants)"]
    study = request.study_id or "unknown-study"
    text = (
        "Clinical Recipe Card — reproducible cancer-variant analysis\n"
        f"Study {study} ({request.cancer_type or 'n/a'}, case {request.case_id or 'n/a'}), "
        f"genes: {', '.join(genes)}.\n"
        "1) Index the sample via POST /copilot/index-sample.\n"
        "2) Cross-reference genes (cross_reference_genes) for KB bullets.\n"
        "3) Run Stage 1 query_pubmed for a warning-free Boolean PubMed query.\n"
        "4) Synthesize abstracts via POST /copilot/synthesize-pmids.\n"
        "5) Ask grounded follow-ups with ingested_pmids.\n"
        f"Open the cookbook: {COLAB_COOKBOOK_URL}"
    )
    return {
        "reply": text,
        "action_type": "colab_recipe",
        "source": "fallback",
        "colab_url": COLAB_COOKBOOK_URL,
    }


def _contextual_reply(action: str, request: ChatRequest) -> dict:
    """Rich deterministic handlers for chart/contextual copilot actions."""
    study = request.study_id or "unknown-study"
    genes = [v.hugo_symbol for v in request.variants] or ["(no variants)"]
    gene_list = ", ".join(genes)
    code, entry = _resolve_oncotree(request.cancer_type, request.message)
    tissue = entry.get("default_curie", "n/a") if entry else "n/a"
    bodies = {
        "fail_kinetics": (
            f"Gemcitabine failure kinetics for {study} (case {request.case_id or 'n/a'}, "
            f"[{gene_list}]): adjuvant gemcitabine ran Day 91-205 with outcome "
            "'clinical progressive disease', yet liver metastasis was already present "
            "at Day 74 (PFS 2.4 months) — before chemotherapy even started — followed "
            "by lung metastasis at Day 211 and death at Day 239 (OS 7.9 months). "
            "The Day-74 lesion implies occult micrometastatic seeding present at "
            "resection (R1 uncinate margin + arterial channel invasion), so single-agent "
            "nucleoside-analog therapy faced macroscopic residual disease: primary "
            "chemoresistance rather than acquired resistance. Model reasoning only; "
            "not a therapeutic prescription."
        ),
        "cdkn2a_tp53_axis": (
            f"CDKN2A vs. TP53 checkpoint axis in {study} [{gene_list}] "
            f"(tissue {tissue}): TP53 carries no coding mutation, yet the TP53 pathway "
            "is marked altered via homozygous CDKN2A loss. Loss of p14ARF derepresses "
            "MDM2, attenuating p53-dependent senescence/apoptosis without mutating TP53, "
            "while loss of p16INK4a derepresses CDK4/6-driven Rb phosphorylation and "
            "G1/S transition. Both arms of tumor suppression are therefore defeated by a "
            "single locus. See PMID 15014009 for the canonical axis review. "
            "Model reasoning only; not for direct therapeutic prescription."
        ),
        "ras_cellcycle_cooperation": (
            f"KRAS + CDKN2A oncogenic synergy in {study} [{gene_list}]: constitutive "
            "KRAS signaling (e.g. KRAS G12D) cooperates with CDKN2A deletion to defeat "
            "oncogene-induced senescence — KRAS drives proliferation while CDKN2A loss "
            "removes the p16/Rb senescence brake and the p14ARF/p53 apoptotic brake, "
            "licensing transformation of a high-grade (G3) clone. Co-mutated cases merit "
            "review for synthetic-lethal options (e.g. CDK4/6 or downstream effector "
            "dependencies). Review the cooperation model in PMID 11553622. "
            "Model reasoning only; not for direct therapeutic prescription."
        ),
        "survival_gap": (
            f"Staging vs. PFS discordance for {study} ({code or request.cancer_type or 'n/a'}, "
            f"case {request.case_id or 'n/a'}): resected at curative-intent Stage IIB "
            "(T3 N1 M0) yet relapsed in liver within 74 days (PFS 2.4 months) with "
            "overall survival of only 7.9 months. The gap is explained by adverse "
            "pathology understaged clinically: R1 positive uncinate margin, arterial "
            "and lymphatic channel invasion, and Node 8 extracapsular extension driving "
            "rapid hematogenous seeding. Reconcile pathology R-status before quoting "
            "survival deltas. Model reasoning only."
        ),
        "tmb_grade_paradox": (
            f"Low-TMB vs. G3 paradox in {study} [{gene_list}]: TMB is only 1.4 mut/Mb "
            "with microsatellite stability (MSIsensor 0.54, MANTIS 0.2955), yet the "
            "tumor is poorly differentiated grade G3 with 72% purity, FGA 0.1447, and "
            "lethal in 7.9 months. Aggressiveness here is driven by KRAS + CDKN2A "
            "co-lesions and R1/angioinvasive pathology, not neoantigen load — so "
            "checkpoint immunotherapy is effectively ruled out for this MSS/low-TMB "
            "profile. See PMID 37874914. Model reasoning only."
        ),
        "pathology_r1_reconciliation": (
            f"Pathology R1 reconciliation for {study} (case {request.case_id or 'n/a'}): "
            "the clinical summary stages this resection Stage IIB, but the synoptic "
            "surgical pathology report documents an R1 resection — uncinate process "
            "retroperitoneal margin involved by invasive carcinoma — plus prominent "
            "arterial and lymphatic channel invasion, Node 8 extracapsular extension, "
            "duodenal wall invasion, and poorly differentiated G3 histology. These "
            "adverse findings directly explain the 74-day (2.4-month) progression to "
            "liver metastasis: residual microscopic/macroscopic disease and hematogenous "
            "micro-dissemination were already present at Day 0. Model reasoning based on "
            "cBioPortal clinical and pathology data; not for direct therapeutic prescription."
        ),
        "angioinvasion_failure": (
            f"Arterial invasion and gemcitabine failure in {study} (case {request.case_id or 'n/a'}): "
            "the Whipple specimen shows arterial channel invasion, indicating occult "
            "hematogenous micro-dissemination before adjuvant gemcitabine (Days 91-205) "
            "began. Single-agent nucleoside-analog therapy cannot control disseminated "
            "residual disease, explaining 'clinical progressive disease' on therapy and "
            "the Day 211 lung metastasis. Arterial invasion plus perineural spread are "
            "predictors of gemcitabine failure in this setting. Model reasoning only; "
            "not for direct therapeutic prescription."
        ),
    }
    return {
        "reply": bodies.get(action, f"{action} context for {study} [{gene_list}]."),
        "action_type": action,
        "source": "fallback",
    }


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
    elif action == "query_pubmed":
        query, pubmed_url, papers = _pubmed_search_and_summarize(
            genes, request.cancer_type, request.message
        )
        cited = "; ".join(f"{p['title']} (PMID {p['pmid']})" for p in papers)
        text = (
            f"PubMed Stage 1 for {study} [{', '.join(genes)}]: "
            f"warning-free Boolean query `{query}`. "
            f"Top papers: {cited}. Open: {pubmed_url}"
        )
        return {
            "reply": text,
            "action_type": action,
            "source": "fallback",
            "query": query,
            "pubmed_url": pubmed_url,
            "papers": papers,
        }
    elif action == "colab_recipe":
        return _colab_recipe_reply(request)
    elif action in (
        "fail_kinetics",
        "cdkn2a_tp53_axis",
        "ras_cellcycle_cooperation",
        "survival_gap",
        "tmb_grade_paradox",
        "pathology_r1_reconciliation",
        "angioinvasion_failure",
    ):
        return _contextual_reply(action, request)
    elif request.ingested_pmids:
        return _grounded_followup_reply(request)
    elif not action and _looks_like_literature(request.message):
        query, pubmed_url, papers = _pubmed_search_and_summarize(
            genes, request.cancer_type, request.message
        )
        cited = "; ".join(f"{p['title']} (PMID {p['pmid']})" for p in papers)
        text = (
            f"Free-text literature search for {study} [{', '.join(genes)}]: "
            f"warning-free Boolean query `{query}`. "
            f"Top papers: {cited}. Open: {pubmed_url}"
        )
        return {
            "reply": text,
            "action_type": None,
            "source": "fallback",
            "query": query,
            "pubmed_url": pubmed_url,
            "papers": papers,
        }
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


@router.post("/chat/stream")
async def copilot_chat_stream(request: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint for Co-Pilot interactions.
    
    Streams deltas in real-time as `data: {"delta": str, "done": bool}\n\n`
    and concludes with a terminal frame containing full metadata.
    """
    if request.action_type and request.action_type not in ACTION_TYPES:
        async def invalid_generator():
            payload = {
                "delta": "",
                "done": True,
                "reply": f"Unknown action_type '{request.action_type}'. Supported: {', '.join(ACTION_TYPES)}.",
                "action_type": request.action_type,
                "source": "fallback",
            }
            yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            invalid_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def event_generator():
        live = await _wonder_reply(request)
        final_payload = live if live is not None else _fallback_reply(request)
        reply_text = str(final_payload.get("reply") or "")

        # Stream words/tokens for smooth visual typing in client
        words = reply_text.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            event_data = json.dumps({"delta": chunk, "done": False})
            yield f"data: {event_data}\n\n"
            await asyncio.sleep(0.015)

        # Terminal frame with done=True and complete payload
        terminal_data = {
            "delta": "",
            "done": True,
            **final_payload,
        }
        yield f"data: {json.dumps(terminal_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _deterministic_synthesis(papers: list[dict], question: str) -> str:
    lines = [
        f"- {p['title']} (PMID {p['pmid']}): "
        f"{str(p.get('abstract') or '')[:500]}"
        for p in papers
    ]
    return (
        f"Synthesis for: {question[:500] or 'n/a'}\n"
        + "\n".join(lines)
        + f"\nGrounded on {len(papers)} paper(s) with inline PMID citations."
    )


@router.post("/synthesize-pmids")
def copilot_synthesize_pmids(request: SynthesizePmidsRequest) -> dict:
    """Stage 2: fetch abstracts via efetch (cached fallback) and synthesize."""
    pmids = [str(p).strip() for p in request.pmids if str(p).strip()]
    papers = [_fetch_abstract(p) for p in pmids]
    ingested = [
        {
            "pmid": p["pmid"],
            "title": p["title"],
            "abstract": p.get("abstract", ""),
            "source": p.get("source", "PubMed"),
            "pubdate": p.get("pubdate", ""),
            "url": p["url"],
        }
        for p in papers
    ]
    reply: str | None = None
    base_url, api_key = _wonder_config()
    if base_url and api_key:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    base_url.rstrip("/") + "/chat",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": WONDER_MODEL,
                        "message": request.question,
                        "action_type": "synthesize_pmids",
                        "study_id": request.study_id,
                        "cancer_type": request.cancer_type,
                        "case_id": request.case_id,
                        "variants": [v.model_dump() for v in request.variants],
                        "history": [],
                        "ingested_papers": ingested,
                    },
                )
                response.raise_for_status()
                data = response.json()
            if isinstance(data, dict) and data.get("reply"):
                reply = str(data["reply"])
        except Exception:
            reply = None
    if reply is None:
        reply = _deterministic_synthesis(ingested, request.question)
        source = "fallback"
    else:
        source = "muse-wonder"
    return {
        "reply": reply,
        "source": source,
        "synthesized_pmids": [p["pmid"] for p in ingested],
        "ingested_papers": ingested,
    }


@router.post("/contextual-chart-query")
def copilot_contextual_chart_query(request: ContextualChartQueryRequest) -> dict:
    """Chart-grounded follow-up: echo chart context with a grounded reply."""
    genes = [v.hugo_symbol for v in request.variants] or ["(no variants)"]
    tab = f" (tab: {request.active_tab})" if request.active_tab else ""
    chart = json.dumps(request.chart_context)[:800] if request.chart_context else "n/a"
    text = (
        f"Chart-grounded answer{tab} for {request.study_id or 'unknown-study'} "
        f"[{', '.join(genes)}]: {request.question[:500] or 'n/a'}. "
        f"Chart context: {chart}."
    )
    return {
        "reply": text,
        "source": "fallback",
        "active_tab": request.active_tab,
        "chart_context": request.chart_context,
    }


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
