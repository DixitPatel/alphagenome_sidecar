"""Study x Sample x Gene x Tissue knowledge store + batch failure inspector.

Task 1.1 ``index_patient_sample`` joins the frontend sample payload with the
cached AlphaGenome MongoDB documents and persists a Study x Gene store plus
deduplicated Muse Wonder training records.

Task 1.2 ``inspect_batch_failures`` reverse-seeks the last 5 MB of the
pipeline debug log and cross-checks MongoDB absence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = "annotator"
MONGO_COLLECTION = "alphagenome.annotation"

OUTPUT_TYPES = ("RNA_SEQ", "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS")
FAILURE_KEYWORDS = ("FAILED", "ERROR", "REJECTED", "MISMATCH", "UNSUPPORTED")
# Fix 5: only these explicit log lines count as batch-failure evidence.
EXPLICIT_FAILURE_KEYWORDS = ("FAILED", "ERROR", "REJECTED", "MISMATCH")
TAIL_BYTES = 5 * 1024 * 1024

# Fix 4: module-level cached pymongo client (one per process, not per variant).
_MONGO_CLIENT = None


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def kb_dir() -> Path:
    override = os.environ.get("COPILOT_KB_DIR")
    if override:
        return Path(override)
    return _workspace_root() / "logs" / "copilot_kb"


def store_path() -> Path:
    return kb_dir() / "study_gene_knowledge_store.json"


def training_jsonl_path() -> Path:
    return kb_dir() / "muse_wonder_training_samples.jsonl"


def debug_log_path() -> Path:
    override = os.environ.get("ALPHAGENOME_DEBUG_LOG")
    if override:
        return Path(override)
    return _workspace_root() / "logs" / "alphagenome_pipeline_debug.log"


def oncotree_map_path() -> Path:
    override = os.environ.get("ONCOTREE_MAP_PATH")
    if override:
        return Path(override)
    return _workspace_root() / "workstreams" / "contracts" / "oncotree_to_uberon.json"


def load_oncotree_map() -> dict:
    try:
        with open(oncotree_map_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"mappings": {}}


def default_curie_for(cancer_type: str | None) -> str | None:
    if not cancer_type:
        return None
    mappings = load_oncotree_map().get("mappings", {})
    entry = mappings.get(str(cancer_type).upper())
    if isinstance(entry, dict):
        return entry.get("default_curie")
    return None


def _mongo_collection():
    """Return the annotation collection, or None when pymongo/Mongo is unavailable.

    Fix 4: the pymongo client is cached at module level instead of creating
    a new client per variant.
    """
    global _MONGO_CLIENT
    try:
        import pymongo  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        return _MONGO_CLIENT[MONGO_DB][MONGO_COLLECTION]
    except Exception:
        return None


def _mongosh_eval(js: str) -> str | None:
    """Run one mongosh --eval inside the gn-mongo container (Fix 4 fallback).

    Retries once: a single transient `docker exec` hiccup must not flip a
    cached variant to absent_from_mongo.
    """
    for _ in range(2):
        try:
            out = subprocess.check_output(
                ["docker", "exec", "gn-mongo", "mongosh", "--quiet", "--eval", js],
                stderr=subprocess.DEVNULL,
                timeout=30,
            ).decode("utf-8", errors="replace")
        except Exception:
            continue
        return out
    return None


def _mongosh_find_one(variant_id: str) -> dict | None:
    """Fetch one annotation doc via `docker exec gn-mongo mongosh` (Fix 4).

    Used when pymongo is unavailable. Tries the same id keys as the pymongo
    path and parses the printed EJSON back into a dict.
    """
    if not variant_id:
        return None
    literal = json.dumps(str(variant_id))
    js = (
        "JSON.stringify(db.getSiblingDB('annotator')"
        f"['{MONGO_COLLECTION}'].findOne("
        f"({{$or:[{{_id:{literal}}},{{variant:{literal}}},"
        f"{{hg38_variant:{literal}}},{{variant_id:{literal}}}]}})))"
    )
    out = _mongosh_eval(js)
    if not out:
        return None
    text = out.strip().splitlines()[-1].strip()
    if not text or text == "null":
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def _mongosh_count() -> int | None:
    """Count cached annotations via `docker exec gn-mongo mongosh` (Fix 4)."""
    out = _mongosh_eval(
        f"db.getSiblingDB('annotator')['{MONGO_COLLECTION}'].countDocuments({{}})"
    )
    if not out:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def mongo_reachable() -> bool:
    """True when live MongoDB docs are readable (pymongo or mongosh)."""
    collection = _mongo_collection()
    if collection is not None:
        try:
            collection.database.client.admin.command("ping")
            return True
        except Exception:
            pass
    return _mongosh_count() is not None


def mongo_cached_count() -> int:
    """Number of cached AlphaGenome docs (pymongo first, mongosh fallback)."""
    collection = _mongo_collection()
    if collection is not None:
        try:
            return int(collection.count_documents({}))
        except Exception:
            pass
    count = _mongosh_count()
    return int(count) if count is not None else 0


def _unwrap_doc(doc: dict | None) -> dict:
    """Unwrap a nested ``annotation`` dict when present (Fix 4)."""
    if not isinstance(doc, dict):
        return {}
    inner = doc.get("annotation")
    if isinstance(inner, dict):
        merged = dict(doc)
        merged.update(inner)
        return merged
    return doc


def fetch_annotation(variant_id: str) -> dict | None:
    """Fetch one cached AlphaGenome doc trying common id keys.

    Fix 4: queries ``_id`` first and falls back to
    ``docker exec gn-mongo mongosh`` when pymongo is unavailable.
    """
    if not variant_id:
        return None
    collection = _mongo_collection()
    if collection is not None:
        query_keys = ("_id", "variant", "hg38_variant", "variant_id")
        try:
            for key in query_keys:
                try:
                    doc = collection.find_one({key: variant_id})
                except Exception:
                    break
                if doc:
                    return _unwrap_doc(doc)
        except Exception:
            pass
        # pymongo is installed but the lookup failed/empty: still try mongosh
        # only when the collection itself errored (keeps offline tests fast).
    else:
        doc = _mongosh_find_one(variant_id)
        if doc:
            return _unwrap_doc(doc)
        return None
    # pymongo path returned nothing; one mongosh attempt for live parity.
    doc = _mongosh_find_one(variant_id)
    return _unwrap_doc(doc) if doc else None


def mongo_contains(variant_id: str) -> bool:
    return fetch_annotation(variant_id) is not None


def _top_tracks(scores: list[dict], per_type: int = 3) -> dict[str, list[dict]]:
    top: dict[str, list[dict]] = {}
    for output_type in OUTPUT_TYPES:
        rows = [r for r in scores if isinstance(r, dict) and r.get("output_type") == output_type]
        rows.sort(key=lambda r: abs(float(r.get("raw_score") or 0.0)), reverse=True)
        top[output_type] = [
            {
                "track_name": r.get("track_name"),
                "ontology_curie": r.get("ontology_curie"),
                "biosample_name": r.get("biosample_name"),
                "raw_score": r.get("raw_score"),
                "quantile_score": r.get("quantile_score"),
            }
            for r in rows[:per_type]
        ]
    return top


def _load_store() -> dict:
    path = store_path()
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("genes", {})
            data.setdefault("samples", {})
            return data
    except (OSError, ValueError):
        pass
    return {"genes": {}, "samples": {}}


def _save_store(data: dict) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _existing_jsonl_keys() -> set[str]:
    path = training_jsonl_path()
    keys: set[str] = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                key = rec.get("dedup_key")
                if key:
                    keys.add(key)
    except OSError:
        pass
    return keys


def _jsonl_top_tracks_empty(record: dict) -> bool:
    """True when a JSONL record carries no real MongoDB top tracks."""
    top = record.get("top_tracks")
    if not isinstance(top, dict) or not top:
        return True
    return not any(
        isinstance(tracks, list) and len(tracks) > 0 for tracks in top.values()
    )


def _record_has_tracks(record: dict) -> bool:
    return not _jsonl_top_tracks_empty(record)


def index_patient_sample(
    study_id: str,
    cancer_type: str,
    case_id: str,
    variants: list[dict],
) -> dict:
    """Index one patient sample into the Study x Gene knowledge store.

    Joins each ``{variant_id, hugo_symbol, protein_change}`` entry with its
    cached MongoDB AlphaGenome document and persists ``genes{}``/``samples{}``
    plus deduplicated JSONL training records.
    """
    variants = variants or []
    store = _load_store()
    genes = store["genes"]
    samples = store["samples"]
    default_curie = default_curie_for(cancer_type)

    sample_key = f"{study_id}::{case_id}"
    indexed_variants: list[dict] = []
    genes_indexed: list[str] = []

    for entry in variants:
        if not isinstance(entry, dict):
            continue
        variant_id = str(
            entry.get("variant_id")
            or entry.get("variantId")
            or entry.get("variant")
            or ""
        )
        hugo_symbol = str(
            entry.get("hugo_symbol")
            or entry.get("hugoGeneSymbol")
            or entry.get("gene")
            or "UNKNOWN"
        )
        protein_change = entry.get("protein_change", entry.get("proteinChange"))
        doc = fetch_annotation(variant_id) if variant_id else None
        # Fix 4: unwrap a nested 'annotation' dict when present so scores,
        # canonical_gene, hg38_variant, score_level and merged_splicing_score
        # populate the KB store and JSONL records.
        doc = _unwrap_doc(doc)
        scores = doc.get("scores") if isinstance(doc.get("scores"), list) else []

        record = {
            "variant_id": variant_id,
            "hugo_symbol": hugo_symbol,
            "protein_change": protein_change,
            "canonical_gene": doc.get("canonical_gene") or hugo_symbol,
            "hg38_variant": doc.get("hg38_variant") or variant_id,
            "score_level": doc.get("score_level"),
            "merged_splicing_score": doc.get("merged_splicing_score"),
            "default_curie": default_curie,
            "default_curie_scores": [
                r for r in scores if r.get("ontology_curie") == default_curie
            ]
            if default_curie
            else [],
            "top_tracks": _top_tracks(scores),
        }
        indexed_variants.append(record)
        if hugo_symbol not in genes_indexed:
            genes_indexed.append(hugo_symbol)

        gene_entry = genes.setdefault(
            hugo_symbol,
            {"studies": [], "samples": [], "variants": [], "top_tracks": {}},
        )
        study_ref = {"study_id": study_id, "cancer_type": cancer_type}
        if study_ref not in gene_entry["studies"]:
            gene_entry["studies"].append(study_ref)
        if sample_key not in gene_entry["samples"]:
            gene_entry["samples"].append(sample_key)
        # Fix 7b: deduplicate gene_entry["variants"] by
        # (study_id, case_id, variant_id), updating in place.
        variant_ref = {
            "study_id": study_id,
            "case_id": case_id,
            "variant_id": variant_id,
            "protein_change": protein_change,
            "score_level": record["score_level"],
            "merged_splicing_score": record["merged_splicing_score"],
        }
        dedup_tuple = (study_id, case_id, variant_id)
        replaced = False
        for i, existing_variant in enumerate(gene_entry["variants"]):
            if (
                isinstance(existing_variant, dict)
                and (existing_variant.get("study_id"), existing_variant.get("case_id"), existing_variant.get("variant_id")) == dedup_tuple
            ):
                gene_entry["variants"][i] = variant_ref
                replaced = True
                break
        if not replaced:
            gene_entry["variants"].append(variant_ref)
        for output_type, tracks in record["top_tracks"].items():
            gene_entry["top_tracks"].setdefault(output_type, [])
            seen = {
                (t.get("ontology_curie"), t.get("track_name"))
                for t in gene_entry["top_tracks"][output_type]
            }
            for track in tracks:
                key = (track.get("ontology_curie"), track.get("track_name"))
                if key not in seen:
                    gene_entry["top_tracks"][output_type].append(track)
                    seen.add(key)

    samples[sample_key] = {
        "study_id": study_id,
        "cancer_type": cancer_type,
        "case_id": case_id,
        "default_curie": default_curie,
        "variants": indexed_variants,
    }
    _save_store(store)

    # Fix 7b: rewrite/update existing JSONL records when a previously empty
    # record now has real MongoDB top tracks; otherwise append new records.
    jsonl_path = training_jsonl_path()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    key_to_index: dict[str, int] = {}
    try:
        with open(jsonl_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                existing_lines.append(line if line.endswith("\n") else line + "\n")
                try:
                    rec = json.loads(stripped)
                except ValueError:
                    continue
                key = rec.get("dedup_key")
                if key and key not in key_to_index:
                    key_to_index[key] = len(existing_lines) - 1
    except OSError:
        pass
    appended = 0
    updated = 0
    pending: list[str] = []
    for record in indexed_variants:
        dedup_key = f"{study_id}::{case_id}::{record['hugo_symbol']}::{record['variant_id']}"
        payload = {
            "dedup_key": dedup_key,
            "study_id": study_id,
            "cancer_type": cancer_type,
            "case_id": case_id,
            "hugo_symbol": record["hugo_symbol"],
            "variant_id": record["variant_id"],
            "protein_change": record["protein_change"],
            "canonical_gene": record["canonical_gene"],
            "hg38_variant": record["hg38_variant"],
            "score_level": record["score_level"],
            "merged_splicing_score": record["merged_splicing_score"],
            "default_curie": record["default_curie"],
            "top_tracks": record["top_tracks"],
        }
        if dedup_key in key_to_index:
            idx = key_to_index[dedup_key]
            try:
                old_rec = json.loads(existing_lines[idx].strip())
            except ValueError:
                old_rec = {}
            if _jsonl_top_tracks_empty(old_rec) and _record_has_tracks(payload):
                existing_lines[idx] = json.dumps(payload) + "\n"
                updated += 1
            continue
        pending.append(json.dumps(payload) + "\n")
        key_to_index[dedup_key] = len(existing_lines) + len(pending) - 1
        appended += 1
    if updated:
        with open(jsonl_path, "w") as f:
            f.writelines(existing_lines)
            f.writelines(pending)
    elif pending:
        with open(jsonl_path, "a") as f:
            f.writelines(pending)

    return {
        "indexed_count": len(indexed_variants),
        "genes_indexed": sorted(genes_indexed),
        "jsonl_appended": appended,
        "jsonl_updated": updated,
        "kb_stats": {
            "studies": len({s.get("study_id") for s in samples.values() if isinstance(s, dict)}),
            "samples": len(samples),
            "genes": len(genes),
        },
    }


def _classify_cause(line: str) -> str:
    upper = line.upper()
    if (
        "LIFTOVER REJECTED" in upper
        or "LIFTOVER_FAILED" in upper
        or "LIFTOVER FAILED" in upper
        or ("LIFTOVER" in upper and ("FAIL" in upper or "REJECT" in upper))
    ):
        return "liftover_failed"
    if "MISMATCH" in upper or "REF_CHECK" in upper or "REF CHECK" in upper:
        return "ref_check_mismatch"
    if "UNSUPPORTED" in upper or "PRE-FILTER" in upper or "PRE_FILTER" in upper:
        return "unsupported_variant"
    return "sidecar_error"


def _is_failure_line(line: str, variant_id: str) -> bool:
    if variant_id not in line:
        return False
    upper = line.upper()
    # Explicit informational and success stages are never failures
    if "BEFORE LIFTOVER" in upper:
        return False
    if any(s in upper for s in ("PASSED", "SUCCESS", "NORMALIZATION COMPLETE")):
        return False
    # Strip harmless errorMessage='null' parameter logging before keyword check
    cleaned = re.sub(r"errorMessage=['\"]null['\"]", "", line, flags=re.IGNORECASE)
    cleaned_upper = cleaned.upper()
    return any(k in cleaned_upper for k in EXPLICIT_FAILURE_KEYWORDS)


def inspect_batch_failures(variant_ids: list[str]) -> list[dict]:
    """Inspect the tail (last 5 MB) of the debug log for batch failures.

    Never reads the whole file: seeks to ``max(0, size - 5MB)`` first so the
    check stays fast even on multi-GB logs.
    """
    wanted = [str(v) for v in (variant_ids or []) if v]
    if not wanted:
        return []
    # Fix 5: absent_from_mongo only counts when MongoDB is actually reachable.
    # Never report cached or unqueried variants as sidecar_error while offline.
    reachable = mongo_reachable()
    path = debug_log_path()
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    chunk = ""
    if size is not None:
        try:
            with open(path, "rb") as f:
                f.seek(max(0, size - TAIL_BYTES))
                chunk = f.read().decode("utf-8", errors="replace")
        except OSError:
            chunk = ""
    failures: list[dict] = []
    chunk_lines = chunk.splitlines() if chunk else []
    for variant_id in wanted:
        # If Mongo is reachable and variant is present, it succeeded and is cached
        in_mongo = reachable and mongo_contains(variant_id)
        if in_mongo:
            continue

        hits = [
            line.strip()
            for line in chunk_lines
            if _is_failure_line(line, variant_id)
        ]
        absent = reachable and not in_mongo
        if hits or absent:
            cause = _classify_cause(hits[-1]) if hits else "sidecar_error"
            failures.append(
                {
                    "variant_id": variant_id,
                    "cause": cause,
                    "absent_from_mongo": absent,
                    "evidence": hits[-1][:2000] if hits else None,
                }
            )
    return failures


def kb_counts() -> dict:
    store = _load_store()
    samples = store.get("samples", {})
    genes = store.get("genes", {})
    try:
        with open(training_jsonl_path()) as f:
            jsonl_count = sum(1 for line in f if line.strip())
    except OSError:
        jsonl_count = 0
    studies = {s.get("study_id") for s in samples.values() if isinstance(s, dict)}
    return {
        "indexed_studies_count": len(studies),
        "indexed_genes_count": len(genes),
        "jsonl_training_records_count": jsonl_count,
    }


CURIE_PATTERN = re.compile(r"^(UBERON|CL):\d+$")
