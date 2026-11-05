"""Skeleton copilot tests (Tester fills/verifies). Offline-safe: mocks MongoDB."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app import sample_trainer
from app.main import app

client = TestClient(app)

SAMPLE_VARIANTS = [
    {"variant_id": "chr17:7674220:G>A", "hugo_symbol": "TP53", "protein_change": "p.R175H"},
    {"variant_id": "chr12:25245350:G>A", "hugo_symbol": "KRAS", "protein_change": "p.G12D"},
]


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setenv("COPILOT_KB_DIR", str(kb))
    log = tmp_path / "debug.log"
    log.write_text("[2026-01-01] [ALPHAGENOME-SIDECAR] [PHASE 4] ERROR chr17:7674220:G>A FAILED liftover\n")
    monkeypatch.setenv("ALPHAGENOME_DEBUG_LOG", str(log))
    monkeypatch.setattr(sample_trainer, "fetch_annotation", lambda vid: None)
    monkeypatch.setattr(sample_trainer, "mongo_contains", lambda vid: False)
    # Offline by default: Mongo unreachable so absent_from_mongo never fires.
    monkeypatch.setattr(sample_trainer, "mongo_reachable", lambda: False)
    return kb


FAKE_DOC = {
    "variant": "chr12:25245350:G>A",
    "hg38_variant": "chr12:25245350:G>A",
    "canonical_gene": "KRAS",
    "score_level": "high",
    "merged_splicing_score": 0.5,
    "scores": [
        {"output_type": "RNA_SEQ", "track_name": "t1", "ontology_curie": "UBERON:0001264",
         "biosample_name": "pancreas", "raw_score": 0.9, "quantile_score": 0.99},
        {"output_type": "SPLICE_JUNCTIONS", "track_name": "j1", "ontology_curie": "CL:0000000",
         "biosample_name": "x", "raw_score": 0.1, "quantile_score": 0.5},
    ],
}


def test_copilot_health(isolated_kb):
    response = client.get("/copilot/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "indexed_genes_count" in body


def test_index_sample(isolated_kb):
    response = client.post(
        "/copilot/index-sample",
        json={"study_id": "paad_tcga", "cancer_type": "PAAD", "case_id": "TCGA-1", "variants": SAMPLE_VARIANTS},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["indexed_count"] == 2
    assert "failed_variants" in body
    store = json.loads((isolated_kb / "study_gene_knowledge_store.json").read_text())
    assert "TP53" in store["genes"]


def test_index_sample_dedup_jsonl(isolated_kb):
    payload = {"study_id": "s", "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS}
    client.post("/copilot/index-sample", json=payload)
    client.post("/copilot/index-sample", json=payload)
    lines = (isolated_kb / "muse_wonder_training_samples.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_inspect_batch_failures_tail_seek(isolated_kb):
    start = time.perf_counter()
    failures = sample_trainer.inspect_batch_failures(["chr17:7674220:G>A"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50
    assert failures and failures[0]["variant_id"] == "chr17:7674220:G>A"


@pytest.mark.parametrize("action", ["map_oncotree", "cross_reference_genes", "show_batch_failures", "prep_literature_context"])
def test_chat_actions_offline(isolated_kb, action, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "hi", "action_type": action, "study_id": "s", "cancer_type": "PAAD",
              "case_id": "c", "variants": SAMPLE_VARIANTS, "history": []},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "fallback"


def test_save_mapping_roundtrip(isolated_kb, tmp_path, monkeypatch):
    mapping = tmp_path / "oncotree_to_uberon.json"
    mapping.write_text(json.dumps({"mappings": {}}))
    monkeypatch.setenv("ONCOTREE_MAP_PATH", str(mapping))
    response = client.post(
        "/copilot/save-mapping",
        json={"oncotree_code": "PAAD", "oncotree_name": "Pancreatic Adenocarcinoma",
              "default_curie": "UBERON:0001264", "default_label": "pancreas", "alternates": []},
    )
    assert response.status_code == 200
    assert response.json()["saved"] is True


def test_score_contract_unchanged():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inspect_offline_never_flags_absent(isolated_kb):
    # Fix 5: unreachable Mongo + no log hits -> no failures (no false positives).
    failures = sample_trainer.inspect_batch_failures(["chr12:25245350:G>A"])
    assert failures == []


def test_inspect_reachable_flags_absent(isolated_kb, monkeypatch):
    # Fix 5: reachable Mongo + missing variant -> sidecar_error failure.
    monkeypatch.setattr(sample_trainer, "mongo_reachable", lambda: True)
    failures = sample_trainer.inspect_batch_failures(["chr12:25245350:G>A"])
    assert len(failures) == 1
    assert failures[0]["absent_from_mongo"] is True
    assert failures[0]["cause"] == "sidecar_error"


def test_unwrap_nested_annotation(isolated_kb, monkeypatch):
    nested = {"annotation": dict(FAKE_DOC), "extra": 1}
    monkeypatch.setattr(sample_trainer, "fetch_annotation", lambda vid: nested)
    result = sample_trainer.index_patient_sample("s", "PAAD", "c", [SAMPLE_VARIANTS[1]])
    assert result["indexed_count"] == 1
    store = json.loads((isolated_kb / "study_gene_knowledge_store.json").read_text())
    variants = store["samples"]["s::c"]["variants"]
    assert variants[0]["canonical_gene"] == "KRAS"
    assert variants[0]["merged_splicing_score"] == 0.5
    assert variants[0]["top_tracks"]["RNA_SEQ"], "nested scores must populate top_tracks"


def test_variant_dedup_across_index_calls(isolated_kb, monkeypatch):
    # Fix 7b: re-indexing the same sample must not duplicate gene variants.
    monkeypatch.setattr(sample_trainer, "fetch_annotation", lambda vid: dict(FAKE_DOC))
    payload = {"study_id": "s", "cancer_type": "PAAD", "case_id": "c", "variants": [SAMPLE_VARIANTS[1]]}
    client.post("/copilot/index-sample", json=payload)
    client.post("/copilot/index-sample", json=payload)
    store = json.loads((isolated_kb / "study_gene_knowledge_store.json").read_text())
    assert len(store["genes"]["KRAS"]["variants"]) == 1
    lines = (isolated_kb / "muse_wonder_training_samples.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_jsonl_empty_record_upgraded_with_tracks(isolated_kb, monkeypatch):
    # Fix 7b: an empty record is rewritten once real MongoDB tracks arrive.
    payload = {"study_id": "s", "cancer_type": "PAAD", "case_id": "c", "variants": [SAMPLE_VARIANTS[1]]}
    client.post("/copilot/index-sample", json=payload)
    before = (isolated_kb / "muse_wonder_training_samples.jsonl").read_text().strip().splitlines()
    assert len(before) == 1
    assert json.loads(before[0])["top_tracks"]["RNA_SEQ"] == []
    monkeypatch.setattr(sample_trainer, "fetch_annotation", lambda vid: dict(FAKE_DOC))
    monkeypatch.setattr(sample_trainer, "mongo_contains", lambda vid: True)
    result = sample_trainer.index_patient_sample("s", "PAAD", "c", [SAMPLE_VARIANTS[1]])
    assert result["jsonl_updated"] == 1
    after = (isolated_kb / "muse_wonder_training_samples.jsonl").read_text().strip().splitlines()
    assert len(after) == 1
    rec = json.loads(after[0])
    assert rec["top_tracks"]["RNA_SEQ"][0]["ontology_curie"] == "UBERON:0001264"
    assert rec["merged_splicing_score"] == 0.5


def test_variant_ref_camelcase_aliases(isolated_kb, monkeypatch):
    # Fix 2: camelCase VariantRef payloads validate and index.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/index-sample",
        json={"study_id": "s", "cancer_type": "PAAD", "case_id": "c",
              "variants": [{"variantId": "chr12:25245350:G>A", "hugoGeneSymbol": "KRAS",
                            "proteinChange": "p.G12D"}]},
    )
    assert response.status_code == 200
    assert response.json()["indexed_count"] == 1
    store = json.loads((isolated_kb / "study_gene_knowledge_store.json").read_text())
    assert "KRAS" in store["genes"]


def test_chat_action_alias_normalized(isolated_kb, monkeypatch):
    # Fix 1: 'action' alias works and 'chat'/'' normalize to free-text.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "hi", "action": "map_oncotree", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": [], "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action_type"] == "map_oncotree"
    assert body["suggested_mapping"]["oncotree_code"] == "PAAD"
    assert body["suggested_mapping"]["oncotree_name"] == "Pancreatic Adenocarcinoma"

    response = client.post(
        "/copilot/chat",
        json={"message": "hello there", "action_type": "chat", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": [], "history": []},
    )
    assert response.status_code == 200
    assert response.json()["action_type"] is None


def test_map_oncotree_message_code_detection(isolated_kb, monkeypatch):
    # Fix 3: OncoTree code token in the message wins over study cancer_type.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "Find mapping for PANET please", "action_type": "map_oncotree",
              "study_id": "s", "cancer_type": "PAAD", "case_id": "c",
              "variants": [], "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_mapping"]["oncotree_code"] == "PANET"


def test_map_oncotree_name_match(isolated_kb, monkeypatch):
    # Fix 3: full oncotree_name resolves to the code entry.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "map it", "action_type": "map_oncotree", "study_id": "s",
              "cancer_type": "Pancreatic Adenocarcinoma", "case_id": "c",
              "variants": [], "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_mapping"]["oncotree_code"] == "PAAD"
    assert body["suggested_mapping"]["oncotree_name"] == "Pancreatic Adenocarcinoma"


def test_rich_kb_formatting(isolated_kb, monkeypatch):
    # Fix 6: xref + lit-context replies carry real KB bullets and briefs.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    monkeypatch.setattr(sample_trainer, "fetch_annotation", lambda vid: dict(FAKE_DOC))
    client.post("/copilot/index-sample",
                json={"study_id": "s", "cancer_type": "PAAD", "case_id": "c",
                      "variants": [SAMPLE_VARIANTS[1]]})
    xref = client.post(
        "/copilot/chat",
        json={"message": "x", "action_type": "cross_reference_genes", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": [SAMPLE_VARIANTS[1]],
              "history": []},
    ).json()
    assert "- KRAS:" in xref["reply"]
    assert "UBERON:0001264" in xref["reply"]
    assert "merged_splicing_score" in xref["reply"]
    brief = client.post(
        "/copilot/chat",
        json={"message": "x", "action_type": "prep_literature_context", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": [SAMPLE_VARIANTS[1]],
              "history": []},
    ).json()
    assert "Literature Foundation Brief" in brief["reply"]
    assert "dominant modality" in brief["reply"]
    assert "PubMed search" in brief["reply"]


def test_inspect_batch_failures_5mb_reverse_seek(tmp_path, monkeypatch):
    tail_bytes = sample_trainer.TAIL_BYTES
    assert tail_bytes == 5 * 1024 * 1024
    log = tmp_path / "big_debug.log"
    head_variant = "chr1:100:G>A"
    tail_variant = "chr17:7674220:G>A"
    filler = ("x" * 120 + "\n").encode()
    with open(log, "wb") as f:
        f.write(
            f"[2026-01-01] [ALPHAGENOME-SIDECAR] [PHASE 4] ERROR {head_variant} FAILED liftover\n".encode()
        )
        for _ in range(tail_bytes // len(filler) + 100):
            f.write(filler)
        f.write(
            f"[2026-01-01] [ALPHAGENOME-SIDECAR] [PHASE 4] ERROR {tail_variant} FAILED liftover\n".encode()
        )
    assert log.stat().st_size > tail_bytes
    monkeypatch.setenv("ALPHAGENOME_DEBUG_LOG", str(log))
    monkeypatch.setattr(sample_trainer, "mongo_reachable", lambda: True)
    monkeypatch.setattr(sample_trainer, "mongo_contains", lambda vid: False)
    start = time.perf_counter()
    failures = sample_trainer.inspect_batch_failures([head_variant, tail_variant])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50
    by_id = {item["variant_id"]: item for item in failures}
    assert "FAILED" in (by_id[tail_variant]["evidence"] or "")
    assert by_id[head_variant]["evidence"] is None


def test_chat_unknown_action_fallback(isolated_kb, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "hi", "action_type": "bogus_action", "study_id": "s", "cancer_type": "PAAD",
              "case_id": "c", "variants": SAMPLE_VARIANTS, "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert "bogus_action" in body["reply"]


def test_save_mapping_persists_entry(isolated_kb, tmp_path, monkeypatch):
    mapping = tmp_path / "oncotree_to_uberon.json"
    mapping.write_text(json.dumps({"mappings": {}}))
    monkeypatch.setenv("ONCOTREE_MAP_PATH", str(mapping))
    response = client.post(
        "/copilot/save-mapping",
        json={"oncotree_code": "PAAD", "oncotree_name": "Pancreatic Adenocarcinoma",
              "default_curie": "UBERON:0001264", "default_label": "pancreas", "alternates": ["CL:0000000"]},
    )
    assert response.status_code == 200
    assert response.json()["saved"] is True
    persisted = json.loads(mapping.read_text())["mappings"]["PAAD"]
    assert persisted["default_curie"] == "UBERON:0001264"
    assert persisted["alternates"] == ["CL:0000000"]


def test_inspect_batch_failures_no_false_positive_on_null_error_or_before_liftover(tmp_path, monkeypatch):
    log = tmp_path / "debug.log"
    variant = "17:g.41245612T>C"
    log.write_text(
        f"[2026-11-02 13:48:28.011] [GN-BACKEND] [PHASE 1: BEFORE LIFTOVER] Input variant='{variant}', query='17,41245612,41245612,T,C', assembly='GRCh37', chrom='17', start=41245612, end=41245612, alleles='T/C', strand=1, successfullyAnnotated=true, errorMessage='null'\n"
        f"[2026-11-02 13:48:28.012] [GN-BACKEND] [PHASE 1: PRE-FILTER PASSED] Variant '{variant}' passed pre-filter: contig='chr17', ref='T', alt='C'\n"
        f"[2026-11-02 13:48:28.314] [GN-BACKEND] [PHASE 2: GRCH38 REF CHECK PASSED] Variant '{variant}' PASSED REF check ('T' matches reference 'T')\n"
        f"[2026-11-02 13:48:28.316] [GN-BACKEND] [PHASE 2: NORMALIZATION COMPLETE] Variant '{variant}' successfully normalized to: chr17:43093595:T>C (geneId='ENSG00000012048', geneSymbol='BRCA1')\n"
        f"[2026-11-02 13:48:32.534] [GN-BACKEND] [PHASE 6: ENRICHER SUCCESS] Variant '{variant}' successfully enriched with AlphaGenomeAnnotation\n"
    )
    monkeypatch.setenv("ALPHAGENOME_DEBUG_LOG", str(log))
    monkeypatch.setattr(sample_trainer, "mongo_reachable", lambda: False)
    failures = sample_trainer.inspect_batch_failures([variant])
    assert failures == []


def _force_pubmed_offline(monkeypatch):
    """Force deterministic PubMed fallback (no network in tests)."""
    import httpx

    from app import copilot_router

    monkeypatch.setattr(copilot_router, "_esearch_pubmed", lambda *a, **k: None)
    monkeypatch.setattr(copilot_router, "_esummary_pubmed", lambda *a, **k: None)

    class _DeadClient:
        def __init__(self, *a, **k):
            raise ConnectionError("offline")

    monkeypatch.setattr(httpx, "Client", _DeadClient)


def test_stage1_query_pubmed_fallback(isolated_kb, monkeypatch):
    from app import copilot_router

    assert "query_pubmed" in copilot_router.ACTION_TYPES
    assert "[mh]" not in copilot_router._build_pubmed_query(["KRAS"], "PAAD").lower().replace("[tiab]", "")
    monkeypatch.setattr(copilot_router, "_icite_metrics", lambda *a, **k: {})
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    _force_pubmed_offline(monkeypatch)
    response = client.post(
        "/copilot/chat",
        json={"message": "search", "action_type": "query_pubmed", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS,
              "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    # The raw Boolean query is deliberately NOT returned to the client.
    assert "query" not in body
    # No variant is scored and none is a splice change, so there are no
    # context genes to build a query from.
    assert body["genes"] == []
    pmids = [p["pmid"] for p in body["papers"]]
    assert "41986348" in pmids and "38291024" in pmids
    for paper in body["papers"]:
        assert set(("pmid", "title", "source", "pubdate", "url")) <= set(paper)


def test_stage1_free_text_literature(isolated_kb, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    _force_pubmed_offline(monkeypatch)
    from app import copilot_router

    monkeypatch.setattr(copilot_router, "_icite_metrics", lambda *a, **k: {})
    response = client.post(
        "/copilot/chat",
        json={"message": "find literature on KRAS for this case", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS,
              "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert "query" not in body
    assert "papers" in body and "pubmed_url" in body
    assert "latest" in body and "cited" in body
    assert any(p["pmid"] == "41986348" for p in body["papers"])


def test_stage2_synthesize_pmids_cached_fallback(isolated_kb, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    _force_pubmed_offline(monkeypatch)
    response = client.post(
        "/copilot/synthesize-pmids",
        json={"pmids": ["41986348", "38291024", "15014009", "11553622", "37874914"],
              "question": "How do KRAS and TP53 cooperate?",
              "study_id": "s", "case_id": "c", "cancer_type": "PAAD",
              "variants": SAMPLE_VARIANTS},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert body["synthesized_pmids"] == ["41986348", "38291024", "15014009", "11553622", "37874914"]
    assert len(body["ingested_papers"]) == 5
    assert all(p["abstract"] for p in body["ingested_papers"])
    assert "41986348" in body["reply"] and "38291024" in body["reply"]


def test_stage3_grounded_followup(isolated_kb, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    _force_pubmed_offline(monkeypatch)
    client.post(
        "/copilot/synthesize-pmids",
        json={"pmids": ["41986348"], "question": "q", "study_id": "s",
              "case_id": "c", "cancer_type": "PAAD", "variants": SAMPLE_VARIANTS},
    )
    response = client.post(
        "/copilot/chat",
        json={"message": "summarize the ingested evidence", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS,
              "history": [], "ingested_pmids": ["41986348"],
              "active_tab": "literature", "chart_context": {"chart": "tmb"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert body["grounded_pmids"] == ["41986348"]
    assert "PMID 41986348" in body["reply"]


CONTEXTUAL_ACTIONS = ["fail_kinetics", "cdkn2a_tp53_axis", "ras_cellcycle_cooperation",
                      "survival_gap", "tmb_grade_paradox",
                      "pathology_r1_reconciliation", "angioinvasion_failure"]


@pytest.mark.parametrize("action", CONTEXTUAL_ACTIONS)
def test_contextual_actions_offline(isolated_kb, action, monkeypatch):
    from app import copilot_router

    assert action in copilot_router.ACTION_TYPES
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat",
        json={"message": "context", "action_type": action, "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS,
              "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert body["action_type"] == action
    assert len(body["reply"]) > 50


def test_contextual_chart_query(isolated_kb):
    response = client.post(
        "/copilot/contextual-chart-query",
        json={"question": "why does this bar spike?", "study_id": "s",
              "cancer_type": "PAAD", "case_id": "c", "variants": SAMPLE_VARIANTS,
              "active_tab": "tmb", "chart_context": {"chart": "tmb", "peak": 42}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert body["active_tab"] == "tmb"
    assert body["chart_context"] == {"chart": "tmb", "peak": 42}
    assert "spike" in body["reply"]


def test_inspect_batch_failures_detects_real_liftover_and_ref_check_failures(tmp_path, monkeypatch):
    log = tmp_path / "debug.log"
    v_lift = "1:g.100A>T"
    v_ref = "1:g.200C>A"
    v_pre = "MT:g.300A>G"
    v_side = "1:g.400T>C"
    log.write_text(
        f"[2026-11-02 02:00:00.000] [GN-BACKEND] [PHASE 2: LIFTOVER REJECTED] Variant '{v_lift}' rejected: liftOver returned null\n"
        f"[2026-11-02 02:00:01.000] [GN-BACKEND] [PHASE 2: GRCH38 REF CHECK REJECTED] Variant '{v_ref}' REJECTED: REF check mismatch! Expected 'G' != Got 'C'\n"
        f"[2026-11-02 02:00:02.000] [GN-BACKEND] [PHASE 1: PRE-FILTER REJECTED] Variant '{v_pre}' rejected: unmapped chromosome 'MT'\n"
        f"[2026-11-02 02:00:03.000] [GN-BACKEND] [PHASE 5: RESPONSE ERROR] AlphaGenome sidecar fetch failed for [{v_side}]: Connection reset\n"
    )
    monkeypatch.setenv("ALPHAGENOME_DEBUG_LOG", str(log))
    monkeypatch.setattr(sample_trainer, "mongo_reachable", lambda: False)
    failures = sample_trainer.inspect_batch_failures([v_lift, v_ref, v_pre, v_side])
    by_id = {f["variant_id"]: f for f in failures}
    assert by_id[v_lift]["cause"] == "liftover_failed"
    assert by_id[v_ref]["cause"] == "ref_check_mismatch"
    assert by_id[v_pre]["cause"] == "unsupported_variant"
    assert by_id[v_side]["cause"] == "sidecar_error"


def test_copilot_chat_stream_sse(isolated_kb, monkeypatch):
    # Hermetic SSE framing check: force the offline fallback path so the live
    # Wonder call (when key + network are present) cannot change the source.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_WONDER_API_URL", raising=False)
    response = client.post(
        "/copilot/chat/stream",
        json={
            "message": "tell me about KRAS",
            "study_id": "paad_tcga_pan_can_atlas_2018",
            "cancer_type": "PAAD",
            "variants": SAMPLE_VARIANTS,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line.strip() for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) >= 2
    events = [json.loads(line[len("data: ") :]) for line in lines]
    # First events have done=False and non-empty delta
    assert events[0]["done"] is False
    assert len(events[0]["delta"]) > 0
    # Final event has done=True
    terminal = events[-1]
    assert terminal["done"] is True
    assert "reply" in terminal
    assert terminal["source"] == "fallback"
