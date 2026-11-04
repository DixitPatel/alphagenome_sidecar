"""Colab Cookbook cell 4 rendering and the /copilot/colab-params route.

Pure-function tests use one real Mongo record (CHEK2 X336_splice, patient
P-0057054, msk_chord_2024), exported once to tests/data. The live test reads
Mongo and is skipped when Mongo is not reachable. No mocks.
"""

import ast
import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import colab, sample_trainer
from app.main import app

RECORD = json.loads(
    (Path(__file__).parent / "data" / "chek2_x336_mongo.json").read_text()
)
COLON = ["UBERON:0001157"]


def _exec_cell(cell: str) -> dict:
    namespace: dict = {}
    exec(compile(ast.parse(cell), "cell_4", "exec"), namespace)
    return namespace


def test_parse_hg38():
    assert colab.parse_hg38("chr12:25245350:C>T") == ("chr12", 25245350, "C", "T")
    assert colab.parse_hg38("chr3:101857135:TC>T") == ("chr3", 101857135, "TC", "T")


def test_summarize_scores_uses_chosen_tissue_and_splice_sites():
    summary = colab.summarize_scores(RECORD, COLON)
    assert summary["score_level"] == RECORD["score_level"]
    types = [row["output_type"] for row in summary["top"]]
    assert types == [t for t in colab.OUTPUT_ORDER if t in types]
    assert len(types) == len(set(types))
    assert "SPLICE_SITES" in types
    for row in summary["top"]:
        source = [
            r for r in RECORD["scores"]
            if r["output_type"] == row["output_type"] and r["track_name"] == row["track_name"]
        ]
        assert source, row
        if row["output_type"] != "SPLICE_SITES":
            assert source[0]["ontology_curie"] in COLON
        best = max(
            abs(r["raw_score"]) for r in RECORD["scores"]
            if r["output_type"] == row["output_type"] and r["raw_score"] is not None
            and (row["output_type"] == "SPLICE_SITES" or r["ontology_curie"] in COLON)
        )
        assert abs(row["raw_score"]) == pytest.approx(round(best, 4))


def test_summarize_scores_other_tissue_has_no_tissue_rows():
    summary = colab.summarize_scores(RECORD, ["UBERON:0000473"])
    assert [row["output_type"] for row in summary["top"]] == ["SPLICE_SITES"]


def test_render_param_cell_parses_and_round_trips():
    summary = colab.summarize_scores(RECORD, COLON)
    cell = colab.render_param_cell(RECORD["hg38_variant"], "CHEK2", COLON, summary)
    assert cell.startswith("# @title 4. Cancer Variant & Target Tissue Parameters\n")
    values = _exec_cell(cell)
    chromosome, position, reference, alternate = colab.parse_hg38(RECORD["hg38_variant"])
    assert values["variant_chromosome"] == chromosome
    assert values["variant_position"] == position
    assert values["variant_reference_bases"] == reference
    assert values["variant_alternate_bases"] == alternate
    assert values["gene_symbol"] == "CHEK2"
    assert values["ontology_terms"] == COLON
    assert json.loads(values["sidecar_scores"]) == summary


def test_param_cell_matches_notebook_cell_4_fields():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_cookbook", Path(__file__).resolve().parents[1] / "colabs" / "build_cookbook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook_fields = [
        line.split(" = ")[0] for line in builder.CELL_4.splitlines() if "# @param" in line
    ]
    cell = colab.render_param_cell("chr12:25245350:C>T", "KRAS", COLON, {"top": []})
    rendered_fields = [line.split(" = ")[0] for line in cell.splitlines() if "# @param" in line]
    assert rendered_fields == notebook_fields
    assert cell.splitlines()[:2] == builder.CELL_4.splitlines()[:2]


def test_colab_url_follows_env(monkeypatch):
    monkeypatch.delenv("COLAB_NOTEBOOK_REF", raising=False)
    monkeypatch.delenv("COLAB_NOTEBOOK_REPO", raising=False)
    assert colab.colab_url() == (
        "https://colab.research.google.com/github/DixitPatel/alphagenome_sidecar"
        "/blob/w6-muse-copilot/colabs/cancer_variant_cookbook.ipynb"
    )
    monkeypatch.setenv("COLAB_NOTEBOOK_REF", "main")
    monkeypatch.setenv("COLAB_NOTEBOOK_REPO", "someone/fork")
    assert "/github/someone/fork/blob/main/colabs/" in colab.colab_url()


def test_put_body_carries_the_committed_notebook():
    raw = colab.NOTEBOOK_FILE.read_bytes()
    body = colab.put_body(raw, "w6-muse-copilot")
    assert body["branch"] == "w6-muse-copilot"
    assert base64.b64decode(body["content"]) == raw
    assert json.loads(raw)["nbformat"] == 4


def _github_reachable() -> bool:
    try:
        return httpx.get("https://api.github.com", timeout=5).status_code < 500
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _github_reachable(), reason="GitHub not reachable")
def test_ensure_notebook_published_existing_file_live(monkeypatch):
    # Read-only: main has the notebook, so nothing is written.
    colab._PUBLISHED.clear()
    monkeypatch.setenv("COLAB_NOTEBOOK_REF", "main")
    monkeypatch.delenv("COLAB_NOTEBOOK_REPO", raising=False)
    assert colab.ensure_notebook_published() == "exists"


@pytest.mark.skipif(not sample_trainer.mongo_reachable(), reason="Mongo not reachable")
def test_colab_params_live():
    response = TestClient(app).post(
        "/copilot/colab-params",
        json={"variant_id": RECORD["_id"], "oncotree_code": "COAD"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hg38_variant"] == RECORD["hg38_variant"]
    assert body["ontology_terms"] == ["UBERON:0000317"]
    _exec_cell(body["param_cell"])
    assert body["colab_url"].startswith("https://colab.research.google.com/github/")
