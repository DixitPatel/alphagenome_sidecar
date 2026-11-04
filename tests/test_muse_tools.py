"""Tests for colabs/muse_tools.py (Muse Wonder grounding tools).

Pure tests use the committed docs KB and fixed strings. Live tests call the
public APIs and are skipped when NCBI is not reachable. No mocks.
"""

import importlib.util
import json
import urllib.request
from pathlib import Path

import pytest

COLABS_DIR = Path(__file__).resolve().parents[1] / "colabs"
spec = importlib.util.spec_from_file_location("muse_tools", COLABS_DIR / "muse_tools.py")
muse_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muse_tools)

# CHEK2 X336_splice in P-0057054 (GRCh38).
CHEK2 = {"chromosome": "chr22", "position": 28699836, "ref": "A", "alt": "G"}


def _ncbi_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{muse_tools.EUTILS}/einfo.fcgi", timeout=5)
        return True
    except Exception:
        return False


live = pytest.mark.skipif(not _ncbi_reachable(), reason="NCBI not reachable")


def test_tool_specs_match_functions():
    names = [s["function"]["name"] for s in muse_tools.TOOL_SPECS]
    assert sorted(names) == sorted(muse_tools.TOOLS)
    for spec_ in muse_tools.TOOL_SPECS:
        params = spec_["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"])


def test_spdi_matches_only_this_allele():
    assert muse_tools.spdi_matches("NC_000022.11:28699835:A:G", **CHEK2)
    assert not muse_tools.spdi_matches("NC_000022.11:28699835:A:T", **CHEK2)
    assert not muse_tools.spdi_matches("NC_000002.12:28699835:A:G", **CHEK2)
    assert muse_tools.refseq_chromosome("chrX") == "NC_000023."


def test_parse_pubmed_xml_reads_labelled_abstract():
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>
    <Article><Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue>
    </Journal><ArticleTitle>T <i>x</i></ArticleTitle><Abstract>
    <AbstractText Label="AIM">a</AbstractText><AbstractText>b</AbstractText>
    </Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [paper] = muse_tools.parse_pubmed_xml(xml)
    assert paper["id"] == "PMID:1"
    assert paper["title"] == "T x"
    assert paper["year"] == "2020"
    assert paper["abstract"] == "AIM: a b"


def test_docs_kb_finds_splicing_score_section():
    kb = muse_tools.load_docs_kb()
    assert len(kb) > 40
    assert all(s["id"].startswith("AGDocs:") and s["url"] for s in kb)
    hits = muse_tools.search_docs(kb, "how to score splicing variants")
    assert hits[0]["id"] == "AGDocs:faqs#how-to-score-splicing-variants"


def test_run_tool_rejects_unknown_tool_and_drops_extra_args():
    registry = muse_tools.Registry()
    assert "Unknown tool" in muse_tools.run_tool("shell", "{}", registry)
    out = json.loads(muse_tools.run_tool(
        "alphagenome_docs", json.dumps({"query": "strandedness", "rm": "-rf"}), registry))
    assert out["results"]
    assert all(k.startswith("AGDocs:") for k in registry.sources)


def test_registry_collects_nested_ids_and_civic_source_pmids():
    registry = muse_tools.Registry()
    registry.add_from({"results": [
        {"id": "CIViC:EID1", "url": "u", "source_pmid": "PMID:9"},
        {"id": "not an id"},
    ]})
    assert registry.sources == {"CIViC:EID1": "u",
                                "PMID:9": "https://pubmed.ncbi.nlm.nih.gov/9/"}


def test_verify_citations_marks_unknown_ids():
    text = ("Donor lost [FACTS]. VUS [ClinVar:VCV005034536]. "
            "Made up [PMID:1; Gene:11200]. See [link](http://x).")
    out, cited, unverified = muse_tools.verify_citations(
        text, {"ClinVar:VCV005034536", "Gene:11200"})
    assert cited == ["ClinVar:VCV005034536", "Gene:11200"]
    assert unverified == ["PMID:1"]
    assert "[unverified PMID:1; Gene:11200]" in out
    assert "[link](http://x)" in out


@live
def test_clinvar_lookup_exact_allele_live():
    out = muse_tools.clinvar_lookup(**CHEK2)
    first = out["results"][0]
    assert first["exact_match"] is True
    assert first["id"] == "ClinVar:VCV005034536"
    assert sum(r["exact_match"] for r in out["results"]) == 1


@live
def test_gene_summary_and_pubmed_live():
    assert muse_tools.gene_summary("CHEK2")["results"][0]["id"] == "Gene:11200"
    found = muse_tools.search_pubmed("CHEK2 colorectal cancer", max_results=2)
    pmids = [r["id"] for r in found["results"]]
    assert len(pmids) == 2
    papers = muse_tools.fetch_abstracts(pmids)["results"]
    assert [p["id"] for p in papers] == pmids


@live
def test_civic_and_cohort_live():
    out = muse_tools.civic_lookup("KRAS", "G12D", max_items=2)
    evidence = [r for r in out["results"] if r["type"] == "evidence"]
    assert len(evidence) == 2 and evidence[0]["id"].startswith("CIViC:EID")
    cohort = muse_tools.cohort_frequency("CHEK2", "msk_chord_2024", "X336_splice")
    row = cohort["results"][0]
    assert row["samples_in_study"] > 20000
    assert row["samples_with_protein_change"] >= 1
