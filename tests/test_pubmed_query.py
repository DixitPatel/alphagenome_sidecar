"""Pure tests for the PubMed query builder and its ranking helpers.

Background: the original builder put the sample's cancer type and the
"modality" words into OR groups alongside generic terms ("tumor", "variant",
"genomics"). An OR group is satisfied by any member, so both clauses stopped
filtering -- the modality group alone matched ~910k PubMed records. These tests
pin the AND structure that replaced it.
"""

from dataclasses import dataclass

from app.copilot_router import (
    build_pubmed_query,
    cancer_terms,
    context_genes,
    rank_most_cited,
)


@dataclass
class V:
    hugo_symbol: str
    protein_change: str | None = None
    score_level: str | None = None
    merged_splicing_score: float | None = None


# --- cancer_terms ---------------------------------------------------------

def test_cancer_terms_expands_a_real_type():
    terms = cancer_terms("Lung Adenocarcinoma")
    assert "LUAD" in terms and "NSCLC" in terms


def test_cancer_terms_is_case_insensitive():
    assert cancer_terms("lung adenocarcinoma") == cancer_terms("Lung Adenocarcinoma")


def test_cancer_terms_drops_study_level_placeholders():
    # pancan_pcawg_2020 reports "mixed" at study level. It is not a PubMed
    # term, so the clause must be dropped rather than emitted.
    for placeholder in ("mixed", "Mixed Cancer Types", "pan-cancer", "NOS", ""):
        assert cancer_terms(placeholder) == ()


def test_cancer_terms_passes_through_an_unmapped_type():
    assert cancer_terms("Thymoma") == ("Thymoma",)


# --- context_genes --------------------------------------------------------

def test_context_genes_keeps_only_scored_genes():
    variants = [
        V("DNAH8", "W2211*"),                       # unscored passenger
        V("TTC13", "X434_splice", "strong", 0.9),
        V("MET", "X1010_splice", "moderate", 0.34),
        V("HLA-A", "Q165*", "none", 0.0),           # scored, but not strong/moderate
    ]
    genes = [g for g, _ in context_genes(variants)]
    assert genes == ["TTC13", "MET"]


def test_context_genes_carries_per_gene_splice_flags():
    variants = [V("TTC13", "X434_splice", "strong", 0.9),
                V("KRAS", "G12D", "moderate", 0.0)]
    assert dict(context_genes(variants)) == {"TTC13": True, "KRAS": False}


def test_context_genes_falls_back_to_splice_annotations_when_unscored():
    variants = [V("DNAH8", "W2211*"), V("TTC13", "X434_splice")]
    assert context_genes(variants) == [("TTC13", True)]


def test_context_genes_is_capped():
    from app.copilot_router import _GENE_CONTEXT_MAX
    variants = [V(f"G{i}", "X1_splice", "strong", 1.0 - i / 100) for i in range(20)]
    assert len(context_genes(variants)) == _GENE_CONTEXT_MAX


def test_context_genes_deduplicates_repeated_symbols():
    # MET is listed twice for this sample (two overlapping chr7 deletions).
    variants = [V("MET", "X1010_splice", "moderate", 0.34),
                V("MET", "X1010_splice", "moderate", 0.34)]
    assert context_genes(variants) == [("MET", True)]


def test_context_genes_empty_without_variants():
    assert context_genes([]) == []


# --- build_pubmed_query ---------------------------------------------------

def test_query_ands_three_clauses():
    q = build_pubmed_query("MET", "Lung Adenocarcinoma", True, 0)
    assert q.count(" AND ") == 2
    assert q.startswith("(MET[tiab])")


def test_query_never_emits_mesh_tags():
    q = build_pubmed_query("MET", "Lung Adenocarcinoma", True, 0)
    assert "[mh]" not in q.lower()


def test_cancer_type_is_not_ored_with_generic_terms():
    # The regression guard. A generic oncology word inside the cancer clause
    # would satisfy it on its own and the clause would stop filtering.
    q = build_pubmed_query("MET", "Lung Adenocarcinoma", True, 0)
    cancer_clause = q.split(" AND ")[1]
    for generic in ("tumor", "neoplasm", "carcinoma[tiab]) AND"):
        assert generic not in cancer_clause


def test_modality_clause_excludes_generic_genomics_words():
    # "variant"/"mutation"/"genomics" match nearly the whole cancer-genomics
    # literature, so they must not appear in the modality clause.
    q = build_pubmed_query("MET", "Lung Adenocarcinoma", True, 0)
    modality = q.split(" AND ")[-1]
    for generic in ("variant[tiab]", "mutation[tiab]", "genomics[tiab]"):
        assert generic not in modality


def test_splice_and_coding_modalities_differ():
    splice = build_pubmed_query("MET", "Lung Adenocarcinoma", True, 0)
    coding = build_pubmed_query("MET", "Lung Adenocarcinoma", False, 0)
    assert "exon skipping[tiab]" in splice
    assert "exon skipping[tiab]" not in coding
    assert "missense[tiab]" in coding


def test_placeholder_cancer_type_drops_the_clause_not_the_query():
    q = build_pubmed_query("MET", "mixed", True, 0)
    assert "mixed[tiab]" not in q
    assert q.count(" AND ") == 1  # gene AND modality only


def test_ladder_rungs_drop_the_expected_clause():
    assert "lung adenocarcinoma[tiab]" not in build_pubmed_query("MET", "Lung Adenocarcinoma", True, 1)
    assert "exon skipping[tiab]" not in build_pubmed_query("MET", "Lung Adenocarcinoma", True, 2)


def test_query_is_none_without_a_gene():
    assert build_pubmed_query(None, "Lung Adenocarcinoma", True, 0) is None


def test_rung_two_is_none_when_the_cancer_type_is_a_placeholder():
    assert build_pubmed_query("MET", "mixed", True, 2) is None


# --- rank_most_cited ------------------------------------------------------

def test_rank_orders_by_citation_count():
    metrics = {"1": {"citation_count": 10}, "2": {"citation_count": 900},
               "3": {"citation_count": 50}}
    assert rank_most_cited(["1", "2", "3"], metrics, 3) == ["2", "3", "1"]


def test_rank_treats_missing_metrics_as_zero_without_dropping_them():
    metrics = {"1": {"citation_count": 5}, "2": {"citation_count": None}}
    assert rank_most_cited(["1", "2", "3"], metrics, 3) == ["1", "2", "3"]


# --- regressions found during live testing (2026-11-04) -------------------

def test_oncotree_codes_expand_like_names():
    # ONCOTREE_CODE reaches the sidecar as "LUAD". It previously fell through
    # to the literal ("LUAD",), losing every spelled-out synonym.
    assert cancer_terms("LUAD") == cancer_terms("Lung Adenocarcinoma")
    assert "non-small cell lung" in cancer_terms("LUAD")
    assert "pancreatic ductal adenocarcinoma" in cancer_terms("PAAD")


def test_unknown_code_still_passes_through():
    assert cancer_terms("ZZZZ") == ("ZZZZ",)


def test_gene_clause_accepts_a_list_and_a_bare_string():
    from app.copilot_router import build_pubmed_query as b
    assert b(["MET", "TTC13"], "LUAD", True, 0).startswith("(MET[tiab] OR TTC13[tiab])")
    assert b("MET", "LUAD", True, 0).startswith("(MET[tiab])")


