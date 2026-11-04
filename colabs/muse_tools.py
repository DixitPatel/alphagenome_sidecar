"""Grounding tools for Muse Wonder (cookbook cell 8 and, later, the sidecar).

Wonder asks for a tool; this module runs it and returns JSON. Every result
record carries an "id" that Wonder must cite, for example [PMID:22948721].
The registry keeps the ids returned in one run, and verify_citations() marks
any cited id that no tool returned.

Public sources only: NCBI E-utilities (PubMed, ClinVar, Gene), CIViC,
the public cBioPortal API and the static AlphaGenome docs KB in kb/.
Standard library only, so the notebook needs no extra install.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CIVIC_GRAPHQL = "https://civicdb.org/api/graphql"
CBIOPORTAL_API = "https://www.cbioportal.org/api"
MUSE_URL = "https://api.meta.ai/v1/chat/completions"
DOCS_KB_PATH = Path(__file__).with_name("kb") / "alphagenome_docs.json"
USER_AGENT = "alphagenome-sidecar-muse-tools"
TIMEOUT_S = 20
MAX_RESULT_CHARS = 8000
ID_PREFIXES = ("PMID", "ClinVar", "CIViC", "Gene", "cBioPortal", "AGDocs")
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
ID_RE = re.compile(r"^(?:%s):\S+$" % "|".join(ID_PREFIXES))


# ---------------------------------------------------------------- HTTP

def _get(url: str, params: dict | None = None) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _post_json(url: str, body: dict, headers: dict | None = None,
               timeout: int = TIMEOUT_S) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT,
                 **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _eutils(endpoint: str, **params) -> bytes:
    params["tool"] = USER_AGENT
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    else:
        time.sleep(0.34)  # NCBI allows 3 requests per second without a key.
    return _get(f"{EUTILS}/{endpoint}.fcgi", params)


# ---------------------------------------------------------------- tools

def search_pubmed(query: str, max_results: int = 5) -> dict:
    """PubMed search. Returns id, title, journal and year per paper."""
    max_results = max(1, min(int(max_results), 10))
    found = json.loads(_eutils("esearch", db="pubmed", term=query,
                               retmode="json", retmax=max_results))
    pmids = found.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return {"query": query, "results": []}
    summary = json.loads(_eutils("esummary", db="pubmed", id=",".join(pmids),
                                 retmode="json"))["result"]
    results = []
    for pmid in pmids:
        entry = summary.get(pmid, {})
        results.append({
            "id": f"PMID:{pmid}",
            "title": entry.get("title", ""),
            "journal": entry.get("source", ""),
            "year": entry.get("pubdate", "")[:4],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"query": query, "results": results}


def parse_pubmed_xml(xml_bytes: bytes, max_chars: int = 2500) -> list[dict]:
    """Title, year and abstract per PubmedArticle in an efetch XML reply."""
    papers = []
    for article in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID", default="")
        parts = []
        for node in article.iter("AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        papers.append({
            "id": f"PMID:{pmid}",
            "title": "".join(article.find(".//ArticleTitle").itertext()).strip()
            if article.find(".//ArticleTitle") is not None else "",
            "year": article.findtext(".//JournalIssue/PubDate/Year", default=""),
            "abstract": " ".join(parts)[:max_chars] or "No abstract in PubMed.",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers


def fetch_abstracts(pmids: list) -> dict:
    """Abstracts for up to 5 PMIDs ("PMID:123" or "123")."""
    ids = [str(p).replace("PMID:", "").strip() for p in pmids][:5]
    ids = [p for p in ids if p.isdigit()]
    if not ids:
        return {"results": [], "note": "No valid PMIDs."}
    return {"results": parse_pubmed_xml(
        _eutils("efetch", db="pubmed", id=",".join(ids), retmode="xml"))}


def refseq_chromosome(chromosome: str) -> str:
    """GRCh38 RefSeq accession prefix for a chromosome, e.g. '22' -> 'NC_000022.'."""
    name = str(chromosome).lower().removeprefix("chr").upper()
    number = {"X": 23, "Y": 24}.get(name) or int(name)
    return f"NC_{number:06d}."


def spdi_matches(spdi: str, chromosome: str, position: int, ref: str, alt: str) -> bool:
    """True when a canonical SPDI is this GRCh38 allele. SPDI is 0-based."""
    return (spdi.startswith(refseq_chromosome(chromosome))
            and spdi.endswith(f":{int(position) - 1}:{ref}:{alt}"))


def clinvar_lookup(chromosome: str, position: int, ref: str, alt: str) -> dict:
    """ClinVar records at a GRCh38 position. exact_match marks this allele."""
    chrom = str(chromosome).lower().removeprefix("chr").upper()
    found = json.loads(_eutils("esearch", db="clinvar", retmode="json", retmax=20,
                               term=f"{chrom}[chr] AND {int(position)}[chrpos38]"))
    uids = found.get("esearchresult", {}).get("idlist", [])
    if not uids:
        return {"results": [], "note": "No ClinVar record at this position."}
    summary = json.loads(_eutils("esummary", db="clinvar", id=",".join(uids),
                                 retmode="json"))["result"]
    results = []
    for uid in uids:
        record = summary.get(uid, {})
        spdi = (record.get("variation_set") or [{}])[0].get("canonical_spdi", "")
        germline = record.get("germline_classification") or {}
        oncogenicity = record.get("oncogenicity_classification") or {}
        results.append({
            "id": f"ClinVar:{record.get('accession', uid)}",
            "title": record.get("title", ""),
            "spdi": spdi,
            "exact_match": spdi_matches(spdi, chromosome, position, ref, alt),
            "germline_classification": germline.get("description", ""),
            "review_status": germline.get("review_status", ""),
            "last_evaluated": germline.get("last_evaluated", ""),
            "oncogenicity_classification": oncogenicity.get("description", ""),
            "molecular_consequence": record.get("molecular_consequence_list", []),
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/",
        })
    results.sort(key=lambda r: not r["exact_match"])
    return {"results": results,
            "note": "exact_match false means another allele at the same position."}


CIVIC_GENE_QUERY = """
query($gene: String!) { gene(entrezSymbol: $gene) { id name description } }
"""
CIVIC_VARIANT_QUERY = """
query($geneId: Int!, $name: String!, $first: Int!) {
  variants(geneId: $geneId, name: $name, first: 3) { nodes {
    name
    singleVariantMolecularProfile { name evidenceItems(first: $first) {
      totalCount
      nodes { name status evidenceType evidenceLevel evidenceDirection significance
              disease { name } therapies { name } description
              source { citationId sourceType } } } } } }
}
"""


def civic_lookup(gene: str, variant_name: str = "", max_items: int = 5) -> dict:
    """CIViC gene summary and, with variant_name (e.g. 'G12D'), its evidence."""
    gene_data = _post_json(CIVIC_GRAPHQL, {"query": CIVIC_GENE_QUERY,
                                           "variables": {"gene": gene}})
    record = (gene_data.get("data") or {}).get("gene")
    if not record:
        return {"results": [], "note": f"{gene} is not in CIViC."}
    results = [{
        "id": f"CIViC:GID{record['id']}",
        "type": "gene_summary",
        "text": (record.get("description") or "No CIViC gene summary.")[:2000],
        "url": f"https://civicdb.org/genes/{record['id']}/summary",
    }]
    if variant_name:
        data = _post_json(CIVIC_GRAPHQL, {"query": CIVIC_VARIANT_QUERY, "variables": {
            "geneId": record["id"], "name": variant_name,
            "first": max(1, min(int(max_items), 10))}})
        if data.get("errors"):
            raise RuntimeError(data["errors"][0].get("message", "CIViC error"))
        variants = (((data.get("data") or {}).get("variants") or {}).get("nodes")) or []
        for variant in variants:
            profile = variant.get("singleVariantMolecularProfile") or {}
            items = profile.get("evidenceItems") or {}
            for item in items.get("nodes", []):
                source = item.get("source") or {}
                results.append({
                    "id": f"CIViC:{item['name']}",
                    "type": "evidence",
                    "molecular_profile": profile.get("name"),
                    "status": item.get("status"),
                    "evidence_type": item.get("evidenceType"),
                    "level": item.get("evidenceLevel"),
                    "direction": item.get("evidenceDirection"),
                    "significance": item.get("significance"),
                    "disease": (item.get("disease") or {}).get("name"),
                    "therapies": [t["name"] for t in item.get("therapies") or []],
                    "text": (item.get("description") or "")[:800],
                    "source_pmid": (f"PMID:{source['citationId']}"
                                    if source.get("sourceType") == "PUBMED" else None),
                    "url": f"https://civicdb.org/evidence/{item['name'][3:]}/summary",
                })
        if not variants:
            results.append({"id": f"CIViC:GID{record['id']}", "type": "note",
                             "text": f"No CIViC variant named {variant_name} in {gene}."})
    return {"results": results}


def gene_summary(gene: str) -> dict:
    """NCBI Gene summary for a human gene symbol."""
    found = json.loads(_eutils("esearch", db="gene", retmode="json",
                               term=f"{gene}[sym] AND human[orgn]"))
    uids = found.get("esearchresult", {}).get("idlist", [])
    if not uids:
        return {"results": [], "note": f"No human gene {gene} in NCBI Gene."}
    record = json.loads(_eutils("esummary", db="gene", id=uids[0],
                                retmode="json"))["result"][uids[0]]
    return {"results": [{
        "id": f"Gene:{uids[0]}",
        "symbol": record.get("name"),
        "name": record.get("description"),
        "location": record.get("maplocation"),
        "summary": (record.get("summary") or "No NCBI summary.")[:2500],
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{uids[0]}",
    }]}


def cohort_frequency(gene: str, study_id: str, protein_change: str = "") -> dict:
    """Samples with a mutation in the gene in a public cBioPortal study."""
    study = json.loads(_get(f"{CBIOPORTAL_API}/studies/{urllib.parse.quote(study_id)}"))
    entrez = json.loads(_get(f"{CBIOPORTAL_API}/genes/{urllib.parse.quote(gene)}"))
    mutations = _post_json(
        f"{CBIOPORTAL_API}/molecular-profiles/{study_id}_mutations/mutations/fetch"
        "?projection=SUMMARY",
        {"sampleListId": f"{study_id}_all", "entrezGeneIds": [entrez["entrezGeneId"]]},
        timeout=60,
    )
    samples = {m["sampleId"] for m in mutations}
    by_change: dict[str, set] = {}
    for m in mutations:
        by_change.setdefault(m.get("proteinChange") or "?", set()).add(m["sampleId"])
    top = sorted(by_change.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    total = study.get("allSampleCount") or study.get("sequencedSampleCount")
    result = {
        "id": f"cBioPortal:{study_id}",
        "study": study.get("name"),
        "gene": gene,
        "samples_in_study": total,
        "samples_with_gene_mutation": len(samples),
        "url": f"https://www.cbioportal.org/results/mutations?cancer_study_list="
               f"{study_id}&gene_list={gene}",
        "note": "Counts use all samples in the study as the denominator.",
    }
    if protein_change:
        change = protein_change.removeprefix("p.")
        result["protein_change"] = change
        result["samples_with_protein_change"] = len(by_change.get(change, set()))
    else:
        result["top_protein_changes"] = [
            {"protein_change": k, "samples": len(v)} for k, v in top]
    return {"results": [result]}


_DOCS_KB: list | None = None
WORD_RE = re.compile(r"[a-z0-9_]+")


def load_docs_kb(path: Path = DOCS_KB_PATH) -> list:
    global _DOCS_KB
    if _DOCS_KB is None:
        _DOCS_KB = json.loads(Path(path).read_text(encoding="utf-8"))
    return _DOCS_KB


def search_docs(kb: list, query: str, max_sections: int = 3) -> list[dict]:
    """Keyword match: title hits count 3x, text hits once per distinct term."""
    terms = {t for t in WORD_RE.findall(query.lower()) if len(t) > 2}
    scored = []
    for section in kb:
        title = set(WORD_RE.findall(section["title"].lower()))
        text = set(WORD_RE.findall(section["text"].lower()))
        score = 3 * len(terms & title) + len(terms & text)
        if score:
            scored.append((score, section))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [dict(s) for _, s in scored[:max(1, min(int(max_sections), 5))]]


def alphagenome_docs(query: str, max_sections: int = 3) -> dict:
    """Sections of the AlphaGenome docs that match the query."""
    results = search_docs(load_docs_kb(), query, max_sections)
    return {"results": results} if results else {
        "results": [], "note": "No docs section matches. Try other words."}


# ---------------------------------------------------------------- registry

TOOLS = {
    "search_pubmed": search_pubmed,
    "fetch_abstracts": fetch_abstracts,
    "clinvar_lookup": clinvar_lookup,
    "civic_lookup": civic_lookup,
    "gene_summary": gene_summary,
    "cohort_frequency": cohort_frequency,
    "alphagenome_docs": alphagenome_docs,
}


def _fn(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


_STR = {"type": "string"}
TOOL_SPECS = [
    _fn("search_pubmed", "Search PubMed. Returns PMIDs with titles. Use gene, "
        "variant or mechanism words, e.g. 'CHEK2 splice site colorectal cancer'.",
        {"query": _STR, "max_results": {"type": "integer"}}, ["query"]),
    _fn("fetch_abstracts", "Get abstracts for up to 5 PMIDs found by search_pubmed.",
        {"pmids": {"type": "array", "items": _STR}}, ["pmids"]),
    _fn("clinvar_lookup", "ClinVar records at a GRCh38 position, with the "
        "classification and review status. exact_match marks this allele.",
        {"chromosome": _STR, "position": {"type": "integer"}, "ref": _STR, "alt": _STR},
        ["chromosome", "position", "ref", "alt"]),
    _fn("civic_lookup", "CIViC cancer evidence: gene summary, and evidence items "
        "for a variant name such as 'G12D' when given.",
        {"gene": _STR, "variant_name": _STR, "max_items": {"type": "integer"}}, ["gene"]),
    _fn("gene_summary", "NCBI Gene summary of what a human gene does.",
        {"gene": _STR}, ["gene"]),
    _fn("cohort_frequency", "How many samples in a public cBioPortal study have a "
        "mutation in the gene, and the most common protein changes.",
        {"gene": _STR, "study_id": _STR, "protein_change": _STR}, ["gene", "study_id"]),
    _fn("alphagenome_docs", "Search the AlphaGenome documentation, e.g. what a "
        "score, output type or track means and how it is computed.",
        {"query": _STR, "max_sections": {"type": "integer"}}, ["query"]),
]


class Registry:
    """Ids and URLs returned by tools in one run."""

    def __init__(self):
        self.sources: dict[str, str] = {}

    def add_from(self, payload) -> None:
        if isinstance(payload, dict):
            if isinstance(payload.get("id"), str) and ID_RE.match(payload["id"]):
                self.sources.setdefault(payload["id"], payload.get("url", ""))
            if isinstance(payload.get("source_pmid"), str):
                pmid = payload["source_pmid"]
                self.sources.setdefault(pmid, f"https://pubmed.ncbi.nlm.nih.gov/{pmid[5:]}/")
            for value in payload.values():
                self.add_from(value)
        elif isinstance(payload, list):
            for value in payload:
                self.add_from(value)


def run_tool(name: str, arguments: str | dict, registry: Registry) -> str:
    """Runs one allow-listed tool. Errors come back as JSON, not exceptions."""
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool {name}."})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        allowed = set(next(s for s in TOOL_SPECS if s["function"]["name"] == name)
                      ["function"]["parameters"]["properties"])
        args = {k: v for k, v in args.items() if k in allowed}
        result = TOOLS[name](**args)
    except Exception as error:  # Network or argument errors go back to the model.
        return json.dumps({"error": f"{type(error).__name__}: {error}"[:300]})
    registry.add_from(result)
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " ...[truncated]"


def verify_citations(text: str, allowed: set) -> tuple[str, list, list]:
    """Marks cited ids that no tool returned. Returns (text, cited, unverified)."""
    cited, unverified = [], []

    def check(match):
        tokens = [t.strip() for t in re.split(r"[;,]", match.group(1))]
        if not any(ID_RE.match(t) or t == "FACTS" for t in tokens):
            return match.group(0)  # Not a citation, e.g. a Markdown link label.
        kept = []
        for token in tokens:
            if token == "FACTS" or token in allowed:
                kept.append(token)
                if token != "FACTS" and token not in cited:
                    cited.append(token)
            elif ID_RE.match(token):
                kept.append(f"unverified {token}")
                unverified.append(token)
            else:
                kept.append(token)
        return "[" + "; ".join(kept) + "]"

    return CITATION_RE.sub(check, text), cited, unverified


# ---------------------------------------------------------------- tool loop

def chat_with_tools(messages: list, api_key: str, model: str = "muse-wonder-1.0",
                    reasoning_effort: str = "high", max_rounds: int = 4,
                    max_completion_tokens: int = 8000, log=print) -> dict:
    """Runs the Wonder tool loop. Returns answer, registry, usage and trace."""
    registry = Registry()
    messages = list(messages)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    trace = []
    for round_number in range(max_rounds + 1):
        body = {"model": model, "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_completion_tokens, "messages": messages}
        if round_number < max_rounds:
            body["tools"] = TOOL_SPECS
        else:
            messages.append({"role": "developer",
                             "content": "Tool limit reached. Answer now with what you have."})
        data = _post_json(MUSE_URL, body, timeout=300,
                          headers={"Authorization": f"Bearer {api_key}"})
        for key in usage:
            usage[key] += (data.get("usage") or {}).get(key) or 0
        message = data["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return {"answer": message.get("content") or "", "registry": registry,
                    "usage": usage, "trace": trace}
        messages.append({"role": "assistant", "content": message.get("content"),
                         "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or "{}"
            log(f"tool {name} {arguments}")
            result = run_tool(name, arguments, registry)
            trace.append({"tool": name, "arguments": arguments, "chars": len(result)})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    raise RuntimeError("Tool loop ended without an answer.")
