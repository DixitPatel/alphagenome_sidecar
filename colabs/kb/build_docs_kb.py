"""Builds colabs/kb/alphagenome_docs.json, the static AlphaGenome docs KB.

The muse_tools.alphagenome_docs tool searches this file. Each record is one
docs section (prose, lists and tables; code and cell outputs are dropped).

Run from the repo root:  python colabs/kb/build_docs_kb.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

BASE_URL = "https://www.alphagenomedocs.com/"
PAGES = (
    "variant_scoring",
    "faqs",
    "exploring_model_metadata",
    "visualization_library_basics",
    "colabs/v1/splicing_variant_scoring",
    "colabs/v1/deriving_PSI_values",
    "colabs/v1/tissue_ontology_mapping",
    "colabs/v1/example_analysis_workflow",
)
KB_PATH = Path(__file__).with_name("alphagenome_docs.json")
MAX_SECTION_CHARS = 4000
DROP_SELECTORS = (
    "div.highlight", "pre", "div.cell_input", "div.cell_output",
    "div.nbinput", "div.nboutput", "a.headerlink", "script", "style", "img",
)


def fetch(page: str) -> str:
    request = urllib.request.Request(
        BASE_URL + page + ".html", headers={"User-Agent": "alphagenome-sidecar-kb"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def clean_text(node) -> str:
    text = node.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def page_sections(page: str, html: str) -> list[dict]:
    """One record per <section> with a heading. Nested sections are separate."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("article") or soup.select_one('div[role="main"]') or soup
    for selector in DROP_SELECTORS:
        for node in main.select(selector):
            node.decompose()
    slug = page.split("/")[-1]
    records = []
    for section in main.find_all("section"):
        heading = section.find(re.compile(r"^h[1-4]$"))
        if heading is None or not section.get("id"):
            continue
        parts = []
        for child in section.children:
            if getattr(child, "name", None) in (None, "section"):
                continue
            if child is heading:
                continue
            text = clean_text(child)
            if text:
                parts.append(text)
        text = " ".join(parts)[:MAX_SECTION_CHARS]
        if len(text) < 40:
            continue
        records.append({
            "id": f"AGDocs:{slug}#{section['id']}",
            "title": clean_text(heading),
            "page": slug,
            "url": f"{BASE_URL}{page}.html#{section['id']}",
            "text": text,
        })
    return records


def build() -> list[dict]:
    records = []
    for page in PAGES:
        records.extend(page_sections(page, fetch(page)))
    return records


if __name__ == "__main__":
    kb = build()
    KB_PATH.write_text(json.dumps(kb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(kb)} sections to {KB_PATH}")
