"""Static checks for colabs/cancer_variant_cookbook.ipynb.

No network and no model calls. The live check is a run in Colab.
"""

import ast
import importlib.metadata
import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

COLABS_DIR = Path(__file__).resolve().parents[1] / "colabs"
NOTEBOOK_PATH = COLABS_DIR / "cancer_variant_cookbook.ipynb"
FORM_FIELDS = (
    "variant_chromosome",
    "variant_position",
    "variant_reference_bases",
    "variant_alternate_bases",
    "gene_symbol",
    "ontology_terms",
    "sidecar_scores",
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_cookbook", COLABS_DIR / "build_cookbook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_is_valid(notebook):
    nbformat.validate(notebook)


def test_notebook_matches_builder():
    # The .ipynb is generated; a hand edit makes this fail.
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == _load_builder().render()


def test_code_cells_parse(code_cells):
    assert len(code_cells) == 7
    for number, source in enumerate(code_cells, start=1):
        python = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("!")
        )
        try:
            ast.parse(python)
        except SyntaxError as exc:
            pytest.fail(f"cell {number} does not parse: {exc}")


def test_pip_pin_matches_sidecar_env(code_cells):
    installed = importlib.metadata.version("alphagenome")
    assert re.search(rf"pip install .*alphagenome=={re.escape(installed)}\b", code_cells[0])


def test_cell_2_has_api_key_field(code_cells):
    cell_2 = code_cells[1]
    assert re.search(r'^API_KEY = "".*# @param', cell_2, re.MULTILINE)
    assert "colab_utils.get_api_key()" in cell_2


def test_cell_4_has_form_fields(code_cells):
    cell_4 = code_cells[3]
    for name in FORM_FIELDS:
        assert re.search(rf"^{name} = .*# @param", cell_4, re.MULTILINE), name


def test_notebook_has_no_outputs(code_cells, notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
