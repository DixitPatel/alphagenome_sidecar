"""Pure tests for the facts functions in notebook cell 8 (Explain with Muse).

The functions are loaded from the builder source with ast, so the test checks
the code that ships in the notebook. Inputs are real AlphaGenome TrackData and
JunctionData objects built from small arrays. No mocks, no network.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alphagenome.data import genome, junction_data, track_data

BUILDER = Path(__file__).resolve().parents[1] / "colabs" / "build_cookbook.py"


@pytest.fixture(scope="module")
def cell8():
    spec = importlib.util.spec_from_file_location("build_cookbook", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    tree = ast.parse(builder.CELL_8)
    keep = {"track_changes", "junction_changes", "exon_context"}
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep],
        type_ignores=[],
    )
    namespace = {"np": np, "TOP_TRACKS": 3}
    exec(compile(module, "cell_8", "exec"), namespace)
    return namespace


def _tracks(values, names, start0=1000):
    values = np.asarray(values, dtype=np.float32)
    return track_data.TrackData(
        values=values,
        metadata=pd.DataFrame({"name": names, "strand": ["-"] * len(names)}),
        resolution=1,
        interval=genome.Interval("chr22", start0, start0 + values.shape[0]),
    )


def test_track_changes_position_offset_and_window(cell8):
    ref = np.zeros((20, 2))
    alt = np.zeros((20, 2))
    alt[12, 0] = -0.5   # inside the window: 1012 is 2 bp right of the variant
    alt[19, 1] = 0.9    # outside a 3 bp window
    alt[9, 1] = 0.2
    rows = cell8["track_changes"](
        _tracks(ref, ["a", "b"]), _tracks(alt, ["a", "b"]), center0=1010, half_window=3
    )
    assert rows[0] == {
        "track": "a", "strand": "-", "position": 1013, "offset_from_variant": 2,
        "ref": 0.0, "alt": -0.5, "delta": -0.5,
    }
    assert rows[1]["track"] == "b" and rows[1]["position"] == 1010
    assert rows[1]["offset_from_variant"] == -1  # 0.9 at 1019 is outside +/-3


def test_track_changes_empty_tracks(cell8):
    empty = _tracks(np.zeros((5, 0)), [])
    assert cell8["track_changes"](empty, empty, 1002, 10) == []


def test_junction_changes_missing_in_alt(cell8):
    meta = pd.DataFrame({"name": ["polyA", "total"], "strand": ["-", "-"]})
    j1 = genome.Junction("chr22", 1000, 1050, strand="-")
    j2 = genome.Junction("chr22", 5000, 5100, strand="-")
    ref = junction_data.JunctionData(
        junctions=np.array([j1, j2]), values=np.array([[0.3, 0.25], [0.5, 0.5]]), metadata=meta
    )
    alt = junction_data.JunctionData(
        junctions=np.array([j2]), values=np.array([[0.5, 0.5]]), metadata=meta
    )
    rows = cell8["junction_changes"](ref, alt, center0=1049, half_window=100)
    assert rows == [{
        "junction": "1001-1050", "strand": "-", "track": "polyA",
        "ref": 0.3, "alt": 0.0, "delta": -0.3, "in_ref": True, "in_alt": False,
    }]


def test_exon_context_edge_distance(cell8):
    gtf = pd.DataFrame({
        "gene_name": ["CHEK2", "CHEK2", "OTHER"],
        "Feature": ["exon", "exon", "exon"],
        "Start": [100, 300, 0],
        "End": [200, 400, 1000],
        "transcript_id": ["t1", "t1", "t9"],
    })
    inside = cell8["exon_context"](gtf, "CHEK2", 199)
    assert inside == {"variant_in_exon": True, "nearest_exon_edge": 200,
                      "distance_bp": 0, "n_transcripts_checked": 1}
    intron = cell8["exon_context"](gtf, "CHEK2", 250)
    assert intron["variant_in_exon"] is False
    # 0-based exon start 300 is 1-based 301; 50 bp from the 0-based variant 250
    assert intron["nearest_exon_edge"] == 301 and intron["distance_bp"] == 50
    assert "note" in cell8["exon_context"](gtf, "KRAS", 10)
