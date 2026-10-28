"""AlphaGenome scoring pipeline for POST /score.

Wraps ``dna_client.score_variant`` over a 1 Mb window with the four recommended
scorers, runs ``tidy_scores`` locally, filters rows to the canonical gene, and
assembles the ``alpha_genome`` payload from
``workstreams/contracts/alpha_genome_schema.json``.

No literature handling lives here: the response carries no ``literature`` key.
"""

import math
import os

import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models.v1 import dna_output
from alphagenome.models.v1 import variant_scorers as variant_scorers_module
from alphagenome.models.v1.variant_scorers import tidy_scores

from app.schemas import ScoreRequest

SEQUENCE_LENGTH_1MB = 1048576
HALF_WINDOW = SEQUENCE_LENGTH_1MB // 2
DEFAULT_MODEL_VERSION = "alphagenome-v1"
API_KEY_ENV_VAR = "ALPHA_GENOME_API_KEY"

RNA_SEQ = "RNA_SEQ"
SPLICE_SITES = "SPLICE_SITES"
SPLICE_SITE_USAGE = "SPLICE_SITE_USAGE"
SPLICE_JUNCTIONS = "SPLICE_JUNCTIONS"
MERGED_SPLICING_SCORERS = (SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS)


class UpstreamUnavailableError(RuntimeError):
    """AlphaGenome API is unreachable, unconfigured, or returned an error."""


class RateLimitedError(RuntimeError):
    """AlphaGenome API rate limit exceeded."""


def build_variant(request: ScoreRequest) -> genome.Variant:
    return genome.Variant(
        chromosome=request.chromosome,
        position=request.position,
        reference_bases=request.reference_allele,
        alternate_bases=request.alternate_allele,
    )


def build_interval(chromosome: str, position: int) -> genome.Interval:
    """1 Mb window centered on the variant, shifted right if it starts < 0."""
    start = position - HALF_WINDOW
    end = position + HALF_WINDOW
    if start < 0:
        end -= start
        start = 0
    return genome.Interval(chromosome, start, end)


def build_scorers() -> list:
    return [
        variant_scorers_module.GeneMaskLFCScorer(
            requested_output=dna_output.OutputType.RNA_SEQ
        ),
        variant_scorers_module.GeneMaskSplicingScorer(
            requested_output=dna_output.OutputType.SPLICE_SITES, width=None
        ),
        variant_scorers_module.GeneMaskSplicingScorer(
            requested_output=dna_output.OutputType.SPLICE_SITE_USAGE, width=None
        ),
        variant_scorers_module.SpliceJunctionScorer(),
    ]


def create_client(api_key: str | None = None) -> dna_client.DnaClient:
    key = api_key or os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise UpstreamUnavailableError(f"{API_KEY_ENV_VAR} is not configured")
    return dna_client.create(key)


def score_with_client(
    request: ScoreRequest, client: dna_client.DnaClient
) -> dict:
    """Run the full pipeline with an injected client (no env access)."""
    variant = build_variant(request)
    interval = build_interval(request.chromosome, request.position)
    try:
        scores = client.score_variant(interval, variant, build_scorers())
    except (RateLimitedError, UpstreamUnavailableError):
        raise
    except Exception as exc:
        raise _map_client_error(exc) from exc

    tidy = tidy_scores(scores, match_gene_strand=True)
    filtered = _filter_to_canonical_gene(
        tidy, request.canonical_gene_id, request.canonical_gene
    )
    rows = [_row_to_dict(row) for _, row in filtered.iterrows()]
    return {
        "variant": request.variant,
        "hg38_variant": f"{request.chromosome}:{request.position}"
        f":{request.reference_allele}>{request.alternate_allele}",
        "scored_interval": f"{interval.chromosome}:{interval.start}-{interval.end}",
        "model_version": DEFAULT_MODEL_VERSION,
        "canonical_gene_id": request.canonical_gene_id,
        "canonical_gene": request.canonical_gene,
        "merged_splicing_score": _merged_splicing_score(filtered),
        "scores": rows,
    }


def _filter_to_canonical_gene(
    tidy: pd.DataFrame | None, gene_id: str | None, gene_name: str
) -> pd.DataFrame:
    if tidy is None or tidy.empty:
        return pd.DataFrame()
    matches_name = tidy["gene_name"] == gene_name
    if gene_id:
        gene_id = gene_id.split(".")[0]
        matches = (tidy["gene_id"] == gene_id) | matches_name
    else:
        matches = matches_name
    return tidy[matches].reset_index(drop=True)


def _merged_splicing_score(filtered: pd.DataFrame) -> float | None:
    """max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS) / 5."""
    if filtered.empty or "output_type" not in filtered.columns:
        return None
    maxima = (
        filtered.groupby("output_type")["raw_score"].max(numeric_only=True).to_dict()
    )
    if not any(key in maxima for key in MERGED_SPLICING_SCORERS):
        return None
    return float(
        maxima.get(SPLICE_SITES, 0.0)
        + maxima.get(SPLICE_SITE_USAGE, 0.0)
        + maxima.get(SPLICE_JUNCTIONS, 0.0) / 5.0
    )


def _row_to_dict(row: pd.Series) -> dict:
    def clean(value):
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str) and value in ("", "nan", "NaN", "None"):
            return None
        return value

    def number(value):
        value = clean(value)
        return None if value is None else float(value)

    def integer(value):
        value = clean(value)
        return None if value is None else int(value)

    raw = row.to_dict()
    return {
        "output_type": raw.get("output_type"),
        "variant_scorer": raw.get("variant_scorer"),
        "gene_id": clean(raw.get("gene_id")),
        "gene_name": clean(raw.get("gene_name")),
        "junction_start": integer(
            raw.get("junction_start", raw.get("junction_Start"))
        ),
        "junction_end": integer(raw.get("junction_end", raw.get("junction_End"))),
        "track_name": clean(raw.get("track_name")),
        "ontology_curie": clean(raw.get("ontology_curie")),
        "biosample_name": clean(raw.get("biosample_name")),
        "raw_score": number(raw.get("raw_score")),
        "quantile_score": number(raw.get("quantile_score")),
    }


def _map_client_error(exc: Exception) -> RuntimeError:
    try:
        import grpc

        if isinstance(exc, grpc.RpcError):
            if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                return RateLimitedError(str(exc))
    except ImportError:
        pass
    return UpstreamUnavailableError(str(exc))
