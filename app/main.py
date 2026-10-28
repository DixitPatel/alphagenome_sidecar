from fastapi import Depends, FastAPI, HTTPException

from app import scoring
from app.schemas import ScoreRequest

app = FastAPI(
    title="AlphaGenome Scoring Sidecar",
    description="Scores GRCh38 variants via AlphaGenome and returns gene-filtered summaries for Genome Nexus.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "alphagenome-sidecar"}


def get_client() -> scoring.dna_client.DnaClient:
    return scoring.create_client()


@app.post("/score")
def score_variant(
    request: ScoreRequest,
    client: scoring.dna_client.DnaClient = Depends(get_client),
) -> dict:
    try:
        return scoring.score_with_client(request, client)
    except scoring.RateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except scoring.UpstreamUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
