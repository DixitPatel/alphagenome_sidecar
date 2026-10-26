from fastapi import FastAPI

app = FastAPI(
    title="AlphaGenome Scoring Sidecar",
    description="Scores GRCh38 variants via AlphaGenome and returns gene-filtered summaries for Genome Nexus.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "alphagenome-sidecar"}
