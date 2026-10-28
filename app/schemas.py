"""Request/response shapes for POST /score.

Mirrors workstreams/contracts/sidecar_openapi.yaml. There is no literature
surface: no ``include_literature`` request field and no ``literature`` response
key. Unknown extra fields are ignored so older payloads still validate.
"""

from pydantic import BaseModel, Field

CHROMOSOME_PATTERN = r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$"
ALLELE_PATTERN = r"^[ACGT]+$"


class ScoreRequest(BaseModel):
    variant: str = Field(description="Original Genome Nexus variant ID, echoed in the response")
    chromosome: str = Field(pattern=CHROMOSOME_PATTERN, examples=["chr7"])
    position: int = Field(ge=1, description="1-based GRCh38 position", examples=[140753336])
    reference_allele: str = Field(pattern=ALLELE_PATTERN, examples=["A"])
    alternate_allele: str = Field(pattern=ALLELE_PATTERN, examples=["T"])
    canonical_gene_id: str | None = Field(default=None, examples=["ENSG00000157764"])
    canonical_gene: str = Field(examples=["BRAF"])
    rsid: str | None = None
    hgvsc: str | None = None
    hgvsp: str | None = None
