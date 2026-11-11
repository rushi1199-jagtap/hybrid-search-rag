
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RAGRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question asked by the user."
    )

    top_k: int = Field(default=5, ge=1, le=20)
    semantic_top_k: int = Field(default=10, ge=1, le=50)
    bm25_top_k: int = Field(default=10, ge=1, le=50)
    use_reranking: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty.")

        return value


class CitationSchema(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    source: Optional[str] = None
    page_number: Optional[int] = None


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    document_id: str
    chunk_text: str

    score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None

    source: Optional[str] = None
    file_type: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None


class SemanticVerificationSchema(BaseModel):
    claim_id: Optional[str] = None
    claim_text: str
    verdict: str
    reason: str
    verified_evidence_ids: list[str]


class MappedEvidenceSchema(BaseModel):
    chunk_id: str
    document_id: str
    chunk_text: str

    source: Optional[str] = None
    page_number: Optional[int] = None


class MappedCitationSchema(BaseModel):
    evidence_id: str
    citation: CitationSchema
    evidence: MappedEvidenceSchema


class ClaimVerificationSchema(BaseModel):
    claim_id: Optional[str] = None
    claim_text: str

    status: str
    citation_valid: bool

    mapped_citations: list[MappedCitationSchema] = Field(
        default_factory=list
    )

    invalid_evidence_ids: list[str] = Field(
        default_factory=list
    )

    semantic_verification: Optional[SemanticVerificationSchema] = None
    overall_status: str


class VerificationSummarySchema(BaseModel):
    overall_status: str
    total_claims: int
    status_counts: dict[str, int]
    claim_results: list[ClaimVerificationSchema]


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationSchema]
    insufficient_evidence: bool

    verification: VerificationSummarySchema

    retrieved_chunks: list[RetrievedChunkSchema]
