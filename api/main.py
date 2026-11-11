
from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="Hybrid Search RAG API",
    description="A RAG system with hybrid retrieval, reranking, and claim-level citation verification.",
    version="1.0.0"
)


app.include_router(
    router,
    prefix="/api/v1",
    tags=["RAG"]
)


@app.get("/")
def root():
    return {
        "message": "Hybrid Search RAG API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
