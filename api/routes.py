
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import RAGRequest, RAGResponse
from generation.answer_pipeline import run_rag_pipeline


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest):

    try:
        result = run_rag_pipeline(
            query=request.query,
            top_k=request.top_k,
            semantic_top_k=request.semantic_top_k,
            bm25_top_k=request.bm25_top_k,
            use_reranking=request.use_reranking
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception("RAG pipeline execution failed.")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question."
        ) from error
