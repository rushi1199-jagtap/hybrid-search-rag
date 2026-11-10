
from retrieval.hybrid_search import hybrid_search
from generation.context_builder import build_context
from generation.gemini_client import generate_answer

from generation.verification.claim_extractor import extract_claims
from generation.verification.citation_mapper import map_claim_citations
from generation.verification.verification_engine import verify_claims
from generation.verification.final_answer import build_final_answer


def run_rag_pipeline(
    query,
    top_k=5,
    semantic_top_k=10,
    bm25_top_k=10,
    use_reranking=True
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    retrieval_results = hybrid_search(
        query=query,
        top_k=top_k,
        semantic_top_k=semantic_top_k,
        bm25_top_k=bm25_top_k,
        use_reranking=use_reranking
    )

    context = build_context(retrieval_results)

    generated_response = generate_answer(
        query=query,
        context=context
    )

    claims = extract_claims(
        generated_response["answer"]
    )

    mapped_claims = map_claim_citations(
        claims=claims,
        citations=generated_response["citations"],
        retrieved_chunks=retrieval_results
    )

    verification_results = verify_claims(
        mapped_claims
    )

    final_response = build_final_answer(
        generated_response=generated_response,
        verification_results=verification_results
    )

    return {
        "query": query,
        **final_response,
        "retrieved_chunks": retrieval_results
    }
