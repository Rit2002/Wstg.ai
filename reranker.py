import time 
import cohere
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from utils import document_key

load_dotenv()



COHERE_API_KEY=os.getenv("COHERE_API_KEY")
COHERE_RERANK_MODEL=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")


MIN_CALL_INTERVAL = 6.5 # SECONDS;
_last_call_time = 0.0
_rerank_cache: dict = {}


_co = None


def _get_client():
    global _co
    if _co is None:
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY is not set in the environment.")
        _co = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _co




def cohere_rerank_with_status(
    query: str, 
    candidates: list[Document], 
    top_n: int
) -> tuple[list[Document], dict[str, Any]]:
    """
    Rerank with Cohere. One API call per query, with cache, throttle, retry and fallback.
    """
    
    global _last_call_time

    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if not candidates:
        return [], {"status": "skipped", "reason": "no_candidates", "model": COHERE_RERANK_MODEL}

    cache_key = (query, tuple(document_key(d) for d in candidates), top_n)
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]
    
    texts = [
        (
            f"Section: {doc.metadata.get('heading_path', '')}\n"
            f"Source: {doc.metadata.get('source', '')}\n"
            f"Content:\n{doc.page_content}"
        )
        for doc in candidates
    ]

    
    last_error = None
    for attempt in range(3):
        # Throttle so we never exceed the per-minute limit
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call_time)

        if wait > 0:
            time.sleep(wait)

        try:
            _last_call_time = time.monotonic()
            response = _get_client().rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=texts,
                top_n=len(candidates)
            )

            ranked_results = []


            # Return only the requested number of final documents.
            ranked_docs = [
                candidates[result.index]
                for result in response.results
            ][:top_n]

            status = {
                "status": "success",
                "model": COHERE_RERANK_MODEL,
                "candidate_count": len(candidates),
            }

            _rerank_cache[cache_key] = (ranked_docs, status)

            # For Debugging : Starts
            for rank, result in enumerate(response.results[:top_n], start=1):
                doc = candidates[result.index]
                print(
                    f"Cohere rank: {rank} | Score: {result.relevance_score:.4f} | "
                    f"Source: {doc.metadata.get('source')} | "
                    f"Heading: {doc.metadata.get('heading_path')}"
                )
            # For Debugging : Ends

            return ranked_docs, status
        
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 and attempt < 2:
                time.sleep(2 ** (attempt + 1))   # 2s, then 4s
                continue
            print(f"Cohere rerank failed ({status or exc}); using RRF order instead.")
            break


    status = {
        "status": "fallback",
        "model": COHERE_RERANK_MODEL,
        "candidate_count": len(candidates),
        "error": f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown error",
    }

    # Fallback: keep the fused (BM25 + vector) order
    return candidates[:top_n], status





def cohere_rerank(query: str, candidates: list[Document], top_n: int) -> list[Document]:
    """Backward-compatible wrapper returning only ranked documents."""
    ranked, _status = cohere_rerank_with_status(query, candidates, top_n)
    return ranked