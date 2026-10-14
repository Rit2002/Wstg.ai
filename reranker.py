import time 
import cohere
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from utils import document_key

load_dotenv()



COHERE_API_KEY=os.getenv("COHERE_API_KEY")
COHERE_RERANK_MODEL=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")


co = cohere.ClientV2(api_key=COHERE_API_KEY)

MIN_CALL_INTERVAL = 6.5 # SECONDS;
_last_call_time = 0.0
_rerank_cache: dict = {}


def cohere_rerank(query: str, candidates: list[Document], top_n: int) -> list[Document]:
    """Rerank with Cohere. One API call per query, with cache, throttle, retry and fallback."""
    
    global _last_call_time

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

    for i, (doc, text) in enumerate(zip(candidates, texts), start=1):
        if (
            "Insecure Direct Object References"
            in doc.metadata.get("heading_path", "")
            or "Test Defenses Against Application Misuse"
            in doc.metadata.get("heading_path", "")
        ):
            print(f"\n--- Reranker input for candidate {i} ---")
            print(text[:1200])

    for attempt in range(3):
        # Throttle so we never exceed the per-minute limit
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call_time)

        if wait > 0:
            time.sleep(wait)

        try:
            _last_call_time = time.monotonic()
            response = co.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=texts,
                top_n=len(candidates)
            )

            ranked_results = []

            for result in response.results:
                doc = candidates[result.index]
                ranked_results.append((doc, result.relevance_score))

                print(
                    f"Cohere rank: {len(ranked_results)} | "
                    f"Score: {result.relevance_score:.4f} | "
                    f"Source: {doc.metadata.get('source')} | "
                    f"Heading: {doc.metadata.get('heading_path')}"
                )

            # Return only the requested number of final documents.
            ranked = [
                doc for doc, score in ranked_results[:top_n]
            ]

            _rerank_cache[cache_key] = ranked
            return ranked
        
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 and attempt < 2:
                time.sleep(2 ** (attempt + 1))   # 2s, then 4s
                continue
            print(f"Cohere rerank failed ({status or exc}); using RRF order instead.")
            break

    # Fallback: keep the fused (BM25 + vector) order
    return candidates[:top_n]


