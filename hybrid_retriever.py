"""
Total: 88 + 25 questions
OWASP WSTG retrieval pipeline.

Provides:
- retrieve_stages(): returns ranked chunks from BM25, Pinecone, RRF, and Cohere.
- hybrid_retrieve(): backward-compatible production entry point returning final chunks.

Evaluation is source-file-level because the current evaluation dataset labels relevant_sources.
The result files still retain the top-k chunks and their metadata.
"""


import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils import document_key
from reranker import cohere_rerank_with_status


load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-en-v1.5"
)

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "wstg-intelligence"
)

NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "wstg-v1"
)

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding,
    namespace=NAMESPACE,
)



CORPUS_PATH = Path("data/wstg_chunks.jsonl")

RRF_K = 60





# BM25: keyword retrieval
def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 keyword retrieval."""
    
    # re is Python's regular expression module.
    # findall() finds all matches in the text. 
    # r	 :  Creates a raw string, which is useful for writing regex patterns.
    # \b :	Word boundary, such as the beginning or end of a word.
    # \w+:	One or more word characters, usually letters, digits, or underscores.
    # \b :	Ensures the match ends at a word boundary.
    return re.findall(r"\b\w+\b", text.lower())




def load_corpus() -> list[Document]:
    """Load previously saved WSTG chunks from JSONL."""
    if not CORPUS_PATH.exists():
         raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH.resolve()}\n"
            "Run your ingestion script to create the saved chunk corpus."
        )
    
    documents = []

    # "r" means read mode.
    with CORPUS_PATH.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            # Skip the empty lines
            if not line.strip():
                continue

            try:
                # Convert the JSON text into a Python dictionary
                record = json.loads(line)
                documents.append(
                    Document(
                        page_content=record["page_content"],
                        metadata=record.get("metadata", {}),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                   f"Invalid corpus record on line {line_number}"
                ) from exc 
            
    if not documents:
        raise ValueError(f"No chunks found in {CORPUS_PATH.resolve()}")
    
    return documents
    

documents = load_corpus()


# BM25Okapi : It builds the data structures needed to rank documents according to their relevance to a search query.
# It considers factors such as:
# - Term frequency (TF): How often a query term appears in a document.
# - Inverse document frequency (IDF): How informative a term is across the document collection.
# - Document length: Adjusts for differences in chunk length.
bm25 = BM25Okapi(
    [tokenize(doc.page_content) for doc in documents]
)








def is_reference_section(doc: Document) -> bool:
    """Return True if any segment of the heading path is 'References'."""
    heading_path = str(doc.metadata.get("heading_path", ""))
    return any(
        part.strip().lower() == "references"
        for part in heading_path.split(">")
    )





def remove_reference_sections(ranking: list[Document]) -> list[Document]:
    return [doc for doc in ranking if not is_reference_section(doc)]






def reciprocal_rank_fusion(
        rankings: list[list[Document]],
        k: int = RRF_K
) -> list[Document]:
    """Combine ranked lists using Reciprocal Rank Fusion."""
    
    # fused_scores stores the final RRF score for each unique document.
    fused_scores: dict[str, float] = {}
    # documents_by_key stores the actual Document object associated with each key.
    documents_by_key: dict[str, Document] = {}

    for ranking in rankings:

        # Creates an empty set to track documents already encountered in the current ranking.
        seen_in_ranking: set[str] = set()

        for rank, doc in enumerate(ranking, start=1):
            key = document_key(doc)

            if key in seen_in_ranking:
                continue

            seen_in_ranking.add(key)
            fused_scores[key] = (
                fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            )
            documents_by_key[key] = doc

    # Sorts the dictionary by value (key=fused_scores.get)
    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )


    return [documents_by_key[key] for key in ranked_keys]




# Hybrid retrieval: BM25 + dense + RRF
def retrieve_stages(
        query: str,
        k: int = 5,
        bm25_k: int = 30,
        vector_k: int = 25,
        candidate_k: int = 20,
) -> list[Document]:
    """
    Return top-k chunks for each ablation stage.

    BM25 and Pinecone candidate rankings are fused before reference sections are
    removed, matching the original hybrid_retrieve() behavior. Standalone
    baselines are filtered for reference sections before their top-k outputs.
    Cohere receives the top candidate_k RRF results and returns at most k.

    'cohere_status' is metadata for the experiment runner; it is not a ranking.
    """
     
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    if k < 1 or candidate_k < 1:
        raise ValueError("k and candidate_k must be positive integers.")

    # Stage 1: BM25 keyword retrieval.
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # range(len(bm25_scores)) : len(bm25_scores) returns 4 & range(4) generates list of indices [0, 1, 2, 3].
    # Sorts the generated list, uses bm25_scores as criteria
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key = lambda index: bm25_scores[index],
        reverse = True
    )[:bm25_k]

    bm25_candidates = [
        documents[index]
        for index in bm25_indices
        if bm25_scores[index] > 0
    ]



    # Stage 2. Dense semantic retrieval from existing Pinecone vectors
    vector_candidates = vector_store.similarity_search(
        query=query,
        k=vector_k,
    )

    # Stage 3. Fuse the rankings
    fused_candidates = reciprocal_rank_fusion(
        [bm25_candidates, vector_candidates]
    )
    # Exclude reference section
    fused_candidates = remove_reference_sections(
        fused_candidates
    )

   

    # Apply the same reference-section exclusion to the individual baselines.
    bm25_output = remove_reference_sections(bm25_candidates)[:k]
    vector_output = remove_reference_sections(vector_candidates)[:k]
    rrf_output = fused_candidates[:k]

    

    # Stage 4: Cohere reranking over the top candidate_k fused candidates.
    rerank_candidates = fused_candidates[:candidate_k]
    if not rerank_candidates:
        reranked_output = []
        cohere_status = {
            "status": "skipped",
            "reason": "no_candidates",
            "model": os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
        }
    elif len(rerank_candidates) <= k:
        # Preserve the production behavior: no API call when reranking cannot
        # change the returned top-k list.
        reranked_output = rerank_candidates[:k]
        cohere_status = {
            "status": "skipped",
            "reason": "candidate_count_not_greater_than_k",
            "model": os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
        }
    else:
        reranked_output, cohere_status = cohere_rerank_with_status(
            query=query,
            candidates=rerank_candidates,
            top_n=k,
        )

    return {
        "bm25": bm25_output,
        "pinecone_vector": vector_output,
        "hybrid_rrf": rrf_output,
        "hybrid_rrf_cohere": reranked_output[:k],
        "cohere_status": cohere_status,
    }

    


# Returns the final result are ranking by cohere
def hybrid_retrieve(
    query: str,
    k: int = 5,
    bm25_k: int = 30,
    vector_k: int = 25,
    candidate_k: int = 20,
) -> list[Document]:
    """
    Backward-compatible production function: return final hybrid + Cohere chunks.
    Existing callers can continue using hybrid_retrieve(query, k=5).
    """
    return retrieve_stages(
        query=query,
        k=k,
        bm25_k=bm25_k,
        vector_k=vector_k,
        candidate_k=candidate_k,
    )["hybrid_rrf_cohere"]

