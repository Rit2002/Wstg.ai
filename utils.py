from langchain_core.documents import Document

# Reciprocal Rank Fusion

# it helps identify the same chunk when it appears in both BM25 and Pinecone search results.
def document_key(doc: Document) -> tuple[str, str, str]:
    """Create a stable key for matching duplicate chunks."""
    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("heading_path", ""),
        doc.page_content,
    )
