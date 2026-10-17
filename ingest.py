import os
import json

from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document

from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

load_dotenv()

wstg_path = os.getenv(
    "WSTG_REPO_PATH",
    "./wstg/document/4-Web_Application_Security_Testing"
)

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
    "wstg-v1",
)

# Load the embedding model
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device" : "cpu"},
    encode_kwargs={"normalize_embeddings" : True},
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
BATCH_SIZE = 32




def load_files() -> list[Document]:
    """Load all Markdown files, including nested directories."""
    docs_dir = Path(wstg_path)

    docs = []

    for file_path in docs_dir.rglob("*.md"):

        content = file_path.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source" : file_path.relative_to(docs_dir).as_posix(),
                    "filename" : file_path.name
                }
            )
        )
    
    print(f"Loaded {len(docs)} Markdown files")
    return docs



# This fn is for tokenizer to measure the length of a sections
def token_length(text: str) -> int:
    return len(
        tokenizer.backend_tokenizer.encode(
            text,
            add_special_tokens=False
        ).ids
    )



def split_documents(documents: list[Document]) -> list[Document]:
    """Split Markdown by headings, then by token count."""


    # Configuring MarkdownHeaderTextSplitter, when to split.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False
    )


    # Then split oversized sections using the model tokenizer.
    # creates a splitter that uses your supplied Hugging Face tokenizer to count tokens.
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_length,
        separators=["\n\n", "\n", ".", " ", ""],
    )


    chunks = []

    for document in documents:

        # splits the .md file by headers
        sections = header_splitter.split_text(
            document.page_content
        )

        if len(sections) >= 2:
            first = sections[0]
            second = sections[1]

            # Merge first two documents becoz the the very first chunk contains only id and title so we merge with preceeding section.
            
            is_title_id_block = (
                token_length(first.page_content) < 50
                and first.metadata.get("h1")
                and not any(
                    first.metadata.get(key)
                    for key in ("h2", "h3", "h4")
                )
                and first.metadata.get("h1")
                == second.metadata.get("h1")
                and second.metadata.get("h2")
            )

            if is_title_id_block:
                merged_section = Document(
                    page_content=(
                        first.page_content
                        +"\n\n"
                        + second.page_content
                    ),
                    metadata={
                        **first.metadata,
                        **second.metadata,
                    },
                )
                sections = [merged_section, *sections[2:]]

        for section in sections:

            # perserves the orginal file's metadata.
            section.metadata.update(document.metadata)

            # Builds a readable heading path, which tells exactly where a chunk belongs in WSTG Markdown document.
            # Example : Testing Directory Traversal File Include > How to Test > Black-Box Testing > Input Vectors Enumeration
            heading_path = " > ".join(
                str(section.metadata[key])
                for key in ("h1", "h2", "h3", "h4")
                if section.metadata.get(key)
            )

            if heading_path:
                section.metadata["heading_path"] = heading_path

        section_chunks = token_splitter.split_documents(sections)
        chunks.extend(section_chunks)

    return chunks



def index_chunks(chunks: list[Document], embedding):

    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("PINCONE_API_KEY is missing from .env")
    
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embedding,
        namespace=NAMESPACE,
    )

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        batch = chunks[start:start+BATCH_SIZE]

        vector_store.add_documents(
            documents=batch,
        )
        print(f"Uploaded {min(start + len(batch), total)}/{total}")
        


def save_chunks(chunks, output_path="data/wstg_chunks.jsonl"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {output}")


documents = load_files()
chunks = split_documents(documents)
save_chunks(chunks)
index_chunks(chunks, embedding)

print("Indexing complete.")
print(f"Index: {INDEX_NAME}")
print(f"Namespace: {NAMESPACE}")



