import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wstg-intelligence")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "wstg-v1")

if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing from .env")

# Require confirmation before deleting data.
confirmation = input(
    f"Type '{NAMESPACE}' to delete all vectors in this namespace: "
)

if confirmation != NAMESPACE:
    print("Cancelled. No deletion requested.")
    raise SystemExit(0)

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# Delete all vectors in this namespace only.
index.delete(
    delete_all=True,
    namespace=NAMESPACE,
)

print(f"Deletion requested for namespace: {NAMESPACE}")
print(f"Index preserved: {INDEX_NAME}")
print("Check Pinecone statistics again after deletion completes.")