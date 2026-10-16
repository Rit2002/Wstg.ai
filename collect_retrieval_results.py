"""
PART A: Collect and save retrieval rankings for the WSTG evaluation questions.

This script calls retrieval systems. It does not calculate evaluation metrics.

Example:
    python collect_retrieval_results.py \
      --dataset wstg_information_gathering_eval.json \
      --output-dir retrieval_experiments

It writes one ranked-results JSON file per method plus a manifest.
It checkpoints after each question, so completed queries are not lost if a run
is interrupted. Use --resume to continue a compatible previous run.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hybrid_retriever as hr

METHODS = (
    "bm25",
    "pinecone_vector",
    "hybrid_rrf",
    "hybrid_rrf_cohere",
)


def sha256_file(path: Path) -> str:
    """
    Generate a unique fingerprint of the dataset's contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        # Reads the file in blocks of 1 MB.
        # Continues until the file returns an empty byte string, b"".
        # Feeds each block into the hash calculation.
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Convert a LangChain Document object into a dictionary that can be saved in JSON.
def serialize_document(doc) -> dict[str, Any]:
    metadata = doc.metadata or {}
    source = metadata.get("source") or metadata.get("file_path") or ""
    return {
        "source": str(source),
        "heading": str(metadata.get("heading_path", "")),
        "content": doc.page_content,
        # Retain metadata useful for later debugging; source is also top-level
        # for compatibility with the evaluation script.
        "metadata": {
            str(key): value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
            for key, value in metadata.items()
        },
    }


# Purpose: Save data to a JSON file more safely.
def write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Collect ranked retrieval outputs.")
    parser.add_argument("--dataset", required=True, help="Evaluation dataset JSON path.")
    parser.add_argument("--output-dir", default="retrieval_experiments", help="Output directory.")
    parser.add_argument("--k", type=int, default=None, help="Override K; defaults to dataset k (5).")
    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--vector-k", type=int, default=25)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume a compatible partial run.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    k = args.k if args.k is not None else int(dataset.get("k", 5))
    if min(k, args.bm25_k, args.vector_k, args.candidate_k) < 1:
        raise ValueError("All k values must be positive integers.")

    settings = {
        "k": k,
        "bm25_k": args.bm25_k,
        "vector_k": args.vector_k,
        "candidate_k": args.candidate_k,
        "rrf_k": hr.RRF_K,
        "embedding_model": hr.EMBEDDING_MODEL,
        "pinecone_index": hr.INDEX_NAME,
        "pinecone_namespace": hr.NAMESPACE,
        "cohere_rerank_model": os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
        "relevance_unit": dataset.get("relevance_unit", "source"),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_name": dataset.get("dataset_name"),
        "query_count": len(dataset.get("queries", [])),
        "settings": settings,
        "methods": list(METHODS),
        "note": (
            "Saved results contain top-k chunks. Current metrics use source-file "
            "relevance labels and deduplicate repeated source filenames."
        ),
    }

    output_paths = {
        method: output_dir / f"{method}_retrieval.json"
        for method in METHODS
    }
    outputs: dict[str, dict] = {}
    completed_ids: set[str] = set()

    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("--resume requested but run_manifest.json does not exist.")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("dataset_sha256") != manifest["dataset_sha256"] or old_manifest.get("settings") != settings:
            raise ValueError("Cannot resume: dataset or retrieval settings differ from the previous run.")
        for method, path in output_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Cannot resume: missing {path}")
            outputs[method] = json.loads(path.read_text(encoding="utf-8"))
        id_sets = [
            {str(item["query_id"]) for item in outputs[m].get("results", [])}
            for m in METHODS
        ]
        if any(ids != id_sets[0] for ids in id_sets[1:]):
            raise ValueError("Cannot resume: method files contain different completed query IDs.")
        completed_ids = id_sets[0]
        manifest = old_manifest
    else:
        outputs = {
            method: {
                "method": method,
                "run_metadata": manifest,
                "results": [],
            }
            for method in METHODS
        }
        write_json(manifest_path, manifest)
        for path, payload in zip(output_paths.values(), outputs.values()):
            write_json(path, payload)

    queries = dataset.get("queries", [])
    for position, item in enumerate(queries, start=1):
        query_id = str(item["id"])
        if query_id in completed_ids:
            print(f"[{position}/{len(queries)}] {query_id}: already collected, skipping.")
            continue

        question = item["question"]
        print(f"[{position}/{len(queries)}] {query_id}: {question}")
        stage_output = hr.retrieve_stages(
            query=question,
            k=k,
            bm25_k=args.bm25_k,
            vector_k=args.vector_k,
            candidate_k=args.candidate_k,
        )

        cohere_status = stage_output["cohere_status"]
        print(f"  Cohere status: {cohere_status.get('status')} ({cohere_status.get('reason', '')})")

        for method in METHODS:
            outputs[method]["results"].append({
                "query_id": query_id,
                "question": question,
                "results": [
                    serialize_document(doc)
                    for doc in stage_output[method]
                ],
                **({"reranker_status": cohere_status} if method == "hybrid_rrf_cohere" else {}),
            })

        # Checkpoint every query so a long Cohere run can be resumed.
        for method, path in output_paths.items():
            write_json(path, outputs[method])
        completed_ids.add(query_id)

    # Update manifest with completion information and write it to every file.
    manifest["completed_query_count"] = len(completed_ids)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    for method, path in output_paths.items():
        outputs[method]["run_metadata"] = manifest
        write_json(path, outputs[method])

    print("\nRetrieval collection finished.")
    print(f"Completed queries: {len(completed_ids)}/{len(queries)}")
    print(f"Output directory: {output_dir.resolve()}")
    for method, path in output_paths.items():
        print(f"  {method}: {path.resolve()}")
    print(f"  manifest: {manifest_path.resolve()}")
    failed = [
        row for row in outputs["hybrid_rrf_cohere"]["results"]
        if row.get("reranker_status", {}).get("status") == "fallback"
    ]
    if failed:
        print(
            f"\nWARNING: Cohere fell back to RRF order for {len(failed)} query(s). "
            "Those rows are not successful Cohere reranking results. Fix the API "
            "problem and recollect before making a reranker-quality claim."
        )


if __name__ == "__main__":
    main()
