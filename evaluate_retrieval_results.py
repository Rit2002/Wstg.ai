"""
PART B: Evaluate saved retrieval rankings.

This script performs no retrieval calls. It only reads the dataset and saved
rankings, calculates source-level Precision@K, Recall@K, MRR@K, MAP@K, NDCG@K,
and writes one overall comparison CSV plus detailed JSON reports.

Example:
    python evaluate_retrieval_results.py \
      --dataset wstg_information_gathering_eval.json \
      --retrieval-dir retrieval_experiments

The evaluation set's relevant_sources labels define relevance. Repeated chunks
from the same source file count once. Results are the top-k chunks returned by
each method, evaluated at source-file level.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

METHOD_FILES = {
    "BM25": "bm25_retrieval.json",
    "Pinecone vector": "pinecone_vector_retrieval.json",
    "BM25 + vector + RRF": "hybrid_rrf_retrieval.json",
    "BM25 + vector + RRF + Cohere": "hybrid_rrf_cohere_retrieval.json",
}
METRICS = (
    "normalized_precision_at_k",
    "recall_at_k",
    "mrr_at_k",
    "map_at_k",
    "ndcg_at_k",
)


def normalize_source(source: Any) -> str:
    if source is None:
        return ""
    return str(source).replace("\\", "/").rstrip("/").split("/")[-1].strip()


def normalize_retrieval_results(raw: dict) -> dict[str, list[dict]]:
    normalized = {}
    for item in raw.get("results", []):
        query_id = item.get("query_id") or item.get("id")
        ranked = item.get("results") or item.get("matches") or item.get("documents") or []
        if query_id is None:
            raise ValueError("Every retrieval result group must have query_id or id.")
        normalized[str(query_id)] = ranked
    return normalized


def unique_ranked_sources(ranked_results: list[dict]) -> list[str]:
    sources = []
    seen = set()
    for result in ranked_results:
        metadata = result.get("metadata", result)
        source = normalize_source(
            result.get("source")
            or metadata.get("source")
            or metadata.get("file_path")
        )
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def normalized_precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = sum(source in relevant for source in ranked[:k])
    return hits / min(len(relevant), k)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(source in relevant for source in ranked[:k]) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, source in enumerate(ranked[:k], start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, source in enumerate(ranked[:k], start=1):
        if source in relevant:
            hits += 1
            precision_sum += hits / rank
    # Explicit convention: penalize relevant sources not found in the top K.
    denominator = min(len(relevant), k)
    return precision_sum / denominator if denominator else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    dcg = sum(
        (1.0 if source in relevant else 0.0) / math.log2(rank + 1)
        for rank, source in enumerate(ranked[:k], start=1)
    )
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_one_method(
    dataset: dict,
    retrieval_by_id: dict[str, list[dict]],
    k: int,
    method_name: str,
) -> dict:
    per_query = []
    missing_ids = []
    dataset_queries = dataset.get("queries", [])

    for query in dataset_queries:
        query_id = str(query["id"])
        if query_id not in retrieval_by_id:
            missing_ids.append(query_id)
            raw_results = []
        else:
            raw_results = retrieval_by_id[query_id]

        retrieved_sources = unique_ranked_sources(raw_results)
        relevant = {
            normalize_source(source)
            for source in query.get("relevant_sources", [])
            if normalize_source(source)
        }
        row = {
            "query_id": query_id,
            "wstg_id": query.get("wstg_id"),
            "question": query.get("question"),
            "retrieved_chunk_count": len(raw_results[:k]),
            "retrieved_unique_sources_top_k": retrieved_sources[:k],
            "relevant_sources": sorted(relevant),
            "relevant_retrieved_top_k": [
                source for source in retrieved_sources[:k] if source in relevant
            ],
            "normalized_precision_at_k": normalized_precision_at_k(retrieved_sources, relevant, k),
            "recall_at_k": recall_at_k(retrieved_sources, relevant, k),
            "mrr_at_k": mrr_at_k(retrieved_sources, relevant, k),
            "map_at_k": average_precision_at_k(retrieved_sources, relevant, k),
            "ndcg_at_k": ndcg_at_k(retrieved_sources, relevant, k),
        }
        per_query.append(row)

    aggregates = {
        metric: round(mean(row[metric] for row in per_query), 6) if per_query else 0.0
        for metric in METRICS
    }
    return {
        "method": method_name,
        "k": k,
        "evaluation_unit": dataset.get("relevance_unit", "source"),
        "query_count": len(dataset_queries),
        "query_count_with_retrieval_output": len(dataset_queries) - len(missing_ids),
        "missing_query_ids": missing_ids,
        "aggregation": "Macro-average: arithmetic mean across all evaluation questions.",
        "map_convention": "Per-query AP@K divides by min(number of known relevant sources, K).",
        "ndcg_convention": "Binary relevance: relevant source=1, non-relevant source=0.",
        "aggregate_metrics": aggregates,
        "per_query_metrics": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved retrieval results.")
    parser.add_argument("--dataset", required=True, help="Evaluation dataset JSON.")
    parser.add_argument("--retrieval-dir", default="retrieval_experiments",
                        help="Directory containing the four *_retrieval.json files.")
    parser.add_argument("--output-dir", default=None,
                        help="Report directory; defaults to <retrieval-dir>/evaluation.")
    parser.add_argument("--k", type=int, default=None, help="Override K; default from dataset.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    retrieval_dir = Path(args.retrieval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else retrieval_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    k = args.k if args.k is not None else int(dataset.get("k", 5))
    if k <= 0:
        raise ValueError("K must be a positive integer.")

    comparison_rows = []
    detailed_reports = {}
    query_id_set = {str(q["id"]) for q in dataset.get("queries", [])}

    for method_name, filename in METHOD_FILES.items():
        path = retrieval_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing retrieval file: {path}\nRun collect_retrieval_results.py first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        retrieval_by_id = normalize_retrieval_results(raw)
        report = evaluate_one_method(dataset, retrieval_by_id, k, method_name)

        # Carry actual Cohere call outcomes into the detailed report.
        if method_name == "BM25 + vector + RRF + Cohere":
            statuses = {
                str(item.get("query_id")): item.get("reranker_status")
                for item in raw.get("results", [])
            }
            for row in report["per_query_metrics"]:
                row["reranker_status"] = statuses.get(row["query_id"])
            report["cohere_status_summary"] = {
                "success": sum(1 for s in statuses.values() if s and s.get("status") == "success"),
                "fallback": sum(1 for s in statuses.values() if s and s.get("status") == "fallback"),
                "skipped": sum(1 for s in statuses.values() if s and s.get("status") == "skipped"),
            }
        detailed_reports[method_name] = report
        write_path = output_dir / f"{safe_filename(method_name)}_evaluation.json"
        write_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        comparison_rows.append({
            "method": method_name,
            **report["aggregate_metrics"],
        })

    # A missing query in any method means the comparison is incomplete.
    incomplete = {
        method: report["missing_query_ids"]
        for method, report in detailed_reports.items()
        if report["missing_query_ids"]
    }

    comparison = {
        "dataset_name": dataset.get("dataset_name", "WSTG Retrieval Evaluation"),
        "dataset_path": str(dataset_path.resolve()),
        "query_count": len(dataset.get("queries", [])),
        "k": k,
        "evaluation_unit": dataset.get("relevance_unit", "source"),
        "aggregation": "Macro-average across all evaluation questions; metrics averaged separately.",
        "methods": detailed_reports,
        "comparison_rows": comparison_rows,
        "incomplete_methods": incomplete,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_columns = list(METRICS)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *metric_columns])
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Evaluation complete. Reports saved to: {output_dir.resolve()}")
    print(f"Questions: {len(dataset.get('queries', []))}; K={k}")
    print("\nOVERALL MACRO-AVERAGES")
    print(f"{'Method':<34}" + "".join(f"{metric:>16}" for metric in metric_columns))
    for row in comparison_rows:
        print(f"{row['method']:<34}" + "".join(f"{row[m]:>16.4f}" for m in metric_columns))

    cohere_report = detailed_reports.get("BM25 + vector + RRF + Cohere", {})
    status_summary = cohere_report.get("cohere_status_summary")
    if status_summary:
        print("\nCohere reranker call status:", status_summary)
        if status_summary.get("fallback", 0):
            print(
                "WARNING: Some Cohere calls fell back to RRF order. Treat the final "
                "method's aggregate metrics cautiously; recollect after fixing the API issue."
            )
    if incomplete:
        print("\nWARNING: Some methods are missing retrieval results:", incomplete)


def safe_filename(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
