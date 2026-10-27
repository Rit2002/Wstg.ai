import os
import asyncio
import traceback
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    FactualCorrectness,
)
from ragas.embeddings import HuggingFaceEmbeddings

from rag_pipeline import generate_answer_stream


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

DATASET_PATH = Path("data/wstg_ragas_new_30_questions.json")
QUESTION_LIMIT = 2

METRIC_NAMES = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
]


# ============================================================
# 2. RAGAS EMBEDDINGS
# ============================================================

ragas_embeddings = HuggingFaceEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    device="cpu",
    normalize_embeddings=True,
)


# ============================================================
# 3. LLM JUDGE: OPENROUTER
# ============================================================

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Add it to your .env file."
    )

evaluator_model = os.getenv(
    "RAGAS_EVALUATOR_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)

client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=180.0,
    max_retries=2,
)

# Do not pass provider-specific extra_body arguments unless the selected
# OpenRouter model explicitly supports them.
llm = llm_factory(
    evaluator_model,
    client=client,
    max_tokens=8192,
    extra_body={"reasoning": {"enabled": False}},
)

print(f"RAGAS evaluator model: {evaluator_model}")


# ============================================================
# 4. INITIALIZE RAGAS METRICS
# ============================================================

context_precision_scorer = ContextPrecision(llm=llm)
context_recall_scorer = ContextRecall(llm=llm)
faithfulness_scorer = Faithfulness(llm=llm)
answer_relevancy_scorer = AnswerRelevancy(
    llm=llm,
    embeddings=ragas_embeddings,
)
factual_correctness_scorer = FactualCorrectness(llm=llm)


# ============================================================
# 5. EVALUATE ONE RAG RESPONSE
# ============================================================

async def evaluate_rag(
    query: str,
    reference: str,
    answer: str,
    retrieved_contexts: list[str],
) -> dict:
    """Run each metric independently and return scores and errors."""

    if not query.strip():
        raise ValueError("Query must not be empty.")
    if not reference.strip():
        raise ValueError("Reference answer must not be empty.")
    if not answer.strip():
        raise ValueError("Generated answer must not be empty.")
    if not retrieved_contexts:
        raise ValueError("Retrieved contexts must not be empty.")

    metric_calls = {
        "context_precision": lambda: context_precision_scorer.ascore(
            user_input=query,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
        "context_recall": lambda: context_recall_scorer.ascore(
            user_input=query,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
        "faithfulness": lambda: faithfulness_scorer.ascore(
            user_input=query,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        ),
        "answer_relevancy": lambda: answer_relevancy_scorer.ascore(
            user_input=query,
            response=answer,
        ),
        "factual_correctness": lambda: factual_correctness_scorer.ascore(
            response=answer,
            reference=reference,
        ),
    }

    scores = {}
    metric_errors = {}

    for metric_name, metric_call in metric_calls.items():
        print(f"\nEvaluating {metric_name}...")
        try:
            result = await metric_call()
            score = getattr(result, "value", result)

            if score is None:
                scores[metric_name] = None
                metric_errors[metric_name] = "Metric returned no score."
            else:
                scores[metric_name] = float(score)
                print(f"  Score: {scores[metric_name]:.4f}")

        except Exception as exc:
            scores[metric_name] = None
            metric_errors[metric_name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {metric_errors[metric_name]}")
            traceback.print_exc()

    return {
        "scores": scores,
        "metric_errors": metric_errors,
    }


# ============================================================
# 6. GENERATE ANSWER AND COLLECT RETRIEVED CONTEXTS
# ============================================================

def generate_answer_and_metadata(question: str) -> dict:
    """Collect the streamed answer and retrieved contexts."""

    answer_parts = []
    metadata = None

    for event in generate_answer_stream(question):
        if event.get("type") == "token":
            answer_parts.append(event.get("content", ""))
        elif event.get("type") == "done":
            metadata = event

    answer = "".join(answer_parts).strip()

    if not answer:
        raise RuntimeError("The RAG pipeline returned an empty answer.")
    if metadata is None:
        raise RuntimeError("The RAG pipeline returned no done event.")

    contexts = metadata.get("retrieved_contexts", [])
    if not contexts:
        raise RuntimeError(
            "No retrieved contexts were returned by rag_pipeline.py."
        )

    return {
        "answer": answer,
        "contexts": contexts,
        "metadata": metadata,
    }


# ============================================================
# 7. MAIN EVALUATION LOOP: FIRST TWO QUESTIONS ONLY
# ============================================================

async def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset.get("queries")

    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a 'queries' list.")
    if not questions:
        raise ValueError("The dataset's 'queries' list is empty.")

    questions = questions[:QUESTION_LIMIT]
    print(f"Evaluating {len(questions)} question(s) only.")

    try:
        for index, item in enumerate(questions, start=1):
            query = item.get("question", "")
            reference = item.get("reference_answer", "")
            question_id = item.get("id", index)

            print("\n" + "=" * 72)
            print(f"[{index}/{len(questions)}] Question ID: {question_id}")
            print(f"Question: {query}")

            try:
                generated = await asyncio.to_thread(
                    generate_answer_and_metadata,
                    query,
                )
                answer = generated["answer"]
                contexts = generated["contexts"]

                # print("\nGenerated answer:")
                # print(answer)

                evaluation = await evaluate_rag(
                    query=query,
                    reference=reference,
                    answer=answer,
                    retrieved_contexts=contexts,
                )

                print("\nEvaluation scores:")
                for metric in METRIC_NAMES:
                    score = evaluation["scores"].get(metric)
                    if score is None:
                        print(f"  {metric}: FAILED / no score")
                    else:
                        print(f"  {metric}: {score:.4f}")

                if evaluation["metric_errors"]:
                    print("\nMetric errors:")
                    for metric, error in evaluation["metric_errors"].items():
                        print(f"  {metric}: {error}")

            except Exception as exc:
                print(f"\nQuestion failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    finally:
        await client.close()

    print("\nEvaluation run finished. No result files were written.")


if __name__ == "__main__":
    asyncio.run(main())
