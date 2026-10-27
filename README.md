# WSTG Intelligence

**Hybrid RAG over the OWASP Web Security Testing Guide, evaluated on retrieval quality and on answer quality.**

A retrieval-augmented question answering system over the
[OWASP Web Security Testing Guide](https://github.com/OWASP/wstg). Retrieval combines BM25 and
dense vector search with Reciprocal Rank Fusion (RRF) and a Cohere reranker, and answers are
generated from the retrieved chunks. Evaluation is treated as a first-class part of the project:
retrieval is benchmarked offline on 113 labeled questions, and end-to-end answer quality is
measured with RAGAS on 29 questions using an LLM judge.

---

## Results at a Glance

**Retrieval quality** (full pipeline: BM25 + vector + RRF + Cohere, chunk 256 / overlap 64, K = 5,
113 questions in 12 datasets)

| Norm. Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 |
|:---:|:---:|:---:|:---:|:---:|
| 0.8558 | 0.8551 | 0.8510 | 0.8429 | 0.8478 |

**Answer quality** (RAGAS, 29 questions, LLM judge)

| Faithfulness | Answer relevancy | Factual correctness | Context precision | Context recall |
|:---:|:---:|:---:|:---:|:---:|
| 0.9357 | 0.8344 | 0.6524 | 0.6570 | 0.5215 |

The two layers tell different stories. The retriever reliably surfaces the right WSTG page
(Recall@5 of 0.86). The generated answers are well grounded in the retrieved text (faithfulness
0.94) and on topic (answer relevancy 0.83), but they agree with the reference answer only about
two-thirds of the time (factual correctness 0.65), and passage-level context quality (precision
0.66, recall 0.52) is the weakest area. Details and caveats are in [Part 2](#part-2-answer-quality-ragas).

---

## Architecture

```
Ingestion:   WSTG Markdown → heading split → token chunks → embed (bge) → Pinecone + JSONL
Retrieval:   BM25 (top 30) + Pinecone (top 25) → RRF (k=60) → Cohere rerank (top 20) → top 5
Generation:  top-5 chunks → LLM answer
Evaluation:  (1) retrieval metrics, 113 questions, 12 datasets, file-level relevance
             (2) RAGAS answer quality, 29 questions, LLM judge via OpenRouter
```

**Models:** `BAAI/bge-base-en-v1.5` embeddings (768-dim, local), Pinecone vector store,
BM25Okapi, Cohere `rerank-v4.0-fast`.
**Corpus:** OWASP WSTG test-case pages (Markdown). Chunks keep `source`, `filename` and a
`heading_path`; the title/ID block is merged with the first section.

**Design notes**

- **Hybrid candidate generation.** BM25 covers exact terms (tool names, payloads, test IDs) and
  the dense index covers paraphrased questions. Their ranked lists are merged with RRF, which
  works on ranks only and needs no score normalization.
- **Rerank as a final precision stage.** Cohere reorders a candidate pool of 20 and the top 5 go
  to the generator.
- **Graceful degradation.** Cohere calls are throttled, retried, and fall back to RRF order with
  a warning if a call fails, so a single API failure does not abort an evaluation run.

---

## How Quality Is Measured

Retrieval and generation fail in different ways, so they are evaluated separately.

| | Part 1: Retrieval | Part 2: Answer quality |
|---|---|---|
| **Question answered** | Does the right WSTG page reach the top 5? | Is the generated answer grounded, relevant and correct? |
| **Method** | Offline metrics against labeled source files | RAGAS metrics scored by an LLM judge |
| **LLM judge** | No | Yes (`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter) |
| **Questions** | 113 (12 datasets) | 29 |
| **Granularity** | File level | Chunk and claim level |

The two sets of numbers measure different things and should not be compared directly. For
example, file-level Recall@5 and RAGAS context recall are different quantities.

---

## Part 1: Retrieval Quality

### Methodology

Each question is labeled with the WSTG file(s) that contain its answer (`relevant_sources`). A
retrieved chunk is relevant if its source file is labeled, and repeated chunks from one file
count once. No LLM judge is involved.

**Metrics (K = 5):**
- **Normalized Precision@K**: relevant in top K / min(|R|, K), where R is the set of relevant sources
- **Recall@K**: relevant in top K / |R|
- **MRR@K**: 1 / rank of the first relevant source
- **MAP@K**: average precision, divided by min(|R|, K)
- **NDCG@K**: binary relevance with a log₂ discount

Raw Precision@5 is capped at 0.2 when a question has one relevant source, so normalized
precision is reported instead. Each dataset is scored separately and the **12 dataset results
are averaged with equal weight**. BM25 is evaluated as a baseline but not shown below.

### A. Chunk size 256 / overlap 64 (latest run)

| Method | Norm. Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Pinecone vector | 0.8458 | 0.8458 | 0.7938 | 0.7892 | 0.8048 |
| BM25 + vector + RRF | 0.8350 | 0.8343 | 0.8030 | 0.7942 | 0.8069 |
| **BM25 + vector + RRF + Cohere** | **0.8558** | **0.8551** | **0.8510** | **0.8429** | **0.8478** |

Cohere reranking is best on every metric. It reorders the same candidate pool, so the largest
gain is in ranking quality (MRR 0.8030 → 0.8510 over RRF alone).
Full-precision values: [`average_metrics_normalized_chunk256_overlap64.csv`](average_metrics_normalized_chunk256_overlap64.csv).

### B. Chunk size 350 / overlap 50 (initial run)

| Method | Norm. Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Pinecone vector | **0.8558** | **0.8551** | 0.7990 | 0.7935 | 0.8108 |
| BM25 + vector + RRF | 0.8392 | 0.8384 | 0.8144 | 0.8097 | 0.8184 |
| **BM25 + vector + RRF + Cohere** | **0.8558** | **0.8551** | **0.8454** | **0.8374** | **0.8436** |

Cohere again leads on MRR, MAP and NDCG and ties Pinecone on precision and recall.
Full-precision values: [`average_metrics_normalized_12_datasets.csv`](average_metrics_normalized_12_datasets.csv).

### A vs B: effect of the smaller chunks

Smaller chunks help only the full pipeline, and only slightly (MRR +0.0056, NDCG +0.0042).
They lower the scores of the other two methods: Pinecone recall drops 0.8551 → 0.8458, and RRF
MRR drops 0.8144 → 0.8030.

### Retrieval takeaways

1. **Reranking is worth it.** Cohere gives the best MRR, MAP and NDCG in both runs.
2. **Chunk size matters less than the pipeline.** 256 / 64 gives the full pipeline a small gain
   but hurts vector-only and RRF retrieval, so the best config depends on the stage.
3. **RRF improves ranking over vector search alone** (MRR 0.8030 vs 0.7938 at 256 / 64) but has
   slightly lower recall, so it needs the reranker to pay off.

---

## Part 2: Answer Quality (RAGAS)

Retrieval metrics say whether the right page was found. They do not say whether the answer
written from those chunks is grounded, relevant or correct, which is what a user actually sees.
Part 2 measures that with [RAGAS](https://github.com/explodinggradients/ragas).

### Setup

- **Questions:** 29, each with a reference answer
- **Judge (LLM as a judge):** `nvidia/nemotron-3-ultra-550b-a55b:free`, set through `RAGAS_EVALUATOR_MODEL`
- **Inference:** OpenRouter
- **Scale:** every metric is in [0, 1], higher is better

| Metric | What it measures |
|---|---|
| Faithfulness | Share of claims in the answer that are supported by the retrieved context |
| Answer relevancy | How directly the answer addresses the question |
| Factual correctness | Claim-level agreement between the answer and the reference answer |
| Context precision | Whether the useful retrieved chunks are ranked ahead of the noise |
| Context recall | Share of the reference answer's claims that the retrieved context covers |

### Results (29 questions)

| Metric | Mean | Median | Std | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|
| Faithfulness | 0.9357 | 0.9583 | 0.070 | 0.760 | 1.000 |
| Answer relevancy | 0.8344 | 0.8383 | 0.081 | 0.676 | 0.995 |
| Factual correctness | 0.6524 | 0.7400 | 0.282 | 0.060 | 1.000 |
| Context precision | 0.6570 | 0.9167 | 0.418 | 0.000 | 1.000 |
| Context recall | 0.5215 | 0.6667 | 0.364 | 0.000 | 1.000 |

### Analysis

**Generation is grounded and on topic.**
- Faithfulness is high and tightly clustered: 21 of 29 answers score 0.9 or above, 8 are
  perfect, and the worst is 0.76. No answer is badly unsupported by its own context.
- Answer relevancy has a narrow range (0.68 to 0.995) with no off-topic answers. This metric
  rarely reaches 1.0 by construction, so 0.83 should not be read as 17% irrelevant.

**Correctness is the metric with real headroom.**
- Factual correctness has a median of 0.74 but a wide spread: 7 answers score 0.9 or above, 14
  fall between 0.5 and 0.9, and 8 fall below 0.5 (the lowest is 0.06).
- Of those 8 low-correctness answers, 6 have faithfulness of 0.9 or above. In other words, the
  model usually stays consistent with its context even when that answer disagrees with the
  reference. A faithful answer built on incomplete or off-target context is still a wrong answer.

**Context precision and recall are bimodal.**
- Context precision has 12 perfect scores and 7 zeros, which is why the median (0.92) sits far
  above the mean (0.66). Context recall has 6 perfect scores and 7 zeros.
- Three questions score 0 on both, meaning the retrieved context contained nothing useful for
  the reference answer. These are the clearest retrieval failures in the set.
- Only 4 of 29 questions score 0.9 or above on both context precision and context recall.

**Retrieval scores explain correctness only weakly in this sample.**
- Pearson correlations with factual correctness are low: 0.18 for context recall and 0.23 for
  context precision. Mean factual correctness is 0.67 when context recall is at least 0.67
  (n = 15) and 0.63 below it (n = 14).
- With n = 29, only the faithfulness / answer-relevancy pair (r ≈ 0.51) clearly exceeds noise.
  The other relationships are not statistically distinguishable from zero.
- So the data does not support the claim that retrieval alone drives answer errors. Two
  questions with context recall below 0.5 still reach factual correctness of 0.8 or higher.

**Overall.** Only 1 of 29 questions scores 0.8 or above on all five metrics, but the pattern is
asymmetric: grounding and relevance are strong, while correctness and context coverage are where
the system loses points.

### Reliability of these numbers

- **Small sample.** The standard error of the mean is about 0.05 for factual correctness, 0.07
  for context recall and 0.08 for context precision. One question moving from 0 to 1 shifts an
  average by about 0.03.
- **Single run, free judge.** One pass with a free-tier judge model, so some of the variance is
  judge noise rather than system behavior. Repeated runs or a stronger judge would tighten this.
- **Metric limits.** Claim-level correctness is sensitive to how the reference answer is
  phrased, so some low scores may reflect wording differences rather than wrong answers.

---

## Combined Reading of the Results

1. **The retriever is solid at page level.** Hybrid retrieval with reranking reaches 0.86
   Recall@5 and 0.85 MRR@5, and the reranker is the single most useful component.
2. **The generator is faithful, not necessarily correct.** High faithfulness (0.94) shows the
   answers follow their context. Factual correctness (0.65) shows that following the context is
   not enough when the context is incomplete.
3. **Page-level success hides passage-level misses.** Retrieval is labeled at file level, so it
   can score a hit even when the chunks shown to the generator lack the needed passage. The gap
   between Part 1 (about 0.85) and RAGAS context precision and recall (0.66 and 0.52) is
   consistent with that, although the two use different questions and definitions, so it is a
   hypothesis and not a measurement.
4. **Both layers are needed.** Retrieval metrics alone would have made this system look
   stronger than its answers are.

---

## Limitations

- Retrieval relevance labels are at file level. They check whether the right page is found, not
  the best passage.
- Retrieval averages are equal-weighted across datasets, not across the 113 questions, and
  differences between configs are small (about 0.01 or less), so treat them as indicative.
- The RAGAS evaluation covers 29 questions, one run, and a free judge model, so treat it as a
  health check and not a benchmark.
- The two evaluations use different question sets, so their scores are not directly comparable.

## Next Steps

- Review the 8 low-correctness answers by hand to separate retrieval gaps, reference-phrasing
  artifacts and judge noise.
- Add passage-level relevance labels so retrieval can be scored on chunks, not files.
- Repeat the RAGAS run, and compare against a stronger judge, to measure judge variance.
- Grow the RAGAS question set beyond 29 so per-metric differences become resolvable.

---

## Setup

```bash
pip install python-dotenv rank-bm25 cohere transformers ragas \
    langchain-core langchain-huggingface langchain-pinecone langchain-text-splitters
# create .env with:  PINECONE_API_KEY=...   COHERE_API_KEY=...   OPENROUTER_API_KEY=...
```

Optional overrides: `EMBEDDING_MODEL`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`,
`COHERE_RERANK_MODEL`, `WSTG_REPO_PATH`, `RAGAS_EVALUATOR_MODEL`.

## Run Order

Replace `<dataset_path>` and `<output_dir>` with the relevant paths for each evaluation dataset.

```bash
# 1. Build the chunk corpus and vector index
#    (set CHUNK_SIZE / CHUNK_OVERLAP in ingest.py first: 350 / 50 or 256 / 64)
python ingest.py

# 2. Collect retrieval results
python collect_retrieval_results.py --dataset <dataset_path> --output-dir <output_dir>

# 3. Evaluate retrieval metrics
python evaluate_retrieval_results.py --dataset <dataset_path> --retrieval-dir <output_dir> --output-dir <output_dir>/evaluation

# 4. (Optional) End-to-end RAGAS evaluation, judged through OpenRouter
python evaluate_ragas.py
```

**Example** for `wstg_injection_testing_eval_20.json`:

```bash
python collect_retrieval_results.py --dataset data/wstg_injection_testing_eval_20.json --output-dir retrieval_experiments/injection_testing
```

```bash
python evaluate_retrieval_results.py --dataset data/wstg_injection_testing_eval_20.json --retrieval-dir retrieval_experiments/injection_testing --output-dir retrieval_experiments/injection_testing/evaluation
```

**Execution order:** run the collection command first, followed by the evaluation command. The first generates retrieval results for the configured methods; the second calculates Normalized Precision@K, Recall@K, MRR@K, MAP@K, and NDCG@K. Use `--resume` on the collection command to continue an interrupted run.

---

## Project Structure

```
├── ingest.py                      # chunk WSTG, save JSONL, index in Pinecone
├── hybrid_retriever.py            # BM25 + Pinecone + RRF + Cohere
├── reranker.py                    # Cohere rerank (throttle, retry, fallback)
├── utils.py                       # chunk key for fusion/caching
├── collect_retrieval_results.py   # Part A: save ranked results
├── evaluate_retrieval_results.py  # Part B: compute metrics
├── evaluate_ragas.py              # end-to-end RAGAS scoring (LLM judge via OpenRouter)
├── average_metrics_normalized_*.csv  # averaged results per chunk config
└── data/                          # wstg_chunks.jsonl + evaluation datasets
```

## Known Setup Gotchas

- The Pinecone index must already exist with 768 dimensions. `ingest.py` adds vectors to the
  chosen namespace, so use a separate `PINECONE_NAMESPACE` for each chunk configuration to
  avoid mixing chunkings.
- `ingest.py` overwrites `data/wstg_chunks.jsonl`, which BM25 reads, so re-collect results
  after changing chunk settings.
- Cohere calls are throttled to one per 6.5 seconds, so a full collection run is slow. If
  Cohere fails for a query the pipeline falls back to RRF order and a warning is printed.
