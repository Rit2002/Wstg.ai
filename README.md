# 🔍 WSTG Intelligence: Hybrid RAG Retrieval over OWASP WSTG

A hybrid retrieval pipeline (BM25 + Pinecone + RRF + Cohere rerank) over the
[OWASP Web Security Testing Guide](https://github.com/OWASP/wstg), with an offline
**evaluation harness** on 113 questions and a chunk-size comparison, not just a demo.


---

## Architecture

```
Ingestion:  WSTG Markdown → heading split → token chunks → embed (bge) → Pinecone + JSONL
Retrieval:  BM25 (top 30) + Pinecone (top 25) → RRF (k=60) → Cohere rerank (top 20) → top 5
Evaluation: offline retrieval metrics, 113 questions in 12 datasets, file-level relevance
End-to-end: answer generation on the retrieved chunks, scored with RAGAS (29 questions)
```

**Models:** `BAAI/bge-base-en-v1.5` embeddings (768-dim, local), Pinecone vector store,
BM25Okapi, Cohere `rerank-v4.0-fast`.
**Corpus:** OWASP WSTG test-case pages (Markdown). Chunks keep `source`, `filename` and a
`heading_path`; the title/ID block is merged with the first section.

---

## Evaluation Methodology

Retrieval is scored **without an LLM judge**. Each question is labeled with the WSTG
file(s) that contain its answer (`relevant_sources`). A retrieved chunk is relevant if
its source file is labeled, and repeated chunks from one file count once.

**Metrics (K = 5):**
- **Normalized Precision@K**: relevant in top K / min(|R|, K), where R is the set of relevant sources
- **Recall@K**: relevant in top K / |R|
- **MRR@K**: 1 / rank of the first relevant source
- **MAP@K**: average precision, divided by min(|R|, K)
- **NDCG@K**: binary relevance with a log₂ discount

Raw Precision@5 is capped at 0.2 when a question has one relevant source, so normalized
precision is reported instead. Each dataset is scored separately and the **12 dataset
results are averaged with equal weight**. BM25 is evaluated as a baseline but not shown below.

---

## Key Findings

### A. Chunk size 256 / overlap 64 (latest run)

| Method | Norm. Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Pinecone vector | 0.8458 | 0.8458 | 0.7938 | 0.7892 | 0.8048 |
| BM25 + vector + RRF | 0.8350 | 0.8343 | 0.8030 | 0.7942 | 0.8069 |
| **BM25 + vector + RRF + Cohere** | **0.8558** | **0.8551** | **0.8510** | **0.8429** | **0.8478** |

Cohere reranking is best on every metric. It reorders the same candidate pool, so the
largest gain is in ranking quality (MRR 0.8030 → 0.8510 over RRF alone).
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
They lower the scores of the other two methods: Pinecone recall drops 0.8551 → 0.8458, and
RRF MRR drops 0.8144 → 0.8030.

---

## End-to-End Evaluation (RAGAS)

The retrieval metrics above only tell me whether the right WSTG page shows up. They say
nothing about the answer the LLM writes from those chunks, so I also ran RAGAS on a separate
set of 29 questions.

- **Questions:** 29
- **Judge (LLM as a judge):** `nvidia/nemotron-3-ultra-550b-a55b:free`, set through `RAGAS_EVALUATOR_MODEL`
- **Inference:** OpenRouter

| Metric | Average (29 questions) |
|---|:---:|
| Context precision | 0.6570 |
| Context recall | 0.5215 |
| Faithfulness | 0.9357 |
| Answer relevancy | 0.8344 |
| Factual correctness | 0.6524 |

What I take from this:

- **Faithfulness is high (0.94).** 21 of the 29 answers scored 0.9 or above and the lowest was
  0.76, so the model mostly sticks to what it was given instead of making things up.
- **Answer relevancy is decent (0.83).** Scores ranged from about 0.68 to 0.995, and no
  answer was completely off topic.
- **Context precision (0.66) and recall (0.52) are the weak spots.** Seven questions got a 0
  on each, and three got a 0 on both. In those cases the retrieved chunks just didn't
  contain what the reference answer needed.
- **Factual correctness (0.65) is all over the place.** It runs from 0.06 to 1.0, with 8
  questions below 0.5.

Put together, the answers are rarely invented. When one is wrong or incomplete, it is usually
because the right context didn't make it into what the model saw, which points back at
retrieval rather than generation.

These numbers are not comparable to the Precision@5 / Recall@5 tables above. Those check
file-level labels, while RAGAS has the judge LLM compare the retrieved text against a
reference answer.

---

## Takeaways

1. **Reranking is worth it.** Cohere gives the best MRR, MAP and NDCG in both runs.
2. **Chunk size matters less than the pipeline.** 256 / 64 gives the full pipeline a small
   gain but hurts vector-only and RRF retrieval, so the best config depends on the stage.
3. **RRF improves ranking over vector search alone** (MRR 0.8030 vs 0.7938 at 256 / 64) but
   has slightly lower recall, so it needs the reranker to pay off.
4. **End to end, retrieval is the weak link.** In the RAGAS run faithfulness is 0.94 but
   context recall is only 0.52, so improving what gets retrieved should help more than
   changing the generator.

Honest note: relevance labels are at file level, so this checks whether the right page is
found, not the best passage. Averages are equal-weighted across datasets (not across the 113
questions), and differences between configs are small (about 0.01 or less), so treat them as
indicative. The RAGAS scores are a single run with a free judge model on 29 questions, so
one question moving changes an average by around 0.03. Read them as a rough health check,
not a benchmark.

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