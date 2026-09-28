# Systems that score on ViDoRe V3 — research notes (2026-09-28)

Scores are English-query NDCG@10 averaged over the 5 public English datasets (computer_science,
finance_en, hr, industrial, pharmaceuticals) unless marked. Retriever numbers come from
`mteb_vidore3_scores_2026-09-28.json`, and pipeline numbers from
[illuin-tech/vidore-benchmark results/metrics](https://github.com/illuin-tech/vidore-benchmark/tree/main/results).
"Not reported" means no public number was found.

## Leaderboard, English queries

| System | CS | Fin-EN | HR | Ind | Pharma | Mean of 5 |
|---|---|---|---|---|---|---|
| Pipeline: NeMo agentic (Opus 4.5 + nemotron-colembed-vl-8b-v2) | 84.5 | 75.9 | 74.5 | 63.4 | 74.8 | **74.6** |
| Pipeline: jina-v4 text (NeMo OCR) + zerank-2 | 83.5 | 74.2 | 69.1 | 58.9 | 68.6 | 70.9 |
| Pipeline: llama-nemotron-embed-vl-1b + rerank-vl-1b | 78.0 | 71.8 | 65.7 | 56.5 | 68.5 | 68.1 |
| Pipeline: jina-v4 visual + jina-reranker-m0 | 78.8 | 68.0 | 64.5 | 58.3 | 67.5 | 67.4 |
| Model: tencent EVIE-8B (Apache-2.0) | 82.4 | 73.7 | 70.2 | 61.1 | 71.1 | 71.7 |
| Model: webAI ColVec1-9b (non-commercial) | 81.3 | 72.4 | 72.8 | 61.9 | 69.2 | 71.5 |
| Model: tencent EVIE-4.5B (Apache-2.0) | 82.3 | 72.3 | 69.4 | 61.2 | 70.5 | 71.1 |
| Model: nemotron-colembed-vl-8b-v2 (CC-BY-NC) | 80.3 | 69.6 | 68.4 | 58.6 | 68.6 | 69.1 |
| Model: tomoro-colqwen3-embed-4b (Apache-2.0) | 77.0 | 69.7 | 64.1 | 58.8 | 67.8 | 67.5 |
| Model: Qwen3-VL-Embedding-8B (single vector) | | | | | | 65.1 |
| Model: colqwen2.5-v0.2 | | | | | | 61.2 |
| Model: BM25S (text) | | | | | | 53.3 |

No closed API embedder (Voyage, Cohere, Gemini, OpenAI) has published ViDoRe V3 numbers.

## Systems

- **ViDoRe V3 paper** (arXiv 2601.08620 v2).
  - Retrieval: visual retrievers beat text retrievers; late-interaction beats single-vector. Chunking pages and captioning images did not help.
  - Rerankers: zerank-2 on text goes 50.4 → 63.6 across all languages; jina-reranker-m0 on images goes 57.6 → 57.8.
  - Answers:
    - Best retrieved result: 66.7% correct (ColEmbed-3B-v2 pages as images, GPT-5.2).
    - Oracle pages as images: 72.6% (Gemini 3 Pro).
    - Hard queries: 54.7% with hybrid retrieval, 64.7% with the oracle.
    - Judge: GPT-5.2 at medium effort, binary verdict, σ 0.22 pp.
  - Grounding: bbox F1 0.089 (Qwen3-VL) and 0.065 (Gemini 3 Pro), against 0.602 between human annotators.
- **NVIDIA NeMo Retriever agentic** (2026-03, [blog](https://huggingface.co/blog/nvidia/nemo-retriever-agentic-retrieval),
  [code](https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/pipelines/agentic.py)).
  - The agent is ReAct with three tools: `think`, `retrieve(query, top_k)` and `final_results`.
  - It runs up to 200 steps over a 500-candidate pool and falls back to RRF over all its retrieval calls.
  - It reads page markdown, not images.
  - Score over all 8 sets: 69.22, against 64.36 for the retriever alone.
  - Cost: ≈136 s, ≈760k input tokens and 9.2 retrieval calls per query.
  - With gpt-oss-120b as the agent the score is 66.38.
- **jina-v4 text + zerank-2.** zerank-2 is a 4B Qwen3 cross-encoder trained with zELO (arXiv 2509.12541). It is Apache-2.0 on HF, but its hosted API closed on 2026-09-04, so it has to be self-hosted.
- **Tencent EVIE-8B / 4.5B** (2026-09, Apache-2.0).
  - Trained on 775k query–page pairs. Mined negatives are re-judged: answerable ones become positives and ambiguous ones are masked.
  - The 4.5B is distilled from the 8B.
  - Page vectors compress from ≈750 to 32 without retraining.
- **nemotron-colembed-vl v2** (arXiv 2602.03992). Qwen3-VL backbone with bidirectional attention, ≈773 tokens per page, hard negatives and checkpoint merging. The 8B is CC-BY-NC.
- **Qwen3-VL-Reranker 2B / 8B** (arXiv 2601.04720, Apache-2.0). Reranking the top-100 of Qwen3-VL-Embedding-2B on V3 goes 52.9 → 60.8 (2B) and → 66.7 (8B).
- **SearchWiki / WikiResearcher** (IBM, arXiv 2608.29953). The only agentic system with V3 *answer* accuracy.
  - It compiles pages offline into a wiki and gives the agent BM25+RRF, grep and read_page.
  - Answer accuracy: 70.94 macro (Qwen3.6-27B, no training), against 62.61 for single-turn RAG; 71.35 after RL.
  - Its judge has 3 levels, so its numbers are not comparable with the paper's binary judge.
- **Crop and zoom agents** (DocLens, AgenticOCR, VRAG-RL; none evaluated on V3). Layout-detected crops help: DocLens loses 4.1 points without them. AgenticOCR is RL-trained on V3's human boxes.

## Patterns

1. Fuse a visual channel with a text channel. BM25 on the supplied markdown scores 53.3 on English queries.
2. Reranking is the cheapest big lever: zerank-2 adds 13.2, Qwen3-VL-Reranker-8B adds 13.8, and an LLM reader adds 5.5.
3. For charts, tables and infographics, retrieve on the image and read with image plus text. OCR loses chart evidence.
4. Keep the recall-first ranked top 10 separate from the answer's citations.
5. Give the generator 4–6 pages, and add a verify pass for hard questions.
6. Ground on layout elements: the model cites element ids and we map them to boxes.
7. Spend the agent only on hard query types, and cap its steps.
