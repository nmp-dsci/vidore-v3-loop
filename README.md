# vidore-v3-loop

An agentic RAG system for [ViDoRe V3](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3):
answer enterprise questions over visually rich documents (text, tables, charts,
infographics) by finding the right pages and reasoning over them. It uses the same scored, versioned,
self-improving loop as [tau2-loop](https://github.com/nmp-dsci/tau2-loop) and
[DABStep-loop](https://github.com/nmp-dsci/DABStep-loop).

- **How it is built and why:** [`AGENTS.md`](./AGENTS.md).
- **Plans:** `.lavish/` (research, the scoreboard, and the build plan).
- **Benchmark:** 8 public datasets on the Hub (`vidore/vidore_v3_*`, CC-BY-4.0 annotations),
  19,256 pages, 2,419 human-verified queries in 6 languages; 2 private datasets scored
  only by the [MTEB leaderboard](http://mteb-leaderboard.hf.space). Paper:
  [arXiv 2601.08620](https://arxiv.org/abs/2601.08620).

## What the benchmark scores

| Layer | Metric | Ground truth |
|---|---|---|
| Retrieval | NDCG@10 over pages (graded 2 = full answer, 1 = required part) | human qrels |
| Answer | % correct by an LLM judge against the gold answer | human answers merged by Qwen2.5-VL-32B |
| Grounding | pixel IoU / F1 of cited bounding boxes | human boxes (inter-annotator F1 0.602) |

## Datasets

| key | domain | docs lang | pages | queries |
|---|---|---|---|---|
| finance_en | Finance | en | 2,942 | 309 |
| finance_fr | Finance | fr | 2,384 | 320 |
| computer_science | Computer Science | en | 1,360 | 215 |
| hr | HR | en | 1,110 | 318 |
| energy | Energy | fr | 2,229 | 308 |
| industrial | Industrial | en | 5,244 | 283 |
| pharmaceuticals | Pharmaceuticals | en | 2,313 | 364 |
| physics | Physics | fr | 1,674 | 302 |

Each dataset is pinned to a Hub revision in `src/vidore_loop/data/registry.py`.

## Commands

```
make setup                          # uv sync + npm ci
make datasets                       # the table above, from the registry
make download DATASET=hr            # queries, qrels, metadata → data/cache (add CORPUS=1 for page images)
make stats DATASET=hr               # queries by language / type / format, qrels grades
make smoke                          # zero-LLM run logged to the central MLflow
make pages DATASET=finance_en       # prefetch its page images + OCR text (≈930 MB for finance; lazy otherwise)
make viewer                         # build the UI and serve it with the API → http://127.0.0.1:8083
make dev                            # API with reload on :8083; `cd frontend && npm run dev` → :5175
make index DATASET=finance_en STAGE=text      # BM25S over the OCR text (seconds)
uv sync --extra retrieval                     # torch + sentence-transformers, for the two stages below
make index DATASET=finance_en STAGE=visual    # EVIE-4.5B over every page image (LIMIT=50 for a timing run)
make index DATASET=finance_en STAGE=reranker  # downloads zerank-2 (≈8 GB)
make split DATASET=finance_en                 # our train/test split, once (committed)
make eval DATASET=finance_en WTEXT=0.25       # score a config on train → runs/ + MLflow (runs inside the viewer when it is up)
make eval DATASET=finance_en SPLIT=test FINAL=1  # the chosen config, once, on test
make answers DATASET=finance_en LIMIT=5       # Claude answers (retrieved / oracle), judged → runs/ + MLflow (subscription)
make answers DATASET=finance_en RESUME=<run>  # carry on an interrupted answer run
make runs                                     # every scored run
make test · make lint
```

## Setup

```
make setup
cp .env.example .env
claude login        # the subscription; no ANTHROPIC_API_KEY
make platform-up    # central MLflow (make -C ../nmp-central-ai up) → http://localhost:5000
```

## Status

| Milestone | State |
|---|---|
| M0 scaffold: dataset registry, loader, NDCG/recall, smoke, CI | done |
| Research + plan (`.lavish/s00_vidore-v3-research-plan.html`) | for review |
| Embedder plan (`.lavish/s03_embedders-and-finetune-plan.html`) | dropped (2026-10-24, the owner's call): no fine-tuning |
| M1 the viewer: FastAPI + React shell, Datasets & questions, Leaderboard | done |
| M2 retrieval pipeline + RAG pipeline tab | done (reviewed 2026-10-09): BM25S + EVIE-4.5B (all 2,942 finance pages) → RRF → zerank-2, every stage scored against gold in the tab |
| M3 retrieval scored + Runs | done (reviewed 2026-10-09). Split v3 (2026-10-11): 155 train / 154 test, v2's random 100 / 100 grown to 50/50 with seed 300 (`make split EXTEND=1`). On split v1 every text weight lowered NDCG@10 and the reranker did not beat visual alone, so the champion is fusion with text weight 0. Baseline on v3: **NDCG@10 74.7 train, 71.1 test** (text alone 51.9 / 48.4) |
| M4 answers + judge + audit + Review | done (2026-10-11), scored on split v3: Sonnet 5 answers from our fused top 5 and from the human pages, judged by Opus 5.5 on the paper's prompt. Train **65.8% / 74.8%** (retrieved / oracle), test **69.5% / 74.0%**; earlier answers are reused when Claude would see the same pages. Carried open: the 40-label judge check and the gold-answer audit on the Review tab, due before M6 optimises against the judge |
| M5 the agent + Agent tab | done (2026-10-25; plan `.lavish/s04_m5-agent-plan.html`, S0–S2 report `.lavish/s05_m5-s0-s2-results.html`). S0: models named by alias only, resolved per run (`vidoreloop models`). S1 (2026-10-24): Sonnet 5.5 (high) is the judge, matching Opus on 95.8% of M4's 310 train verdicts; Haiku 5.5 matched on 87.1% and scored 12.9 points stricter (`vidoreloop rejudge`). S2: Haiku 5.5 (high) answers train at **61.3% / 65.8%** (retrieved / oracle, Sonnet judge) against Sonnet 5's 65.2% / 74.8%, so the agent starts on Sonnet 5.5. S2b: Sonnet 5.5 (medium, judged by Sonnet 5.5 medium) answers train at **68.4% / 76.1%**. S3: the agent (`vidoreloop agent`: search + view_page, a structured reply), the Agent tab, and an all-309 figure under each Leaderboard row. S4 smoke (10 train questions): NDCG@10 67.8 against the pipeline's 55.5 on the same questions, 5 of 10 answers correct. S5 (all 309, Sonnet 5.5 medium agent and judge): NDCG@10 **74.3 train / 67.8 test / 71.1 all** against the pipeline's 74.7 / 71.1 / 72.9; accuracy **63.9% train / 68.2% test / 66.0% all** against the board's best top-5 67.3%. The agent mostly searches once, views 1–2 pages and ranks 2 pages itself; it rewrites the question into keywords on 307 of 308 first searches, and the same search scores 3.8 (train) / 4.1 (test) NDCG@10 lower on its words than on the question as written, so the pipeline still leads. Carried open: start the agent from the question as written and have it read the top pages (train first), re-check the Sonnet judge at medium (S1 checked it at high), and the 40-label judge check |
