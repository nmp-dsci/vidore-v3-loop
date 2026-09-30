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
| Answer | % correct by an LLM judge against the reference answer | human answers merged by Qwen2.5-VL-32B |
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
| M1 the viewer: FastAPI + React shell, Datasets & questions, Leaderboard | built, in review |
| M2 retrieval pipeline + Search tab · M3 retrieval scored + Runs | designed |
