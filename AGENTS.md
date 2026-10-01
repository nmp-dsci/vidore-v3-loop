# AGENTS.md — vidore-v3-loop

The source of truth for how this project is built and why. `CLAUDE.md` points
here. Plans live in `.lavish/` (never gitignored).

## 1 · What it is

A RAG system over [ViDoRe V3](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)
(ILLUIN Technology + NVIDIA, [arXiv 2601.08620](https://arxiv.org/abs/2601.08620)).
Given a question, it finds the relevant pages in a corpus of enterprise PDFs
and answers from them. It is scored on three layers: retrieval (NDCG@10),
the answer (LLM judge vs the reference), and grounding (bounding-box IoU/F1). An error
loop improves it, the same loop as tau2-loop and DABStep-loop.

## 2 · Decisions, and the reasons

| Decision | Choice | Why |
|---|---|---|
| Data | the 8 public datasets from the Hub, each pinned by revision (`data/registry.py`); `paper_revision` recorded beside it | the benchmark moves; a run must name the bytes it read |
| Scope | English datasets, one at a time (`finance_en`, `computer_science`, `hr`, `industrial`, `pharmaceuticals`), English queries first | the owner's call: depth on one corpus before breadth |
| Split | ours, per dataset, not built yet; upstream has a single `test` split | nothing is held out upstream, so the loop needs its own train/test |
| Metrics | NDCG@10 with linear graded gain, as `pytrec_eval`/MTEB | numbers comparable with the leaderboard |
| Billing | the Claude subscription via the Claude Agent SDK, `BILLING=subscription`, no key | the portfolio rule since tau2-loop |
| Tracking | the central MLflow, experiment `vidore-v3-loop/evals`, required tags `project`, `git_sha`, `env`, `billing` | PLATFORM.md |
| Milestones | vertical slices: each ships its backend piece and the viewer tab that shows it (M1 viewer, M2 retrieval pipeline + RAG pipeline tab (answers join it in M4), M3 retrieval scored + Runs, M4 answers + Review, M5 agent + Agent, M6 loop + Optimise, M7 grounding) | the owner's call: every milestone ends with something to open in the browser |
| Viewer | React 18 + Vite + TypeScript + react-router, served by FastAPI (API :8083, dev UI :5175), tau2-loop's and DataAgentBench's layout and URL grammar | one portfolio, one way of reading a run |
| Retrieval (M2) | text BM25S + visual EVIE-4.5B (Apache-2.0, 128-d Matryoshka cut) → RRF → zerank-2 (Apache-2.0), all local; heavy deps in the `retrieval` extra; a stage runs only when its index and weights are on disk | the plan's §7c; no request may wait on a multi-GB download, and CI never needs torch |
| Memory | one large model loaded at a time while building; the two (≈9 GB + ≈8 GB) only coexist in the server | a 32 GB machine killed parallel loads on 2026-10-01 |
| Project colour | orange `--accent` (#A8440B light, #F2945A dark); `--ok` green carries "passed"; `--amber` retuned to olive (#6F5A00 / #CDB04A) for "partial" so it never reads as the accent | the owner's call; a recorded divergence from the site palette, as tau2-loop's red |

## 3 · Layout

```
src/vidore_loop/
  config.py            paths + settings (the only reader of os.environ)
  cli.py               `vidoreloop` — datasets, download, stats, smoke
  data/registry.py     the 8 datasets, revisions, sizes
  data/loader.py       load a config at its revision into data/cache/
  eval/metrics.py      NDCG@k, recall@k over graded qrels
  tracking/mlflow_log.py  preflight, required tags, smoke run
  data/catalog.py      English questions shaped for the viewer; hand checks
  data/pages.py        page images + OCR text by row group → data/cache/pages/
  data/leaderboard.py  the boards from docs/research/
  serving/app.py       FastAPI: /api/datasets, /queries, /pages, /leaderboard; serves frontend/dist , /api/search, /api/retrieval/status
  retrieval/text.py    channel A: BM25S over OCR markdown (+ document name) → data/cache/index/<key>/text/
  retrieval/visual.py  channel B: EVIE-4.5B multi-vectors (128-d, fp16), MaxSim → …/visual/
  retrieval/fuse.py    stage C: reciprocal rank fusion (K = 60) → top 50
  retrieval/rerank.py  stage D: zerank-2 cross-encoder over the 50 → top 10
  retrieval/pipeline.py  one question through the stages, scored by eval/metrics.report
frontend/              React + Vite viewer (:5175 dev), orange tokens, design lint
data/audits/           hand checks of reference answers (hand_checks.jsonl)
docs/research/         the committed sources behind every board number
tests/                 offline only
data/splits/           our committed train/test cut (to come)
ai_specs/              specs the coding agents executed
```

## 4 · Conventions

- `uv` for everything Python; Ruff (line 100); mypy strict; pytest offline only.
- Never commit `.env`, keys or `data/cache/`. Never add `.lavish/` to `.gitignore`.
- A number on a page has its denominator; a figure has a committed source.
- Run folders will be immutable once scored. Fix the code and re-run.
- The test split is reported, never optimised on.
- Commit messages: imperative subject, a body that says why.
