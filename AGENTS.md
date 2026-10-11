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
| Split | ours, per dataset, seed 300, committed in `data/splits/`. v3 (2026-10-11, finance 155 / 154): v2's stratified random 100 train / 100 test kept, its 109-question holdout dealt to the side with fewer in each query-type group (`make split EXTEND=1`). Retired splits and their runs: `data/splits/archive/`, `runs/archive/split_v1/`, `runs/archive/split_v2/` | nothing is held out upstream, so the loop needs its own train/test; growing a split never moves a scored question, so earlier answers stay reusable |
| Metrics | NDCG@10 with linear graded gain, as `pytrec_eval`/MTEB | numbers comparable with the leaderboard |
| Billing | the Claude subscription via the Claude Agent SDK, `BILLING=subscription`, no key | the portfolio rule since tau2-loop |
| Tracking | the central MLflow, experiment `vidore-v3-loop/evals`, required tags `project`, `git_sha`, `env`, `billing` | PLATFORM.md |
| Milestones | vertical slices: each ships its backend piece and the viewer tab that shows it (M1 viewer, M2 retrieval pipeline + RAG pipeline tab (answers join it in M4), M3 retrieval scored + Runs, M4 answers + Review, M5 agent + Agent, M6 loop + Optimise, M7 grounding) | the owner's call: every milestone ends with something to open in the browser |
| Viewer | React 18 + Vite + TypeScript + react-router, served by FastAPI (API :8083, dev UI :5175), tau2-loop's and DataAgentBench's layout and URL grammar | one portfolio, one way of reading a run |
| Retrieval (M2) | text BM25S + visual EVIE-4.5B (Apache-2.0, 128-d Matryoshka cut) → RRF → zerank-2 (Apache-2.0), all local; heavy deps in the `retrieval` extra; a stage runs only when its index and weights are on disk | the plan's §7c; no request may wait on a multi-GB download, and CI never needs torch |
| Answers (M4) | Claude answers in two settings: `retrieved` (top 5 of a finished retrieval run) and `oracle` (every human-marked page); the paper's no-document setting is dropped (the owner's call: it measures model memory, not the system); each page as image + OCR text; the paper's prompts word for word (arXiv 2601.08620 Figs 24–25); answerer Sonnet 5, judge Opus 5.5, both at medium effort | numbers that sit beside the paper's Table 3; an answer run names the ranking it read, so it never re-retrieves |
| Answer reuse | an answer run copies a setting's answer and verdict from any earlier run (archived too) that gave Claude the same pages in the same order under the same models, effort, modality and prompts; it names `cached_from` and its tokens are not counted again (`--no-reuse` to refuse) | a split change or a rerun should not pay twice for an identical call |
| Judge | the paper's three-level judge; only `Correct` counts as right (its binary label, App. I); a person's label on the Review tab overrules it in the audited score, and 40 labels calibrate it | a judge scoring against a wrong reference fails a correct system (2 of 5 hand checks were off) |
| Audit | `data/audits/references.jsonl` (disputed / holds) and `judge_labels.jsonl`, append-only, latest wins; accuracy reported raw (the judge) and audited (disputed left out, labels applied) | s00 §8: a disputed failure never drives an optimiser edit |
| Memory | one large model loaded at a time while building; the two (≈9 GB + ≈8 GB) only coexist in the server | a 32 GB machine killed parallel loads on 2026-10-01 |
| Project colour | orange `--accent` (#A8440B light, #F2945A dark); `--ok` green carries "passed"; `--amber` retuned to olive (#6F5A00 / #CDB04A) for "partial" so it never reads as the accent | the owner's call; a recorded divergence from the site palette, as tau2-loop's red |

## 3 · Layout

```
src/vidore_loop/
  config.py            paths + settings (the only reader of os.environ)
  cli.py               `vidoreloop` — datasets, download, stats, smoke, split, eval, answer, runs
  data/registry.py     the 8 datasets, revisions, sizes
  data/loader.py       load a config at its revision into data/cache/
  eval/metrics.py      NDCG@k, recall@k over graded qrels
  tracking/mlflow_log.py  preflight, required tags, smoke run, retrieval runs
  eval/split.py        our train/test split → data/splits/<key>.json (written once)
  eval/runs.py         a scored run: a config over a split → runs/<id>/ (run.json + queries.jsonl), MLflow; the test guard
  data/catalog.py      English questions shaped for the viewer; hand checks
  data/pages.py        page images + OCR text by row group → data/cache/pages/
  data/leaderboard.py  the boards from docs/research/
  serving/app.py       FastAPI: /api/datasets, /queries, /pages, /leaderboard, /api/search, /api/retrieval/status, /api/runs (+ POST, runs in-process), /api/split, /api/answer, /api/answer-runs, /api/runs/{id}/answers, /api/audits/*; serves frontend/dist
  retrieval/text.py    channel A: BM25S over OCR markdown (+ document name) → data/cache/index/<key>/text/
  retrieval/visual.py  channel B: EVIE-4.5B multi-vectors (128-d, fp16), MaxSim → …/visual/
  retrieval/fuse.py    stage C: weighted reciprocal rank fusion (K = 60) → top 50; WEIGHTS chosen on train
  retrieval/rerank.py  stage D: zerank-2 cross-encoder over the 50 → top 10
  retrieval/pipeline.py  compute(): every stage's ranking (shared by the tab and runs); run(): one question for the tab
  answer/claude.py     one sealed Agent SDK call on the subscription (no tools, allow-listed env, waits out a session limit)
  answer/generate.py   the paper's answer prompts; pages as image + OCR text; settings retrieved / oracle / none
  answer/judge.py      the paper's judge prompt, a JSON-schema verdict; only Correct counts
  eval/answers.py      an answer run → runs/<id>/ (run.json + answers.jsonl), resumable; raw and audited accuracy, MLflow
  data/audits.py       the Review tab's writes: disputed references, a person's label on a judged answer
frontend/              React + Vite viewer (:5175 dev), orange tokens, design lint
data/audits/           hand checks of gold answers, disputed golds, judge labels (committed)
docs/research/         the committed sources behind every board number
tests/                 offline only
data/splits/           our committed train/test cut
runs/                  scored runs, committed (run.json + queries.jsonl, or answers.jsonl for an answer run)
ai_specs/              specs the coding agents executed
```

## 4 · Conventions

- `uv` for everything Python; Ruff (line 100); mypy strict; pytest offline only.
- Never commit `.env`, keys or `data/cache/`. Never add `.lavish/` to `.gitignore`.
- A number on a page has its denominator; a figure has a committed source.
- Run folders are immutable once scored. Fix the code and re-run.
- A run on test needs `--final`, and a configuration already scored on test is refused without `--force`.
- The test split is reported, never optimised on.
- Commit messages: imperative subject, a body that says why.
