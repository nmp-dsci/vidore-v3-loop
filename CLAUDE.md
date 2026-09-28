# CLAUDE.md — vidore-v3-loop

> Read [`AGENTS.md`](./AGENTS.md) first: what this is, the decisions, the layout.
> This file is a pointer plus the rules that bite.

## Quick reference

- `make datasets` · `make download DATASET=hr` · `make stats DATASET=hr` · `make smoke`
- `uv run pytest -q` · `make lint`
- MLflow: central (`make platform-up` → `make -C ../nmp-central-ai up`) → http://localhost:5000

## Rules

- **Billing.** Model calls run on the subscription through the Claude Agent SDK;
  `.env` has `BILLING=subscription` and no `ANTHROPIC_API_KEY`.
- **Datasets are pinned.** Load through `vidore_loop.data.loader`, never with a
  bare `load_dataset` at `main`. A revision change is a registry edit and a note.
- **English, one dataset at a time** (the owner's scope). Do not widen it unasked.
- **The test split is reported, never optimised on.**
- `data/cache/` is never committed. Never add `.lavish/` to `.gitignore`.
