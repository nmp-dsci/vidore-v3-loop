# vidore-v3-loop — every target is a thin wrapper over `uv run vidoreloop …`.
.DEFAULT_GOAL := help
DATASET ?= finance_en
API_PORT ?= 8083
MLFLOW_TRACKING_URI ?= http://localhost:5000

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## python deps (uv) and frontend deps (npm)
	uv sync
	cd frontend && npm ci

datasets: ## the eight public datasets, sizes and pinned revisions
	uv run vidoreloop datasets

download: ## DATASET=<key>|all at its pinned revision → data/cache (CORPUS=1 adds page images)
	uv run vidoreloop download --dataset $(DATASET) $(if $(CORPUS),--corpus)

stats: ## DATASET's queries by language / type / format, and its qrels grades
	uv run vidoreloop stats $(DATASET)

pages: ## prefetch DATASET's page images + OCR text into data/cache/pages (the viewer fetches lazily otherwise)
	uv run vidoreloop pages --dataset $(DATASET)

dev: ## the API on :$(API_PORT), reloading on code changes (UI: cd frontend && npm run dev → :5175)
	uv run vidoreloop serve --port $(API_PORT) --reload

viewer: ## build the frontend and serve it with the API on :$(API_PORT)
	cd frontend && npm run build
	uv run vidoreloop serve --port $(API_PORT)

platform-up: ## start the central MLflow (nmp-central-ai on :5000)
	$(MAKE) -C ../nmp-central-ai up

platform-status: ## preflight: the central MLflow must answer /health
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## zero-LLM run logged to the central MLflow
	uv run vidoreloop smoke

test: ## pytest (offline)
	uv run pytest -q

lint: ## ruff + mypy (+ frontend design lint when node_modules exist)
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy
	@if [ -d frontend/node_modules ]; then cd frontend && npm run lint:design; fi

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

.PHONY: help setup datasets download stats pages dev viewer platform-up platform-status smoke test lint fmt
