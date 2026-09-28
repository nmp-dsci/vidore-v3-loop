# vidore-v3-loop — every target is a thin wrapper over `uv run vidoreloop …`.
.DEFAULT_GOAL := help
DATASET ?= hr
MLFLOW_TRACKING_URI ?= http://localhost:5000

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## python deps (uv)
	uv sync

datasets: ## the eight public datasets, sizes and pinned revisions
	uv run vidoreloop datasets

download: ## DATASET=<key>|all at its pinned revision → data/cache (CORPUS=1 adds page images)
	uv run vidoreloop download --dataset $(DATASET) $(if $(CORPUS),--corpus)

stats: ## DATASET's queries by language / type / format, and its qrels grades
	uv run vidoreloop stats $(DATASET)

platform-up: ## start the central MLflow (nmp-central-ai on :5000)
	$(MAKE) -C ../nmp-central-ai up

platform-status: ## preflight: the central MLflow must answer /health
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## zero-LLM run logged to the central MLflow
	uv run vidoreloop smoke

test: ## pytest (offline)
	uv run pytest -q

lint: ## ruff + mypy
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

.PHONY: help setup datasets download stats platform-up platform-status smoke test lint fmt
