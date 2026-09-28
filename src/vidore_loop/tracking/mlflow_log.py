"""Log to the central MLflow (nmp-central-ai, http://localhost:5000).

MLflow is the index, never the record: a scored run's record will be its folder
under `runs/`. Never falls back to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

import urllib.error
import urllib.request

import mlflow

from vidore_loop.config import settings

EXPERIMENT = "vidore-v3-loop/evals"
PROJECT = "vidore-v3-loop"


class TrackingDownError(RuntimeError):
    """The central MLflow is not reachable; the remedy is `make platform-up`."""


def preflight() -> str:
    uri = settings().mlflow_tracking_uri.rstrip("/")
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=3) as r:  # noqa: S310 - local platform URL
            if r.status != 200:
                raise TrackingDownError(f"{uri}/health returned {r.status}")
    except (urllib.error.URLError, OSError) as e:
        raise TrackingDownError(
            f"MLflow at {uri} is not reachable ({e}). Start the platform: `make platform-up`"
        ) from e
    return uri


def required_tags(billing: str | None = None) -> dict[str, str]:
    """The four tags PLATFORM.md requires on every run and trace."""
    s = settings()
    return {
        "project": PROJECT,
        "git_sha": s.code_sha,
        "env": "local",
        "billing": billing or s.billing,
    }


def log_smoke(ndcg: float, recall: float) -> str:
    """One `kind=smoke` run with no model call behind it; returns the MLflow run id."""
    preflight()
    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="smoke") as run:
        mlflow.set_tags({**required_tags(billing="none"), "kind": "smoke"})
        mlflow.log_params({"k": 10, "fixture": "cli.SMOKE_RANKING"})
        mlflow.log_metrics({"ndcg_at_10": ndcg, "recall_at_10": recall})
        run_id: str = run.info.run_id
    return run_id
