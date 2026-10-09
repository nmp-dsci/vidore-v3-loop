"""Log to the central MLflow (nmp-central-ai, http://localhost:5000).

MLflow is the index, never the record: a scored run's record will be its folder
under `runs/`. Never falls back to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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


HEADLINE = ("ndcg_10", "recall_10", "recall_50", "hit_full_10", "complete_50", "mrr", "map_10")


def log_retrieval_run(doc: dict[str, Any], folder: Path) -> str:
    """One `kind=retrieval` run: the configuration as params, the headline metrics per
    stage, and run.json as the artifact. The folder under runs/ stays the record."""
    preflight()
    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    cfg = doc["config"]
    with mlflow.start_run(run_name=doc["run_id"]) as run:
        tags = required_tags(billing="none")
        tags["git_sha"] = doc["code_sha"]
        mlflow.set_tags(
            {
                **tags,
                "kind": "retrieval",
                "dataset": doc["dataset"],
                "split": doc["split"],
                "final": str(doc["final"]).lower(),
                "fingerprint": doc["fingerprint"],
                "run_dir": f"runs/{doc['run_id']}",
            }
        )
        mlflow.log_params(
            {
                "stages": ",".join(cfg["stages"]),
                "rerank_depth": cfg["rerank_depth"],
                "w_text": cfg["weights"][0],
                "w_visual": cfg["weights"][1],
                "n_questions": doc["n"],
                "channel_k": doc["channel_k"],
                "fused_k": doc["fused_k"],
                "rrf_k": doc["rrf_k"],
                "visual_model": doc["models"]["visual"],
                "reranker": doc["models"]["reranker"],
                "split_seed": doc["split_seed"],
                "split_fingerprint": doc["split_fingerprint"],
            }
        )
        for stage, st in (doc["summary"] or {}).items():
            mlflow.log_metrics(
                {f"{stage}.{k}": st[k]["mean"] for k in HEADLINE if st[k]["mean"] is not None}
            )
        mlflow.log_artifact(str(folder / "run.json"))
        run_id: str = run.info.run_id
    return run_id
