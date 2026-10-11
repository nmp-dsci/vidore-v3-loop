"""Log to the central MLflow (nmp-central-ai, http://localhost:5000).

MLflow is the index, never the record: a scored run's record will be its folder
under `runs/`. Never falls back to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

import functools
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


@functools.cache
def _experiment_id() -> str | None:
    try:
        preflight()
        mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(EXPERIMENT)
    except Exception:  # noqa: BLE001 - a link is a convenience; the platform may be down
        return None
    return str(exp.experiment_id) if exp else None


def run_url(mlflow_run_id: str | None) -> str | None:
    """The MLflow UI's page for a run, or None when the platform cannot be reached."""
    eid = _experiment_id() if mlflow_run_id else None
    if not eid:
        return None
    return f"{settings().mlflow_tracking_uri.rstrip('/')}/#/experiments/{eid}/runs/{mlflow_run_id}"


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


def log_answer_run(doc: dict[str, Any], folder: Path) -> str:
    """One `kind=answer` run: models and settings as params, raw accuracy per setting and the
    run's tokens as metrics, run.json as the artifact. Audited accuracy changes as people
    audit, so it lives in the viewer, not here."""
    preflight()
    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    cfg = doc["config"]
    with mlflow.start_run(run_name=doc["run_id"]) as run:
        tags = required_tags(billing=doc["billing"])
        tags["git_sha"] = doc["code_sha"]
        mlflow.set_tags(
            {
                **tags,
                "kind": "answer",
                "dataset": doc["dataset"],
                "split": doc["split"],
                "final": str(doc["final"]).lower(),
                "fingerprint": doc["fingerprint"],
                "run_dir": f"runs/{doc['run_id']}",
            }
        )
        mlflow.log_params(
            {
                "modes": ",".join(cfg["modes"]),
                "k": cfg["k"],
                "modality": cfg["modality"],
                "source_run": cfg["source_run"],
                "source_stage": cfg["source_stage"],
                "answer_model": doc["models"]["answer"],
                "answer_effort": doc["models"]["effort"],
                "judge_model": doc["models"]["judge"],
                "judge_effort": doc["models"]["judge_effort"],
                "answer_prompt": doc["prompts"]["answer"],
                "judge_prompt": doc["prompts"]["judge"],
                "n_questions": doc["n"],
                "split_fingerprint": doc["split_fingerprint"],
            }
        )
        metrics: dict[str, float] = {}
        for mode, row in (doc["summary"] or {}).items():
            if mode == "retrieval_cost":
                if row.get("raw") is not None:
                    metrics["retrieval_cost.raw"] = row["raw"]
                continue
            if row["raw"]["mean"] is not None:
                metrics[f"{mode}.accuracy"] = row["raw"]["mean"]
            metrics[f"{mode}.errors"] = row["errors"]
        t = doc.get("totals") or {}
        for k in ("calls", "reused", "input_tokens", "output_tokens", "cost_usd"):
            if k in t:
                metrics[f"total.{k}"] = t[k]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(folder / "run.json"))
        run_id: str = run.info.run_id
    return run_id
