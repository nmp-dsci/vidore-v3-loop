"""Scored retrieval runs (M3): a pipeline configuration over one split, kept as a folder.

A run is the record; MLflow is its index. Each run is a folder under `runs/`:

- `run.json`: the configuration, models, split hash, code version, status and
  progress, then the summary (every metric per stage, each with its denominator),
  the slices (by query type, content type, format and author) and the costs.
- `queries.jsonl`: one line per question: its metadata, its gold pages, and each
  stage's top 100 with the full metric set, so any query can be drilled into later.

Every number comes from `eval.metrics.report`, the function the RAG pipeline tab
uses, and every ranking from `retrieval.pipeline.compute`, so a question scores the
same in a run as in the tab.

The test split is reported, never optimised on: a test run needs `final=True`, and
a configuration already scored on test is refused unless `force=True`.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidore_loop.config import ROOT, RUNS_DIR
from vidore_loop.data import catalog
from vidore_loop.eval import metrics, split
from vidore_loop.retrieval import fuse, pipeline

# the metrics a run averages; the rest of `report` are counts or ranks, handled below
MEANS = (
    "ndcg_10",
    "map_10",
    "mrr",
    "precision_5",
    "precision_10",
    "complete_10",
    "complete_50",
    *(f"recall_{k}" for k in metrics.RECALL_KS),
)
FULL_ONLY = ("hit_full_5", "hit_full_10")  # only questions with a grade-2 page count
RANKS = ("first_rank", "first_full_rank")
SLICE_METRICS = ("ndcg_10", "recall_10", "recall_50", "hit_full_10", "complete_50", "mrr")
SLICES: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "query type": lambda q: list(q["query_types"]),
    "content": lambda q: [c for c in (q.get("content_type") or []) if not c.startswith("N/A")],
    "format": lambda q: [q["query_format"]],
    "author": lambda q: ["person" if q["query_generator"] == "human" else "synthetic"],
}
_ACTIVE = threading.Lock()  # one run at a time in a process


class TestSplitGuardError(RuntimeError):
    """A run on the test split without `final`, or a repeat of a configuration already on test."""


class RunBusyError(RuntimeError):
    """Another run is already executing in this process."""


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    split: str
    stages: tuple[str, ...] = pipeline.STAGES
    rerank_depth: int = 20
    weights: tuple[float, float] = fuse.WEIGHTS  # (text, visual)
    note: str = ""
    limit: int | None = None  # the first N questions of the split: a smoke run, never final

    def label(self) -> str:
        w = "-".join(f"{x:g}" for x in self.weights)
        rr = f"d{self.rerank_depth}" if "reranked" in self.stages else "norr"
        return f"w{w}_{rr}" + (f"_n{self.limit}" if self.limit else "")

    def models(self) -> dict[str, Any]:
        from vidore_loop.retrieval import rerank, visual

        mk = visual.active(self.dataset)
        return {
            "text": "bm25s (OCR markdown, document name prepended)",
            "visual": visual.MODELS[mk].model_id if mk else None,
            "visual_dim": visual.MODELS[mk].dim if mk else None,
            "reranker": rerank.MODEL_ID if "reranked" in self.stages else None,
            "reranker_max_tokens": rerank.MAX_TOKENS if "reranked" in self.stages else None,
        }

    def fingerprint(self) -> str:
        """Everything that changes a ranking; two runs with one fingerprint should agree."""
        doc = {
            "dataset": self.dataset,
            "stages": list(self.stages),
            "rerank_depth": self.rerank_depth if "reranked" in self.stages else None,
            "weights": list(self.weights),
            "channel_k": pipeline.CHANNEL_K,
            "fused_k": pipeline.FUSED_K,
            "rrf_k": fuse.K,
            "models": self.models(),
            "split": split.fingerprint(self.dataset),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Progress:
    done: int = 0
    n: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def runs_dir() -> Path:
    return Path(os.environ.get("VIDORE_RUNS_DIR", str(RUNS_DIR)))


def code_version() -> str:
    """The commit, with `-dirty` when the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "frontend/src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    return (sha or "unknown") + ("-dirty" if dirty else "")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)


# ---- reading ------------------------------------------------------------------


def list_runs() -> list[dict[str, Any]]:
    """Every run's run.json, newest first, with a stale 'running' shown as interrupted."""
    d = runs_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*/run.json"), reverse=True):
        out.append(_live_status(json.loads(p.read_text())))
    return out


def get(run_id: str) -> dict[str, Any]:
    p = runs_dir() / run_id / "run.json"
    if not p.exists() or "/" in run_id or ".." in run_id:
        raise KeyError(run_id)
    return _live_status(json.loads(p.read_text()))


def queries(run_id: str) -> list[dict[str, Any]]:
    get(run_id)
    p = runs_dir() / run_id / "queries.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _live_status(doc: dict[str, Any]) -> dict[str, Any]:
    if doc.get("status") == "running" and not _alive(doc.get("pid")):
        doc["status"] = "interrupted"
    return doc


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def tested(fingerprint: str) -> list[str]:
    """Finished test runs of this configuration (the guard reads it)."""
    return [
        r["run_id"]
        for r in list_runs()
        if r.get("split") == "test"
        and r.get("fingerprint") == fingerprint
        and r.get("status") == "finished"
    ]


# ---- scoring ------------------------------------------------------------------


def summarise(records: Iterable[dict[str, Any]], stages: Iterable[str]) -> dict[str, Any]:
    """Every metric per stage, each with the number of questions behind it."""
    recs = list(records)
    out: dict[str, Any] = {}
    for s in stages:
        ms = [r["stages"][s]["metrics"] for r in recs if s in r["stages"]]
        ms = [m for m in ms if m["n_relevant"] > 0]
        full = [m for m in ms if m["n_full"] > 0]
        st: dict[str, Any] = {}
        for k in MEANS:
            st[k] = {"mean": _mean(float(m[k]) for m in ms), "n": len(ms)}
        for k in FULL_ONLY:
            st[k] = {"mean": _mean(float(m[k]) for m in full), "n": len(full)}
        for k, pool in (("first_rank", ms), ("first_full_rank", full)):
            found = [m[k] for m in pool if m[k] is not None]
            st[k] = {
                "median": statistics.median(found) if found else None,
                "found": len(found),
                "n": len(pool),
            }
        out[s] = st
    return out


def slices(records: Iterable[dict[str, Any]], stages: Iterable[str]) -> dict[str, Any]:
    """The headline metrics per value of each slice; a question with two types counts in both."""
    recs, stages = list(records), list(stages)
    out: dict[str, Any] = {}
    for dim, values_of in SLICES.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            for v in values_of(r):
                groups.setdefault(v, []).append(r)
        rows = []
        for v, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            summ = summarise(rs, stages)
            rows.append(
                {
                    "value": v,
                    "n": len(rs),
                    "stages": {
                        s: {k: summ[s][k] for k in SLICE_METRICS if k in summ[s]} for s in stages
                    },
                }
            )
        out[dim] = rows
    return out


def _mean(xs: Iterable[float]) -> float | None:
    v = list(xs)
    return round(sum(v) / len(v), 4) if v else None


def score_question(cfg: RunConfig, qid: int, st: dict[str, Any]) -> dict[str, Any]:
    """One question through every configured stage: rankings, metrics, timings."""
    q = catalog.question(cfg.dataset, qid)
    if q is None:
        raise KeyError(f"no English query {qid} in {cfg.dataset}")
    gold = {int(p["corpus_id"]): int(p["score"]) for p in q["pages"]}
    lists, out = pipeline.compute(
        cfg.dataset, q["query"], cfg.stages, cfg.rerank_depth, cfg.weights, st
    )
    stages: dict[str, Any] = {}
    for s in cfg.stages:
        if s not in lists:
            raise RuntimeError(f"stage {s} unavailable: {out.get(s, {}).get('unavailable')}")
        ranking = [cid for cid, _ in lists[s]][: pipeline.CHANNEL_K]
        rec: dict[str, Any] = {
            "ranking": ranking,
            "metrics": metrics.report(ranking, gold),
            "ms": out[s].get("ms"),
        }
        for k in ("scored", "cached", "seconds", "depth"):
            if k in out[s]:
                rec[k] = out[s][k]
        stages[s] = rec
    return {
        "query_id": qid,
        "query": q["query"],
        "query_types": q["query_types"],
        "content_type": q.get("content_type") or [],
        "query_format": q["query_format"],
        "query_generator": q["query_generator"],
        "gold": {str(c): g for c, g in gold.items()},
        "stages": stages,
    }


def _costs(records: list[dict[str, Any]], stages: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in stages:
        ms = [r["stages"][s]["ms"] for r in records if r["stages"][s].get("ms") is not None]
        c: dict[str, Any] = {"mean_ms": round(sum(ms) / len(ms)) if ms else None}
        if s == "reranked":
            c["pages_scored"] = sum(r["stages"][s].get("scored", 0) for r in records)
            c["pages_cached"] = sum(r["stages"][s].get("cached", 0) for r in records)
            c["rerank_seconds"] = round(sum(r["stages"][s].get("seconds", 0) for r in records), 1)
        out[s] = c
    return out


def plan(cfg: RunConfig, final: bool = False, force: bool = False) -> tuple[str, list[int], str]:
    """Check the guards and return (run_id, question ids, fingerprint) without running."""
    if cfg.split not in split.SPLITS:
        raise ValueError(f"a run scores train or test, not {cfg.split!r}")
    if cfg.split == "test":
        if not final:
            raise TestSplitGuardError(
                "the test split is reported once, never tuned on: pass final=True (--final) "
                "for the configuration chosen on train"
            )
        if cfg.limit:
            raise TestSplitGuardError("a final test run scores the whole split: drop the limit")
    fp = cfg.fingerprint()
    if cfg.split == "test" and not force and (done := tested(fp)):
        raise TestSplitGuardError(
            f"this configuration is already scored on test ({done[0]}); --force to repeat it"
        )
    ids = split.ids(cfg.dataset, cfg.split)
    if cfg.limit:
        ids = ids[: cfg.limit]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{cfg.dataset}_{cfg.label()}_{cfg.split}", ids, fp


def execute(
    cfg: RunConfig,
    final: bool = False,
    force: bool = False,
    log: bool = True,
    on_progress: Callable[[Progress], None] | None = None,
    run_id: str | None = None,
) -> str:
    """Score a configuration on a split; writes the run folder and logs it to MLflow."""
    if not _ACTIVE.acquire(blocking=False):
        raise RunBusyError("a run is already executing in this process")
    try:
        return _execute(cfg, final, force, log, on_progress, run_id)
    finally:
        _ACTIVE.release()


def _execute(
    cfg: RunConfig,
    final: bool,
    force: bool,
    log: bool,
    on_progress: Callable[[Progress], None] | None,
    run_id: str | None,
) -> str:
    from vidore_loop.tracking import mlflow_log

    rid, ids, fp = plan(cfg, final, force)
    rid = run_id or rid
    if log:
        mlflow_log.preflight()
    folder = runs_dir() / rid
    folder.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "run_id": rid,
        "kind": "retrieval",
        "dataset": cfg.dataset,
        "split": cfg.split,
        "final": final,
        "config": {**asdict(cfg), "stages": list(cfg.stages), "weights": list(cfg.weights)},
        "label": cfg.label(),
        "fingerprint": fp,
        "models": cfg.models(),
        "channel_k": pipeline.CHANNEL_K,
        "fused_k": pipeline.FUSED_K,
        "rrf_k": fuse.K,
        "split_seed": split.load(cfg.dataset)["seed"],
        "split_fingerprint": split.fingerprint(cfg.dataset),
        "code_sha": code_version(),
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "pid": os.getpid(),
        "progress": {"done": 0, "n": len(ids)},
        "n": len(ids),
        "summary": None,
        "slices": None,
        "costs": None,
        "mlflow_run_id": None,
    }
    _write(folder / "run.json", doc)
    st = pipeline.status(cfg.dataset)
    prog = Progress(n=len(ids))
    records: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        with (folder / "queries.jsonl").open("w") as f:
            for qid in ids:
                rec = score_question(cfg, qid, st)
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                prog.done += 1
                prog.seconds = round(time.time() - t0, 1)
                if prog.done % 5 == 0 or prog.done == prog.n:
                    doc["progress"] = {"done": prog.done, "n": prog.n, "seconds": prog.seconds}
                    _write(folder / "run.json", doc)
                if on_progress:
                    on_progress(prog)
    except BaseException as e:
        doc.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        _write(folder / "run.json", doc)
        raise
    doc.update(
        status="finished",
        finished_at=_now(),
        seconds=round(time.time() - t0, 1),
        summary=summarise(records, cfg.stages),
        slices=slices(records, cfg.stages),
        costs=_costs(records, cfg.stages),
    )
    _write(folder / "run.json", doc)
    if log:
        try:
            doc["mlflow_run_id"] = mlflow_log.log_retrieval_run(doc, folder)
        except Exception as e:  # the run is the record; a logging failure must not lose it
            doc["mlflow_error"] = f"{type(e).__name__}: {e}"
        _write(folder / "run.json", doc)
    return rid


def start(cfg: RunConfig, final: bool = False, force: bool = False) -> str:
    """Execute in a background thread (the viewer's process, sharing its loaded models)."""
    rid, _, _ = plan(cfg, final, force)  # guard errors surface to the caller now
    if not _ACTIVE.acquire(blocking=False):
        raise RunBusyError("a run is already executing in the viewer")

    def go() -> None:
        try:
            _execute(cfg, final, force, True, None, rid)
        except BaseException:  # noqa: S110 - execute records the failure in run.json
            pass
        finally:
            _ACTIVE.release()

    threading.Thread(target=go, name=f"run-{rid}", daemon=True).start()
    return rid


def busy() -> bool:
    return _ACTIVE.locked()
