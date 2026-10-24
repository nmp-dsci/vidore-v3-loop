"""M5 S1: an earlier answer run's answers judged again by another model, never re-answered.

A judge check scores a judge, not the system, so it lives under `runs/judge_checks/`
(`run.json` + `verdicts.jsonl`), out of the Runs tab. Each new verdict sits beside the
source run's, and the summary says, per setting and overall:

- how often the two judges agree on the paper's binary label (only Correct counts) and on
  all three levels, with Cohen's kappa on the binary label;
- how far accuracy moves (a lenient judge raises it, a strict one lowers it);
- where a person labelled the answer on the Review tab, how often each judge matches them.

Judging the source judge's own model again gives the noise floor a cheaper judge is held to.
A check of a test run needs `final`, like any test run.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidore_loop.answer import claude, judge
from vidore_loop.data import audits, catalog
from vidore_loop.eval import answers, runs

KIND = "judge_check"


@dataclass(frozen=True)
class JudgeCheckConfig:
    source_run: str  # a finished answer run
    judge_model: str = "haiku"
    judge_effort: str = "high"
    workers: int = 4
    limit: int | None = None  # the first N questions (a smoke)

    def label(self) -> str:
        return f"judge_{self.judge_model}-{self.judge_effort}" + (
            f"_n{self.limit}" if self.limit else ""
        )


def checks_dir() -> Path:
    return runs.runs_dir() / "judge_checks"


def list_checks() -> list[dict[str, Any]]:
    d = checks_dir()
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*/run.json"), reverse=True)]


def get(check_id: str) -> dict[str, Any]:
    p = checks_dir() / check_id / "run.json"
    if not p.exists():
        raise KeyError(check_id)
    doc: dict[str, Any] = json.loads(p.read_text())
    return doc


def verdicts(check_id: str) -> list[dict[str, Any]]:
    p = checks_dir() / check_id / "verdicts.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ---- scoring --------------------------------------------------------------------


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The check judge against the source judge over judged rows."""
    rows = [r for r in rows if r.get("correct") is not None and r.get("source_correct") is not None]
    src = [bool(r["source_correct"]) for r in rows]
    new = [bool(r["correct"]) for r in rows]
    n = len(rows)
    confusion: dict[str, dict[str, int]] = {
        s: {v: 0 for v in judge.VERDICTS} for s in judge.VERDICTS
    }
    for r in rows:
        if r["source_verdict"] in confusion and r["verdict"] in judge.VERDICTS:
            confusion[r["source_verdict"]][r["verdict"]] += 1
    return {
        "n": n,
        "binary_agree": (sum(x == y for x, y in zip(src, new, strict=True)) / n) if n else None,
        "three_level_agree": (
            sum(r["verdict"] == r["source_verdict"] for r in rows) / n if n else None
        ),
        "kappa": kappa(src, new),
        "source_accuracy": (sum(src) / n) if n else None,
        "check_accuracy": (sum(new) / n) if n else None,
        "shift": ((sum(new) - sum(src)) / n) if n else None,
        "lenient": sum(1 for x, y in zip(src, new, strict=True) if y and not x),
        "strict": sum(1 for x, y in zip(src, new, strict=True) if x and not y),
        "confusion": confusion,
    }


def against_people(
    rows: list[dict[str, Any]], labels: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    """Each judge's match rate with a person's label, over the labelled answers."""
    both = [
        (r, labels[(int(r["query_id"]), r["mode"])])
        for r in rows
        if (int(r["query_id"]), r["mode"]) in labels and r.get("correct") is not None
    ]
    n = len(both)
    if not n:
        return {"n": 0, "source": None, "check": None}
    return {
        "n": n,
        "source": sum(bool(r["source_correct"]) == bool(lab["correct"]) for r, lab in both) / n,
        "check": sum(bool(r["correct"]) == bool(lab["correct"]) for r, lab in both) / n,
    }


def summarise(rows: list[dict[str, Any]], source_run: str) -> dict[str, Any]:
    modes = sorted({r["mode"] for r in rows})
    labels = audits.judge_labels(source_run)
    return {
        "all": agreement(rows),
        "modes": {m: agreement([r for r in rows if r["mode"] == m]) for m in modes},
        "people": against_people(rows, labels),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    t = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "ms": 0}
    for r in rows:
        u = r.get("usage") or {}
        if not u:
            continue
        t["calls"] += 1
        t["input_tokens"] += int(u.get("input_tokens", 0))
        t["output_tokens"] += int(u.get("output_tokens", 0))
        t["cost_usd"] += float(u.get("cost_usd") or 0)
        t["ms"] += int(u.get("ms", 0))
    t["cost_usd"] = round(float(t["cost_usd"]), 2)
    return t


# ---- running --------------------------------------------------------------------


def _items(source: dict[str, Any], limit: int | None) -> list[tuple[dict[str, Any], str]]:
    recs = answers.records(source["run_id"])
    if limit:
        recs = recs[:limit]
    return [
        (r, mode)
        for r in recs
        for mode in source["config"]["modes"]
        if r["modes"].get(mode, {}).get("answer") is not None
        and r["modes"][mode].get("correct") is not None
    ]


def check_one(
    dataset: str, rec: dict[str, Any], mode: str, model: str, effort: str
) -> dict[str, Any]:
    q = catalog.question(dataset, int(rec["query_id"]))
    if q is None:
        raise KeyError(rec["query_id"])
    m = rec["modes"][mode]
    out: dict[str, Any] = {
        "query_id": int(rec["query_id"]),
        "mode": mode,
        "source_verdict": m["verdict"],
        "source_correct": bool(m["correct"]),
    }
    try:
        j = judge.judge(q["query"], q["answer"], m["answer"], model, effort)
        out.update(
            verdict=j["verdict"],
            correct=j["correct"],
            explanation=j["explanation"],
            usage=j["usage"],
        )
    except claude.ClaudeError as e:
        out.update(verdict=None, correct=None, error=str(e)[:500])
    return out


def execute(
    cfg: JudgeCheckConfig,
    final: bool = False,
    log: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    from vidore_loop.tracking import mlflow_log

    claude.check_billing()
    source = runs.get(cfg.source_run)
    if source.get("kind") != answers.KIND or source.get("status") != "finished":
        raise ValueError(f"{cfg.source_run} is not a finished answer run")
    if source["split"] == "test" and not final:
        raise runs.TestSplitGuardError(
            "re-judging a test run is reporting, done once: pass final=True (--final)"
        )
    if log:
        mlflow_log.preflight()
    items = _items(source, cfg.limit)
    model_id = claude.resolve(cfg.judge_model)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cid = f"{stamp}_{source['dataset']}_{cfg.label()}_{source['split']}"
    folder = checks_dir() / cid
    folder.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "run_id": cid,
        "kind": KIND,
        "dataset": source["dataset"],
        "split": source["split"],
        "final": final,
        "config": asdict(cfg),
        "label": cfg.label(),
        "source": {
            "run_id": source["run_id"],
            "judge": source["models"]["judge"],
            "judge_effort": source["models"]["judge_effort"],
            "answer": source["models"]["answer"],
        },
        "models": {"judge": model_id, "judge_effort": cfg.judge_effort},
        "prompts": {"judge": judge.prompt_version()},
        "billing": claude.billing(),
        "code_sha": runs.code_version(),
        "started_at": runs._now(),
        "n": len(items),
        "status": "running",
        "summary": None,
    }
    runs._write(folder / "run.json", doc)
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()
    try:
        with (
            (folder / "verdicts.jsonl").open("w") as f,
            ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool,
        ):
            futs = [
                pool.submit(check_one, source["dataset"], r, m, model_id, cfg.judge_effort)
                for r, m in items
            ]
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                if on_progress:
                    on_progress(len(rows), len(items))
    except BaseException as e:
        doc.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=runs._now())
        runs._write(folder / "run.json", doc)
        raise
    rows.sort(key=lambda r: (r["query_id"], r["mode"]))
    doc.update(
        status="finished",
        finished_at=runs._now(),
        seconds=round(time.time() - t0, 1),
        summary=summarise(rows, source["run_id"]),
        totals=totals(rows),
    )
    runs._write(folder / "run.json", doc)
    if log:
        try:
            doc["mlflow_run_id"] = mlflow_log.log_judge_check(doc, folder)
        except Exception as e:  # the folder is the record; a logging failure must not lose it
            doc["mlflow_error"] = f"{type(e).__name__}: {e}"
        runs._write(folder / "run.json", doc)
    return cid


def rescore(check_id: str) -> dict[str, Any]:
    """A check's summary recomputed with today's labels (people label after a check runs)."""
    doc = get(check_id)
    return summarise(verdicts(check_id), doc["source"]["run_id"])
