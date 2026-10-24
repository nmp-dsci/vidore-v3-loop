"""M5: an agent run. The agent answers every question of a split; both halves are scored.

A run is a folder under `runs/` beside the others (`kind: "agent"`): `run.json` and
`agent.jsonl`, one record per question shaped so the existing scorers read it:

- `stages.agent`: the agent's final 10 pages and the full retrieval metric set
  (`metrics.report`), so NDCG@10 sits beside the pipeline board;
- `modes.agent`: the answer, the judge's verdict and the tokens, so accuracy sits beside the
  paper's answer table and M4's runs;
- `steps`, `cited`, `searches`, `views`: the path the agent took, for the Agent tab.

Same guards as every run: test needs `final`, once per configuration; the model aliases are
resolved once and the ids go into the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from vidore_loop.answer import agent, claude, judge
from vidore_loop.data import catalog
from vidore_loop.eval import answers, metrics, runs, split
from vidore_loop.retrieval import fuse

KIND = "agent"
STAGE = MODE = "agent"


@dataclass(frozen=True)
class AgentConfig:
    dataset: str
    split: str
    model: str = "sonnet"
    effort: str = "medium"
    judge_model: str = "sonnet"
    judge_effort: str = "medium"
    max_searches: int = agent.MAX_SEARCHES
    max_views: int = agent.MAX_VIEWS
    workers: int = 4
    note: str = ""
    limit: int | None = None

    def label(self) -> str:
        return f"agent_{self.model}-{self.effort}" + (f"_n{self.limit}" if self.limit else "")

    def fingerprint(self) -> str:
        doc = {
            "dataset": self.dataset,
            "model": [claude.resolve(self.model), self.effort],
            "judge": [claude.resolve(self.judge_model), self.judge_effort],
            "caps": [self.max_searches, self.max_views],
            "prompts": [agent.prompt_version(), judge.prompt_version()],
            "search": {"stage": "fused", "weights": list(fuse.WEIGHTS)},
            "split": split.fingerprint(self.dataset),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12]


def plan(cfg: AgentConfig, final: bool = False, force: bool = False) -> tuple[str, list[int], str]:
    if cfg.split not in split.SPLITS:
        raise ValueError(f"an agent run scores train or test, not {cfg.split!r}")
    if cfg.split == "test":
        if not final:
            raise runs.TestSplitGuardError(
                "the test split is reported once, never tuned on: pass final=True (--final)"
            )
        if cfg.limit:
            raise runs.TestSplitGuardError(
                "a final test run answers the whole split: drop the limit"
            )
    fp = cfg.fingerprint()
    if cfg.split == "test" and not force and (done := runs.tested(fp)):
        raise runs.TestSplitGuardError(
            f"this configuration is already scored on test ({done[0]}); --force to repeat it"
        )
    ids = split.ids(cfg.dataset, cfg.split)
    if cfg.limit:
        ids = ids[: cfg.limit]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{cfg.dataset}_{cfg.label()}_{cfg.split}", ids, fp


# ---- one question ---------------------------------------------------------------


def run_question(
    cfg: AgentConfig, qid: int, search_fn: agent.SearchFn = agent.champion_search
) -> dict[str, Any]:
    q = catalog.question(cfg.dataset, qid)
    if q is None:
        raise KeyError(qid)
    gold = {int(p["corpus_id"]): int(p["score"]) for p in answers.gold_pages(cfg.dataset, qid)}
    rec: dict[str, Any] = {
        "query_id": qid,
        "query": q["query"],
        "query_types": q["query_types"],
        "content_type": q.get("content_type") or [],
        "query_format": q["query_format"],
        "query_generator": q["query_generator"],
        "gold": {str(c): g for c, g in gold.items()},
        "n_full": sum(1 for g in gold.values() if g == 2),
    }
    t0 = time.time()
    try:
        a = agent.answer(
            cfg.dataset,
            q["query"],
            cfg.model,
            cfg.effort,
            cfg.max_searches,
            cfg.max_views,
            search_fn,
        )
    except claude.ClaudeError as e:
        rec.update(stages={}, modes={MODE: {"answer": None, "verdict": None, "correct": None,
                                            "error": str(e)[:500]}})  # fmt: skip
        return rec
    ms = int((time.time() - t0) * 1000)
    mode: dict[str, Any] = {"answer": a["answer"], "pages": a["ranking"]}
    try:
        j = judge.judge(q["query"], q["answer"], a["answer"], cfg.judge_model, cfg.judge_effort)
        mode.update(verdict=j["verdict"], correct=j["correct"], explanation=j["explanation"])
        judge_usage = j["usage"]
    except claude.ClaudeError as e:
        mode.update(verdict=None, correct=None, error=str(e)[:500])
        judge_usage = None
    mode["usage"] = {"answer": a["usage"], **({"judge": judge_usage} if judge_usage else {})}
    cited = a["cited"]
    rec.update(
        stages={STAGE: {"ranking": a["ranking"], "metrics": metrics.report(a["ranking"], gold), "ms": ms}},
        modes={MODE: mode},
        agent_ranking=a["agent_ranking"],
        cited=cited,
        cited_gold=bool(set(cited) & set(gold)),
        cited_full=bool(set(cited) & {c for c, g in gold.items() if g == 2}),
        steps=a["steps"],
        searches=a["searches"],
        views=a["views"],
    )  # fmt: skip
    return rec


# ---- scoring --------------------------------------------------------------------


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(records)
    done = [r for r in recs if STAGE in r.get("stages", {})]
    return {
        "retrieval": runs.summarise(done, [STAGE]).get(STAGE),
        "answers": answers.summarise(recs, [MODE]).get(MODE),
        "cited_gold": _rate([r["cited_gold"] for r in done]),
        "cited_full": _rate([r["cited_full"] for r in done if r["n_full"]]),
        "searches": _avg([r["searches"] for r in done]),
        "views": _avg([r["views"] for r in done]),
        "turns": _avg([r["modes"][MODE]["usage"]["answer"].get("turns", 0) for r in done]),
        "seconds": _avg([r["stages"][STAGE]["ms"] / 1000 for r in done]),
        "errors": sum(1 for r in recs if r["modes"][MODE].get("error")),
    }


def _rate(xs: list[bool]) -> dict[str, Any]:
    return {"mean": (sum(xs) / len(xs)) if xs else None, "n": len(xs)}


def _avg(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def records(run_id: str) -> list[dict[str, Any]]:
    runs.get(run_id)
    p = runs.runs_dir() / run_id / "agent.jsonl"
    if not p.exists():
        return []
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return sorted(recs, key=lambda r: int(r["query_id"]))


# ---- running --------------------------------------------------------------------


def execute(
    cfg: AgentConfig,
    final: bool = False,
    force: bool = False,
    log: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    search_fn: agent.SearchFn = agent.champion_search,
) -> str:
    if not runs._ACTIVE.acquire(blocking=False):
        raise runs.RunBusyError("a run is already executing in this process")
    try:
        return _execute(cfg, final, force, log, on_progress, search_fn)
    finally:
        runs._ACTIVE.release()


def _execute(
    cfg: AgentConfig,
    final: bool,
    force: bool,
    log: bool,
    on_progress: Callable[[int, int], None] | None,
    search_fn: agent.SearchFn,
) -> str:
    from vidore_loop.tracking import mlflow_log

    claude.check_billing()
    if log:
        mlflow_log.preflight()
    rid, ids, fp = plan(cfg, final, force)
    folder = runs.runs_dir() / rid
    folder.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "run_id": rid,
        "kind": KIND,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "final": final,
        "config": asdict(cfg),
        "label": cfg.label(),
        "fingerprint": fp,
        "models": {
            "answer": claude.resolve(cfg.model),
            "effort": cfg.effort,
            "judge": claude.resolve(cfg.judge_model),
            "judge_effort": cfg.judge_effort,
        },
        "prompts": {"agent": agent.prompt_version(), "judge": judge.prompt_version()},
        "search": {"stage": "fused", "weights": list(fuse.WEIGHTS)},
        "billing": claude.billing(),
        "split_seed": split.load(cfg.dataset)["seed"],
        "split_fingerprint": split.fingerprint(cfg.dataset),
        "code_sha": runs.code_version(),
        "started_at": runs._now(),
        "n": len(ids),
        "status": "running",
        "pid": os.getpid(),
        "finished_at": None,
        "progress": {"done": 0, "n": len(ids)},
        "summary": None,
    }
    runs._write(folder / "run.json", doc)
    recs: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()
    try:
        with (
            (folder / "agent.jsonl").open("w") as f,
            ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool,
        ):
            futs = [pool.submit(run_question, cfg, q, search_fn) for q in ids]
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    recs.append(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    doc["progress"] = {
                        "done": len(recs),
                        "n": len(ids),
                        "seconds": round(time.time() - t0, 1),
                    }
                    runs._write(folder / "run.json", doc)
                if on_progress:
                    on_progress(len(recs), len(ids))
    except BaseException as e:
        doc.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=runs._now())
        runs._write(folder / "run.json", doc)
        raise
    recs.sort(key=lambda r: int(r["query_id"]))
    doc.update(
        status="finished",
        finished_at=runs._now(),
        seconds=round(time.time() - t0, 1),
        summary=summarise(recs),
        slices={
            "retrieval": runs.slices([r for r in recs if r.get("stages")], [STAGE]),
            "answers": answers.slices(recs, [MODE]),
        },
        totals=answers.totals(recs),
    )
    runs._write(folder / "run.json", doc)
    if log:
        try:
            doc["mlflow_run_id"] = mlflow_log.log_agent_run(doc, folder)
        except Exception as e:  # the run is the record; a logging failure must not lose it
            doc["mlflow_error"] = f"{type(e).__name__}: {e}"
        runs._write(folder / "run.json", doc)
    return rid
