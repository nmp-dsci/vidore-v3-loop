"""M4: Claude answers every question of a split two ways, and a judge scores each answer.

An answer run is a folder under `runs/` beside the retrieval runs (`kind: "answer"`):
`run.json` and `answers.jsonl`, one record per question with, for each setting, the
pages given, the answer, the judge's verdict and the tokens spent.

- `retrieved`: the top k pages of a finished retrieval run on the same split (by default
  the latest one with the viewer's fusion weights), so an answer run never re-retrieves
  and always names the ranking it answered from;
- `oracle`: every human-marked page, the ceiling.

The retrieval cost is oracle minus retrieved. Scores are reported raw (the judge alone)
and audited (disputed references left out, a person's label over the judge's), and the
audited figure is recomputed whenever the Review tab writes an audit.

Same guards as a retrieval run: test needs `final`, once per configuration. A run can be
resumed after an interruption; questions already answered are kept.

Answers are reused across runs (`reuse`, on by default): a setting's answer and verdict
are copied from any earlier run, archived ones included, that gave Claude the same pages
in the same order under the same models, effort, modality and prompts. A copied answer
names the run it came from (`cached_from`) and its tokens are not counted again.
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

from vidore_loop.answer import claude, generate, judge
from vidore_loop.data import audits, catalog
from vidore_loop.eval import runs, split
from vidore_loop.retrieval import fuse

KIND = "answer"


@dataclass(frozen=True)
class AnswerConfig:
    dataset: str
    split: str
    modes: tuple[str, ...] = generate.MODES
    k: int = generate.K
    source_run: str | None = None  # the retrieval run whose ranking `retrieved` answers from
    source_stage: str = "fused"
    modality: str = "image+text"
    model: str = "sonnet"
    effort: str = "medium"
    judge_model: str = "opus"
    judge_effort: str = "medium"
    workers: int = 4
    note: str = ""
    limit: int | None = None

    def label(self) -> str:
        m = "-".join(x[0] for x in self.modes)  # r-o
        return f"ans_{m}_k{self.k}_{self.model}" + (f"_n{self.limit}" if self.limit else "")

    def fingerprint(self, source_fp: str | None) -> str:
        doc = {
            "dataset": self.dataset,
            "modes": list(self.modes),
            "k": self.k,
            "source": [source_fp, self.source_stage] if "retrieved" in self.modes else None,
            "modality": self.modality,
            "model": [claude.resolve(self.model), self.effort],
            "judge": [claude.resolve(self.judge_model), self.judge_effort],
            "prompts": [generate.prompt_version(), judge.prompt_version()],
            "split": split.fingerprint(self.dataset),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12]


def default_source(key: str, which: str, stage: str = "fused") -> str | None:
    """The latest finished, full retrieval run on this split with the viewer's fusion weights."""
    for r in runs.list_runs():
        c = r.get("config", {})
        if (
            r.get("kind", "retrieval") == "retrieval"
            and r.get("dataset") == key
            and r.get("split") == which
            and split.current(key, r.get("split_fingerprint"))
            and r.get("status") == "finished"
            and not c.get("limit")
            and list(c.get("weights", [])) == list(fuse.WEIGHTS)
            and stage in c.get("stages", [])
        ):
            return str(r["run_id"])
    return None


def _source(cfg: AnswerConfig) -> tuple[str | None, str | None, dict[int, list[int]]]:
    """(run id, its fingerprint, query id → ranking) for the `retrieved` setting."""
    if "retrieved" not in cfg.modes:
        return None, None, {}
    rid = cfg.source_run or default_source(cfg.dataset, cfg.split, cfg.source_stage)
    if rid is None:
        raise ValueError(
            f"no finished retrieval run on {cfg.dataset} {cfg.split} to answer from: "
            f"`make eval DATASET={cfg.dataset} SPLIT={cfg.split}` first"
        )
    doc = runs.get(rid)
    if doc.get("split") != cfg.split or doc.get("dataset") != cfg.dataset:
        raise ValueError(f"{rid} is a {doc.get('dataset')} {doc.get('split')} run, not {cfg.split}")
    if cfg.source_stage not in doc["config"]["stages"]:
        raise ValueError(f"{rid} has no {cfg.source_stage} stage")
    rankings = {
        int(r["query_id"]): list(r["stages"][cfg.source_stage]["ranking"])
        for r in runs.queries(rid)
    }
    return rid, str(doc["fingerprint"]), rankings


def plan(cfg: AnswerConfig, final: bool = False, force: bool = False) -> tuple[str, list[int], str]:
    if cfg.split not in split.SPLITS:
        raise ValueError(f"an answer run scores train or test, not {cfg.split!r}")
    bad = [m for m in cfg.modes if m not in generate.MODES]
    if bad or not cfg.modes:
        raise ValueError(f"modes are some of {generate.MODES}, not {list(cfg.modes)}")
    if cfg.modality not in generate.MODALITIES:
        raise ValueError(f"modality is one of {generate.MODALITIES}")
    if cfg.split == "test":
        if not final:
            raise runs.TestSplitGuardError(
                "the test split is reported once, never tuned on: pass final=True (--final)"
            )
        if cfg.limit:
            raise runs.TestSplitGuardError(
                "a final test run answers the whole split: drop the limit"
            )
    _, source_fp, _ = _source(cfg)
    fp = cfg.fingerprint(source_fp)
    if cfg.split == "test" and not force and (done := runs.tested(fp)):
        raise runs.TestSplitGuardError(
            f"this configuration is already scored on test ({done[0]}); --force to repeat it"
        )
    ids = split.ids(cfg.dataset, cfg.split)
    if cfg.limit:
        ids = ids[: cfg.limit]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{cfg.dataset}_{cfg.label()}_{cfg.split}", ids, fp


# ---- reuse ----------------------------------------------------------------------

CacheKey = tuple[Any, ...]


def _setup_key(
    c: dict[str, Any], models: dict[str, Any], prompts: dict[str, Any]
) -> tuple[Any, ...]:
    return (
        c.get("modality", "image+text"),
        models.get("answer"),
        models.get("effort"),
        prompts.get("answer"),
        models.get("judge"),
        models.get("judge_effort"),
        prompts.get("judge"),
    )


def cache_key(cfg: AnswerConfig, qid: int, mode: str, given: list[int]) -> CacheKey:
    setup = _setup_key(
        {"modality": cfg.modality},
        {
            "answer": claude.resolve(cfg.model),
            "effort": cfg.effort,
            "judge": claude.resolve(cfg.judge_model),
            "judge_effort": cfg.judge_effort,
        },
        {"answer": generate.prompt_version(), "judge": judge.prompt_version()},
    )
    return (cfg.dataset, int(qid), mode, tuple(int(c) for c in given), *setup)


def answer_cache(key: str) -> dict[CacheKey, dict[str, Any]]:
    """Every judged answer already on disk for a dataset, by what Claude was given."""
    out: dict[CacheKey, dict[str, Any]] = {}
    for p in sorted(runs.runs_dir().glob("**/answers.jsonl")):
        meta = p.parent / "run.json"
        if not meta.exists():
            continue
        doc = json.loads(meta.read_text())
        if doc.get("kind") != KIND or doc.get("dataset") != key:
            continue
        setup = _setup_key(doc.get("config", {}), doc.get("models", {}), doc.get("prompts", {}))
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for mode, m in rec.get("modes", {}).items():
                if m.get("correct") is None or m.get("cached_from") or m.get("pages") is None:
                    continue
                k = (key, int(rec["query_id"]), mode, tuple(int(c) for c in m["pages"]), *setup)
                out.setdefault(k, {**m, "cached_from": doc["run_id"]})
    return out


# ---- one question ---------------------------------------------------------------


def gold_pages(key: str, qid: int) -> list[dict[str, Any]]:
    return [p for p in catalog.qrels(key).get(qid, []) if p["score"] >= 1]


def answer_question(
    cfg: AnswerConfig,
    qid: int,
    ranking: list[int] | None,
    cache: dict[CacheKey, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every setting's answer to one question, each judged against the reference."""
    q = catalog.question(cfg.dataset, qid)
    if q is None:
        raise KeyError(qid)
    gold = gold_pages(cfg.dataset, qid)
    full = {int(p["corpus_id"]) for p in gold if p["score"] == 2}
    grade = {int(p["corpus_id"]): int(p["score"]) for p in gold}
    rec: dict[str, Any] = {
        "query_id": qid,
        "query_types": q["query_types"],
        "content_type": q["content_type"],
        "query_format": q["query_format"],
        "query_generator": q["query_generator"],
        "n_full": len(full),
        "modes": {},
    }
    for mode in cfg.modes:
        if mode == "retrieved":
            given = (ranking or [])[: cfg.k]
        else:
            given = sorted(grade, key=lambda c: (-grade[c], c))
        hit = (cache or {}).get(cache_key(cfg, qid, mode, given))
        if hit is not None:
            rec["modes"][mode] = dict(hit)
            continue
        out: dict[str, Any] = {"pages": given}
        try:
            a = generate.answer(cfg.dataset, q["query"], given, cfg.model, cfg.effort, cfg.modality)
            j = judge.judge(q["query"], q["answer"], a["answer"], cfg.judge_model, cfg.judge_effort)
            out.update(
                answer=a["answer"],
                verdict=j["verdict"],
                correct=j["correct"],
                explanation=j["explanation"],
                usage={"answer": a["usage"], "judge": j["usage"]},
            )
        except claude.ClaudeError as e:
            out.update(answer=None, verdict=None, correct=None, error=str(e)[:500])
        out["grades"] = [grade.get(c, 0) for c in given]
        out["full_given"] = bool(full & set(given))
        rec["modes"][mode] = out
    return rec


# ---- scoring --------------------------------------------------------------------


def _audited(
    rec: dict[str, Any], mode: str, labels: dict[tuple[int, str], dict[str, Any]]
) -> bool | None:
    m = rec["modes"].get(mode, {})
    lab = labels.get((int(rec["query_id"]), mode))
    return bool(lab["correct"]) if lab else m.get("correct")


def summarise(
    records: Iterable[dict[str, Any]],
    modes: Iterable[str],
    disputed: set[int] | None = None,
    labels: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per setting: raw accuracy (the judge), audited accuracy, verdict counts, tokens."""
    recs = list(records)
    disputed = disputed or set()
    labels = labels or {}
    out: dict[str, Any] = {}
    for mode in modes:
        ms = [(r, r["modes"][mode]) for r in recs if mode in r["modes"]]
        judged = [(r, m) for r, m in ms if m.get("correct") is not None]
        kept = [(r, m) for r, m in judged if int(r["query_id"]) not in disputed]
        aud = [_audited(r, mode, labels) for r, _ in kept]
        counts = {v: sum(1 for _, m in judged if m.get("verdict") == v) for v in judge.VERDICTS}
        tok_in = [m["usage"]["answer"]["input_tokens"] for _, m in judged if m.get("usage")]
        row: dict[str, Any] = {
            "raw": _mean_n([bool(m["correct"]) for _, m in judged]),
            "audited": _mean_n([bool(x) for x in aud if x is not None]),
            "verdicts": counts,
            "errors": sum(1 for _, m in ms if m.get("error")),
            "answer_tokens": round(sum(tok_in) / len(tok_in)) if tok_in else None,
            "ms": _avg([m["usage"]["answer"]["ms"] for _, m in judged if m.get("usage")]),
        }
        if mode == "retrieved":
            given = [(r, m) for r, m in judged if r["n_full"]]
            row["full_given"] = _mean_n([bool(m.get("full_given")) for _, m in given])
            row["correct_when_full_given"] = _mean_n(
                [bool(m["correct"]) for _, m in given if m.get("full_given")]
            )
            row["correct_when_full_missed"] = _mean_n(
                [bool(m["correct"]) for _, m in given if not m.get("full_given")]
            )
        out[mode] = row
    if "oracle" in out and "retrieved" in out:
        for kind in ("raw", "audited"):
            o, r = out["oracle"][kind]["mean"], out["retrieved"][kind]["mean"]
            out.setdefault("retrieval_cost", {})[kind] = None if o is None or r is None else o - r
    return out


def slices(records: Iterable[dict[str, Any]], modes: Iterable[str]) -> dict[str, Any]:
    """Raw accuracy per setting, cut by the question's type, content, format and author."""
    recs = list(records)
    out: dict[str, Any] = {}
    for dim, fn in runs.SLICES.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            for v in fn(r):
                groups.setdefault(v, []).append(r)
        out[dim] = sorted(
            (
                {
                    "value": v,
                    "n": len(rs),
                    "modes": {
                        m: _mean_n(
                            [
                                bool(r["modes"][m]["correct"])
                                for r in rs
                                if r["modes"].get(m, {}).get("correct") is not None
                            ]
                        )
                        for m in modes
                    },
                }
                for v, rs in groups.items()
            ),
            key=lambda row: -row["n"],
        )
    return out


def _mean_n(xs: list[bool]) -> dict[str, Any]:
    return {"mean": (sum(xs) / len(xs)) if xs else None, "n": len(xs)}


def _avg(xs: list[int]) -> int | None:
    return round(sum(xs) / len(xs)) if xs else None


def totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Tokens and notional per-token cost across every call of the run (the subscription pays)."""
    t = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "reused": 0}
    for r in records:
        for m in r["modes"].values():
            if m.get("cached_from"):
                t["reused"] += 1
                continue
            for u in (m.get("usage") or {}).values():
                t["calls"] += 1
                t["input_tokens"] += int(u.get("input_tokens", 0))
                t["output_tokens"] += int(u.get("output_tokens", 0))
                t["cost_usd"] += float(u.get("cost_usd") or 0)
    t["cost_usd"] = round(float(t["cost_usd"]), 2)
    return t


def audited_view(doc: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's summary recomputed with today's audits."""
    return summarise(
        records,
        doc["config"]["modes"],
        audits.disputed(doc["dataset"]),
        audits.judge_labels(doc["run_id"]),
    )


# ---- reading --------------------------------------------------------------------


def records(run_id: str) -> list[dict[str, Any]]:
    runs.get(run_id)
    p = runs.runs_dir() / run_id / "answers.jsonl"
    if not p.exists():
        return []
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return sorted(recs, key=lambda r: int(r["query_id"]))


# ---- running --------------------------------------------------------------------


def execute(
    cfg: AnswerConfig,
    final: bool = False,
    force: bool = False,
    log: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    run_id: str | None = None,
    resume: str | None = None,
    reuse: bool = True,
) -> str:
    if not runs._ACTIVE.acquire(blocking=False):
        raise runs.RunBusyError("a run is already executing in this process")
    try:
        return _execute(cfg, final, force, log, on_progress, run_id, resume, reuse)
    finally:
        runs._ACTIVE.release()


def _execute(
    cfg: AnswerConfig,
    final: bool,
    force: bool,
    log: bool,
    on_progress: Callable[[int, int], None] | None,
    run_id: str | None,
    resume: str | None,
    reuse: bool = True,
) -> str:
    from vidore_loop.tracking import mlflow_log

    claude.check_billing()
    if log:
        mlflow_log.preflight()
    if resume:
        doc = runs.get(resume)
        cfg = AnswerConfig(**{**doc["config"], "modes": tuple(doc["config"]["modes"])})
        rid, ids, fp = (
            resume,
            split.ids(cfg.dataset, cfg.split)[: cfg.limit or None],
            doc["fingerprint"],
        )
        source_rid, _, rankings = _source(cfg)
        done = {int(r["query_id"]): r for r in records(resume) if _complete(r)}
    else:
        rid, ids, fp = plan(cfg, final, force)
        rid = run_id or rid
        source_rid, _, rankings = _source(cfg)
        done = {}
        doc = {
            "run_id": rid,
            "kind": KIND,
            "dataset": cfg.dataset,
            "split": cfg.split,
            "final": final,
            "config": {**asdict(cfg), "modes": list(cfg.modes), "source_run": source_rid},
            "label": cfg.label(),
            "fingerprint": fp,
            "models": {
                "answer": claude.resolve(cfg.model),
                "effort": cfg.effort,
                "judge": claude.resolve(cfg.judge_model),
                "judge_effort": cfg.judge_effort,
            },
            "prompts": {"answer": generate.prompt_version(), "judge": judge.prompt_version()},
            "billing": claude.billing(),
            "split_seed": split.load(cfg.dataset)["seed"],
            "split_fingerprint": split.fingerprint(cfg.dataset),
            "code_sha": runs.code_version(),
            "started_at": runs._now(),
            "n": len(ids),
        }
    cache = answer_cache(cfg.dataset) if reuse else {}
    folder = runs.runs_dir() / rid
    folder.mkdir(parents=True, exist_ok=True)
    doc.update(
        status="running",
        pid=os.getpid(),
        finished_at=None,
        progress={"done": len(done), "n": len(ids)},
        summary=None,
    )
    runs._write(folder / "run.json", doc)
    todo = [q for q in ids if q not in done]
    recs = list(done.values())
    lock = threading.Lock()
    t0 = time.time()
    try:
        with (folder / "answers.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
                futs = {
                    pool.submit(answer_question, cfg, q, rankings.get(q), cache): q for q in todo
                }
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
        summary=summarise(recs, cfg.modes),
        slices=slices(recs, cfg.modes),
        totals=totals(recs),
    )
    runs._write(folder / "run.json", doc)
    if log:
        try:
            doc["mlflow_run_id"] = mlflow_log.log_answer_run(doc, folder)
        except Exception as e:  # the run is the record; a logging failure must not lose it
            doc["mlflow_error"] = f"{type(e).__name__}: {e}"
        runs._write(folder / "run.json", doc)
    return rid


def _complete(rec: dict[str, Any]) -> bool:
    return all(m.get("correct") is not None for m in rec["modes"].values())


def start(
    cfg: AnswerConfig, final: bool = False, force: bool = False, resume: str | None = None
) -> str:
    """Execute in a background thread of the viewer's process."""
    rid = resume or plan(cfg, final, force)[0]
    if not runs._ACTIVE.acquire(blocking=False):
        raise runs.RunBusyError("a run is already executing in the viewer")

    def go() -> None:
        try:
            _execute(cfg, final, force, True, None, None if resume else rid, resume)
        except BaseException:  # noqa: S110 - execute records the failure in run.json
            pass
        finally:
            runs._ACTIVE.release()

    threading.Thread(target=go, name=f"answers-{rid}", daemon=True).start()
    return rid
