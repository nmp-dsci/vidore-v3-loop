"""What a person decided on the Review tab (M4): a reference disputed, or a judge overruled.

Two append-only logs under `data/audits/`, committed with the runs they audit:

- `references.jsonl`: a reference answer marked `disputed` (it misreads its page) or
  `holds`; the latest entry per question wins. A hand check (`hand_checks.jsonl`) that
  found the reference `off` counts as disputed until someone marks it `holds`.
- `judge_labels.jsonl`: a person's own verdict on one judged answer (`correct` true or
  false). Where it differs from the judge's, the audited score uses the person's.

Audited accuracy leaves disputed questions out and applies the person's labels; the raw
accuracy is the judge's alone. Both are reported (s00 §8: a disputed failure never
drives an optimiser edit).
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidore_loop.config import DATA_DIR
from vidore_loop.data import catalog

AUDITS_DIR = DATA_DIR / "audits"
REFERENCE_STATES = ("disputed", "holds")
_LOCK = threading.Lock()


def _path(name: str) -> Path:
    return AUDITS_DIR / name


def _read(name: str) -> list[dict[str, Any]]:
    p = _path(name)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _append(name: str, rec: dict[str, Any]) -> dict[str, Any]:
    rec = {**rec, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    with _LOCK:
        p = _path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


# ---- references -----------------------------------------------------------------


def references(key: str) -> dict[int, dict[str, Any]]:
    """query id → its current reference state {status, note, by, at, source}."""
    out: dict[int, dict[str, Any]] = {}
    for (ds, qid), c in catalog.hand_checks().items():
        if ds == key and c["verdict"] == "off":
            out[qid] = {
                "status": "disputed",
                "note": c["note"],
                "by": c["checked_by"],
                "at": c["date"],
                "source": "hand check",
            }
    for r in _read("references.jsonl"):
        if r["dataset"] == key:
            out[int(r["query_id"])] = {
                "status": r["status"],
                "note": r.get("note", ""),
                "by": r.get("by", ""),
                "at": r["at"],
                "source": "review",
            }
    return out


def disputed(key: str) -> set[int]:
    return {q for q, r in references(key).items() if r["status"] == "disputed"}


def set_reference(
    key: str, query_id: int, status: str, note: str = "", by: str = "reviewer"
) -> dict[str, Any]:
    if status not in REFERENCE_STATES:
        raise ValueError(f"status is one of {REFERENCE_STATES}, not {status!r}")
    if catalog.question(key, query_id) is None:
        raise KeyError(f"{key} has no question {query_id}")
    return _append(
        "references.jsonl",
        {"dataset": key, "query_id": query_id, "status": status, "note": note, "by": by},
    )


# ---- judge labels ---------------------------------------------------------------


def judge_labels(run_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    """(query id, mode) → the person's latest label on that answer of this run."""
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for r in _read("judge_labels.jsonl"):
        if r["run_id"] == run_id:
            out[(int(r["query_id"]), r["mode"])] = r
    return out


def label_judge(
    run_id: str,
    query_id: int,
    mode: str,
    correct: bool,
    judge_correct: bool,
    note: str = "",
    by: str = "reviewer",
) -> dict[str, Any]:
    return _append(
        "judge_labels.jsonl",
        {
            "run_id": run_id,
            "query_id": query_id,
            "mode": mode,
            "correct": bool(correct),
            "judge_correct": bool(judge_correct),
            "agrees": bool(correct) == bool(judge_correct),
            "note": note,
            "by": by,
        },
    )
