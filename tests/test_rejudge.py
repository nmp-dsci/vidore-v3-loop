"""M5 S0 and S1 offline: aliases instead of pinned ids, and a judge check with a fake judge."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from test_answers import fake  # noqa: F401 - the fixture
from vidore_loop.answer import claude, judge
from vidore_loop.data import audits
from vidore_loop.eval import answers, rejudge, runs

SRC = Path(__file__).resolve().parents[1] / "src"
MODEL_ID = re.compile(r"claude-(haiku|sonnet|opus|fable|mythos)-\d")


def test_no_model_id_is_written_into_the_source() -> None:
    hits = [
        f"{p.relative_to(SRC)}:{n}"
        for p in SRC.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if MODEL_ID.search(line)
    ]
    assert hits == [], "name models by alias; the SDK resolves them to the newest id"


def test_an_alias_resolves_once_to_the_model_that_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude, "_RESOLVED", {})
    asked: list[str] = []

    async def fake_query(system: str, blocks: Any, model: str, *a: Any) -> claude.Reply:
        asked.append(model)
        return claude.Reply("OK", model="claude-haiku-9-9" if model == "haiku" else model)

    monkeypatch.setattr(claude, "_query", fake_query)
    assert claude.resolve("haiku") == claude.resolve(" Haiku ") == "claude-haiku-9-9"
    assert asked == ["haiku"]  # one probe, then the cache
    claude.ask("s", [], model="haiku")
    assert asked[-1] == "claude-haiku-9-9"  # later calls name the id, not the alias
    assert claude.resolve("claude-sonnet-1-0") == "claude-sonnet-1-0"  # a full id passes through


def test_a_probe_that_names_no_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude, "_RESOLVED", {})

    async def fake_query(system: str, blocks: Any, model: str, *a: Any) -> claude.Reply:
        return claude.Reply("OK", model=model)  # echoes the alias back

    monkeypatch.setattr(claude, "_query", fake_query)
    with pytest.raises(claude.ClaudeError, match="did not name its model"):
        claude.resolve("opus")


def test_kappa_and_agreement() -> None:
    assert rejudge.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert rejudge.kappa([True, True, False, False], [True, False, True, False]) == 0.0
    rows = [
        {"query_id": 1, "mode": "oracle", "source_verdict": "Correct", "source_correct": True,
         "verdict": "Correct", "correct": True},
        {"query_id": 2, "mode": "oracle", "source_verdict": "Partially Correct",
         "source_correct": False, "verdict": "Correct", "correct": True},
        {"query_id": 3, "mode": "oracle", "source_verdict": "Incorrect", "source_correct": False,
         "verdict": "Incorrect", "correct": False},
        {"query_id": 4, "mode": "oracle", "source_verdict": "Correct", "source_correct": True,
         "verdict": None, "correct": None, "error": "no reply"},
    ]  # fmt: skip
    a = rejudge.agreement(rows)
    assert a["n"] == 3 and a["binary_agree"] == pytest.approx(2 / 3)
    assert a["lenient"] == 1 and a["strict"] == 0 and a["shift"] == pytest.approx(1 / 3)
    assert a["confusion"]["Partially Correct"]["Correct"] == 1


def test_a_judge_check_rejudges_without_reanswering(
    fake: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = answers.execute(answers.AnswerConfig("finance_en", "train"), log=False)
    seen: list[tuple[str, str]] = []

    def lenient(query: str, true: str, test: str, model: str, effort: str) -> dict[str, Any]:
        seen.append((model, effort))
        return {"verdict": "Correct", "correct": True, "explanation": "", "usage": {"ms": 1}}

    monkeypatch.setattr(judge, "judge", lenient)
    rec = answers.records(src)[0]
    audits.label_judge(src, rec["query_id"], "oracle", True, rec["modes"]["oracle"]["correct"])
    cid = rejudge.execute(rejudge.JudgeCheckConfig(src, "haiku", "high"), log=False)
    doc = rejudge.get(cid)
    assert doc["status"] == "finished" and doc["models"]["judge"] == "claude-haiku-test"
    assert set(seen) == {("claude-haiku-test", "high")} and len(seen) == 10 * 2
    s = doc["summary"]
    assert s["all"]["check_accuracy"] == 1.0 and s["all"]["strict"] == 0
    assert s["all"]["lenient"] == round((1 - s["all"]["source_accuracy"]) * 20)
    assert s["people"]["n"] == 1 and s["people"]["check"] == 1.0
    assert [c["run_id"] for c in rejudge.list_checks()] == [cid]
    assert cid not in {r["run_id"] for r in runs.list_runs()}  # kept out of the Runs tab


def test_a_judge_check_of_a_test_run_needs_final(fake: Path) -> None:  # noqa: F811
    cfg = answers.AnswerConfig("finance_en", "test", modes=("oracle",))
    src = answers.execute(cfg, final=True, log=False)
    with pytest.raises(runs.TestSplitGuardError):
        rejudge.execute(rejudge.JudgeCheckConfig(src), log=False)
