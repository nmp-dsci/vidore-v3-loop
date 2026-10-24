"""M5 S3 offline: the agent's tools and caps, the ranking fill, and an agent run with a fake Claude."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from test_answers import fake  # noqa: F401 - the fixture
from vidore_loop.answer import agent, claude, judge
from vidore_loop.data import pages
from vidore_loop.eval import agents, runs


@pytest.fixture()
def corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8jpeg")
    known = {7, 8, 9, *range(1000, 1100)}
    monkeypatch.setattr(
        pages,
        "page_meta",
        lambda key, cid: (
            {"doc_id": "bank_10k", "page": cid, "markdown": f"Deposits {cid} " * 80}
            if cid in known
            else None
        ),
    )
    monkeypatch.setattr(pages, "image_path", lambda key, cid: img if cid in known else None)
    monkeypatch.setattr(pages, "page_index", lambda key: {c: {} for c in known})
    return tmp_path


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def test_search_is_capped_and_shows_ten_snippets(corpus: Path) -> None:
    tr = agent.Trace()
    out = agent.run_search("finance_en", tr, "deposits", 2, lambda k, q: list(range(1000, 1050)))
    text = out["content"][0]["text"]
    assert text.count("page 10") == 10 and "1 searches left" in text
    assert len(text.splitlines()[1].strip()) <= agent.SNIPPET
    agent.run_search("finance_en", tr, "again", 2, lambda k, q: [7, 8])
    capped = agent.run_search("finance_en", tr, "third", 2, lambda k, q: [9])
    assert "limit reached" in capped["content"][0]["text"] and len(tr.searches) == 2
    assert [s.get("refused") for s in tr.steps] == [None, None, "cap"]


def test_view_page_gives_image_and_text_once_and_is_capped(corpus: Path) -> None:
    tr = agent.Trace()
    out = agent.run_view("finance_en", tr, 7, 1)
    assert [c["type"] for c in out["content"]] == ["text", "image", "text"]
    assert "already shown" in agent.run_view("finance_en", tr, "7", 1)["content"][0]["text"]
    assert "limit reached" in agent.run_view("finance_en", tr, 8, 1)["content"][0]["text"]
    assert agent.run_view("finance_en", tr, 5, 9)["is_error"] is True  # no such page
    assert tr.viewed == [7]


def test_a_short_ranking_is_filled_from_rrf_over_the_searches() -> None:
    assert agent.fill([5, 6], [[1, 2, 3], [2, 4]], k=5) == [5, 6, 2, 1, 4]
    assert agent.fill(list(range(12)), [[99]]) == list(range(10))
    assert agent.fill([1], []) == [1]


def test_the_prompt_names_the_caps_and_keeps_the_papers_answer_line() -> None:
    s = agent.system_prompt("finance_en", 5, 10)
    assert "search 5 times" in s and "view 10 pages" in s
    assert "Keep the response short when appropriate. Output the answer only." in s


def _fake_agent(final: dict[str, Any], calls: list[str]) -> Any:
    def stand_in(system: str, blocks: Any, kit: Any, **kw: Any) -> claude.Reply:
        kit()  # a fresh attempt's tools
        calls.append(blocks[0]["text"])
        _run(kit.handlers["search"]({"query": blocks[0]["text"]}))
        _run(kit.handlers["view_page"]({"page_id": 7}))
        return claude.Reply("", structured=final, turns=3, cost_usd=0.02, ms=9, input_tokens=100)

    return stand_in


def test_the_agent_returns_a_filled_ranking_and_its_steps(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    final = {"ranking": [7, 7, "x", 404], "answer": " 42 ", "cited": [7, 404]}
    monkeypatch.setattr(claude, "agent", _fake_agent(final, calls))
    a = agent.answer("finance_en", "q?", search_fn=lambda k, q: [8, 9, 7])
    assert a["agent_ranking"] == [7] and a["ranking"] == [7, 8, 9]  # cleaned, then filled
    assert a["answer"] == "42" and a["cited"] == [7]
    assert [s["tool"] for s in a["steps"]] == ["search", "view_page"]
    assert a["searches"] == 1 and a["views"] == 1 and a["usage"]["turns"] == 3
    monkeypatch.setattr(
        claude, "agent", _fake_agent({"ranking": [], "answer": "", "cited": []}, [])
    )
    with pytest.raises(claude.ClaudeError, match="no structured reply"):
        agent.answer("finance_en", "q?", search_fn=lambda k, q: [8])


def test_an_agent_run_scores_ranking_and_answer(
    fake: Path,  # noqa: F811
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_answer(key: str, query: str, *a: Any) -> dict[str, Any]:
        qid = int(query.split()[-1])
        ranking = [1000 + qid, 7, 8]  # the full-answer page first
        return {
            "ranking": ranking,
            "agent_ranking": ranking,
            "answer": f"answer {qid}" if qid % 2 == 0 else "no idea",
            "cited": [1000 + qid],
            "steps": [{"tool": "search", "query": query, "results": ranking}],
            "searches": 1,
            "views": 0,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "cost_usd": 0.01,
                "ms": 5,
                "turns": 2,
            },
        }

    monkeypatch.setattr(agent, "answer", fake_answer)
    rid = agents.execute(agents.AgentConfig("finance_en", "train"), log=False)
    doc = runs.get(rid)
    assert doc["kind"] == "agent" and doc["status"] == "finished"
    assert doc["models"]["answer"] == "claude-sonnet-test"
    s = doc["summary"]
    assert s["errors"] == 0 and s["searches"] == 1.0 and s["turns"] == 2.0
    assert s["answers"]["raw"]["n"] == 10
    assert s["retrieval"]["ndcg_10"]["n"] == 10
    recs = agents.records(rid)
    assert all(r["cited_gold"] for r in recs)
    even = [r for r in recs if r["query_id"] % 2 == 0]
    assert all(r["modes"]["agent"]["correct"] for r in even)
    assert doc["totals"]["calls"] == 20  # an agent call and a judge call per question
    assert rid not in {r["run_id"] for r in runs.list_runs() if r.get("kind") == "answer"}


def test_an_agent_run_on_test_needs_final(fake: Path) -> None:  # noqa: F811
    with pytest.raises(runs.TestSplitGuardError):
        agents.plan(agents.AgentConfig("finance_en", "test"))
    with pytest.raises(runs.TestSplitGuardError):
        agents.plan(agents.AgentConfig("finance_en", "test", limit=3), final=True)
    assert judge.VERDICTS  # the judge module is the one the run calls


def test_the_api_draws_an_agent_run_and_pools_train_with_test(
    fake: Path,  # noqa: F811
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from vidore_loop.eval import answers
    from vidore_loop.serving import app as app_module

    def fake_answer(key: str, query: str, *a: Any) -> dict[str, Any]:
        qid = int(query.split()[-1])
        return {
            "ranking": [1000 + qid, 7],
            "agent_ranking": [1000 + qid],
            "answer": f"answer {qid}",
            "cited": [1000 + qid],
            "steps": [
                {"tool": "search", "query": query, "results": [1000 + qid, 7]},
                {"tool": "view_page", "page": 1000 + qid},
            ],
            "searches": 1,
            "views": 1,
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost_usd": 0.01, "ms": 5},
        }

    monkeypatch.setattr(agent, "answer", fake_answer)
    rid = agents.execute(agents.AgentConfig("finance_en", "train", limit=3), log=False)
    c = TestClient(app_module.create_app())
    doc = c.get(f"/api/agent-runs/{rid}").json()
    assert len(doc["questions"]) == 3 and doc["questions"][0]["correct"] is True
    qid = doc["questions"][1]["query_id"]
    one = c.get(f"/api/agent-runs/{rid}/{qid}").json()
    assert [s["tool"] for s in one["steps"]] == ["search", "view_page"]
    assert one["steps"][1]["page"]["grade"] == 2 and one["ranking"][0]["grade"] == 2
    assert one["prev"] is not None and one["next"] is not None and one["agent_ranked"] == 1
    assert c.get(f"/api/agent-runs/{rid}/999999").status_code == 404
    # a configuration scored on train and once on test pools to every question
    answers.execute(answers.AnswerConfig("finance_en", "train", modes=("oracle",)), log=False)
    answers.execute(
        answers.AnswerConfig("finance_en", "test", modes=("oracle",)), final=True, log=False
    )
    row = c.get("/api/leaderboard").json()["ours_answers"][0]
    assert row["all"]["oracle"]["n"] == 20 and row["accuracy"]["oracle"] is not None
