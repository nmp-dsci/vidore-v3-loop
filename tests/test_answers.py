"""M4 offline: the prompts, the judge's parsing, an answer run with a fake Claude, audits, the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from test_runs import _questions
from vidore_loop.answer import claude, generate, judge
from vidore_loop.data import audits, catalog, pages
from vidore_loop.eval import answers, runs, split
from vidore_loop.serving import app as app_module


def test_billing_refuses_a_key_under_the_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(claude.BillingError):
        claude.check_billing()
    env = claude.sealed_env({"ANTHROPIC_API_KEY": "sk-test", "PATH": "/bin", "SECRET": "x"})
    assert env["ANTHROPIC_API_KEY"] == "" and env["SECRET"] == "" and env["PATH"] == "/bin"


def test_session_limit_waits_for_the_named_reset() -> None:
    from datetime import datetime

    now = datetime(2026, 10, 9, 15, 0)
    assert claude.seconds_until_reset("session limit · resets 4:40pm", now) == 100 * 60 + 60
    assert claude.seconds_until_reset("overloaded", now) is None


def test_the_prompts_follow_the_paper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8jpeg")
    monkeypatch.setattr(pages, "image_path", lambda key, cid: img)
    monkeypatch.setattr(
        pages,
        "page_meta",
        lambda key, cid: {"doc_id": "bank_10k", "page": cid, "markdown": "Deposits 376,007"},
    )
    blocks = generate.build("finance_en", "total deposits?", [7, 9], "image+text")
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "text", "image", "text", "text", "image", "text", "text"]
    assert blocks[0]["text"].startswith("You are an expert at answering query based on documents.")
    assert (
        blocks[-1]["text"].endswith("Output the answer only.")
        and "total deposits?" in blocks[-1]["text"]
    )
    assert [b["type"] for b in generate.build("finance_en", "q", [7], "image")].count("image") == 1


def test_the_judge_counts_only_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(
        system: str, blocks: Any, model: str = "", effort: str = "", schema: Any = None
    ) -> claude.Reply:
        assert schema == judge.SCHEMA and "True Answer: 42" in blocks[0]["text"]
        return claude.Reply(
            "", structured={"explanation": "close", "judgment": "Partially Correct"}
        )

    monkeypatch.setattr(claude, "ask", fake)
    j = judge.judge("q", "42", "about 40")
    assert j["verdict"] == "Partially Correct" and j["correct"] is False
    monkeypatch.setattr(claude, "ask", lambda *a, **k: claude.Reply("not json"))
    with pytest.raises(claude.ClaudeError):
        judge.judge("q", "42", "x")


# ---- an answer run with a fake Claude --------------------------------------------


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    qs = _questions(20)
    for q in qs:
        q["answer"] = f"answer {q['query_id']}"
    by_id = {q["query_id"]: q for q in qs}
    qrels = {q["query_id"]: q["pages"] for q in qs}
    monkeypatch.setattr(split, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(split, "SIZE", {"train": 10, "test": 10})
    monkeypatch.setattr(catalog, "queries", lambda key: qs)
    monkeypatch.setattr(catalog, "question", lambda key, qid: by_id.get(qid))
    monkeypatch.setattr(catalog, "qrels", lambda key: qrels)
    monkeypatch.setattr(catalog, "hand_checks", lambda: {})
    monkeypatch.setattr(audits, "AUDITS_DIR", tmp_path / "audits")
    monkeypatch.setenv("VIDORE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    split.write("finance_en")
    # a finished retrieval run to answer from: the full-answer page first for even questions
    rid = "20260101T000000Z_finance_en_w0-1_norr_train"
    folder = tmp_path / "runs" / rid
    folder.mkdir(parents=True)
    train = split.ids("finance_en", "train")
    (folder / "run.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "kind": "retrieval",
                "dataset": "finance_en",
                "split": "train",
                "status": "finished",
                "fingerprint": "abc",
                "split_fingerprint": split.fingerprint("finance_en"),
                "config": {"stages": ["fused"], "weights": [0.0, 1.0], "limit": None},
            }
        )
    )
    with (folder / "queries.jsonl").open("w") as f:
        for qid in train:
            ranking = [1000 + qid, 7, 8] if qid % 2 == 0 else [7, 8, 9]
            f.write(json.dumps({"query_id": qid, "stages": {"fused": {"ranking": ranking}}}) + "\n")

    def fake_answer(key: str, query: str, ids: Any, *a: Any) -> dict[str, Any]:
        qid = int(query.split()[-1])
        # right whenever the full-answer page was given
        right = 1000 + qid in ids
        return {
            "answer": f"answer {qid}" if right else "I don't know",
            "pages": ids,
            "usage": {"input_tokens": 10, "output_tokens": 1, "ms": 5, "cost_usd": 0.01},
        }

    def fake_judge(query: str, true: str, test: str, *a: Any) -> dict[str, Any]:
        ok = true == test
        return {
            "verdict": "Correct" if ok else "Incorrect",
            "correct": ok,
            "explanation": "",
            "usage": {"input_tokens": 5, "output_tokens": 1, "ms": 1, "cost_usd": 0.0},
        }

    monkeypatch.setattr(generate, "answer", fake_answer)
    monkeypatch.setattr(judge, "judge", fake_judge)
    return tmp_path


def test_an_answer_run_scores_each_setting(fake: Path) -> None:
    rid = answers.execute(answers.AnswerConfig("finance_en", "train", workers=2), log=False)
    doc = runs.get(rid)
    assert doc["kind"] == "answer" and doc["status"] == "finished" and doc["n"] == 10
    assert doc["config"]["source_run"].endswith("_w0-1_norr_train")
    s = doc["summary"]
    evens = sum(1 for q in split.ids("finance_en", "train") if q % 2 == 0)
    assert s["oracle"]["raw"] == {"mean": 1.0, "n": 10}
    assert s["retrieved"]["raw"]["mean"] == pytest.approx(evens / 10)
    assert s["retrieval_cost"]["raw"] == pytest.approx(1 - evens / 10)
    assert s["retrieved"]["correct_when_full_given"]["mean"] == 1.0
    assert doc["totals"]["calls"] == 10 * 2 * 2
    recs = answers.records(rid)
    odd = next(r for r in recs if r["query_id"] % 2)
    assert (
        odd["modes"]["retrieved"]["pages"] == [7, 8, 9]
        and odd["modes"]["retrieved"]["full_given"] is False
    )
    assert odd["modes"]["oracle"]["pages"][0] == 1000 + odd["query_id"]  # grade 2 first


def test_audits_change_the_audited_score_not_the_raw(fake: Path) -> None:
    rid = answers.execute(
        answers.AnswerConfig("finance_en", "train", modes=("retrieved",)), log=False
    )
    recs = answers.records(rid)
    wrong = next(r for r in recs if not r["modes"]["retrieved"]["correct"])
    right = next(r for r in recs if r["modes"]["retrieved"]["correct"])
    before = answers.audited_view(runs.get(rid), recs)["retrieved"]
    audits.set_reference("finance_en", wrong["query_id"], "disputed", "misreads the table")
    audits.label_judge(rid, right["query_id"], "retrieved", correct=False, judge_correct=True)
    after = answers.audited_view(runs.get(rid), recs)["retrieved"]
    assert after["raw"] == before["raw"]
    assert after["audited"]["n"] == before["audited"]["n"] - 1
    n_ok = sum(1 for r in recs if r["modes"]["retrieved"]["correct"]) - 1
    assert after["audited"]["mean"] == pytest.approx(n_ok / (len(recs) - 1))
    audits.set_reference("finance_en", wrong["query_id"], "holds")
    assert wrong["query_id"] not in audits.disputed("finance_en")
    with pytest.raises(ValueError):
        audits.set_reference("finance_en", 1, "maybe")


def test_answer_runs_are_guarded_and_resumable(fake: Path) -> None:
    with pytest.raises(ValueError):  # no retrieval run on test to answer from
        answers.plan(answers.AnswerConfig("finance_en", "test"), final=True)
    cfg = answers.AnswerConfig("finance_en", "test", modes=("oracle",))
    with pytest.raises(runs.TestSplitGuardError):
        answers.plan(cfg)
    answers.execute(cfg, final=True, log=False)
    with pytest.raises(runs.TestSplitGuardError):
        answers.plan(cfg, final=True)
    # an interrupted train run: half its questions on disk, carried on by --resume
    rid = answers.execute(answers.AnswerConfig("finance_en", "train", modes=("oracle",)), log=False)
    p = runs.runs_dir() / rid / "answers.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:4]) + "\n")
    answers.execute(answers.AnswerConfig("finance_en", "train"), resume=rid, log=False)
    assert len(answers.records(rid)) == 10 and runs.get(rid)["summary"]["oracle"]["raw"]["n"] == 10


def test_the_answer_api(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pages, "page_index", lambda key: {})
    rid = answers.execute(answers.AnswerConfig("finance_en", "train"), log=False)
    c = TestClient(app_module.create_app())
    assert [r["run_id"] for r in c.get("/api/runs?kind=answer").json()] == [rid]
    assert all(r.get("kind", "retrieval") == "retrieval" for r in c.get("/api/runs").json())
    doc = c.get(f"/api/runs/{rid}").json()
    assert (
        doc["audited"]["oracle"]["audited"]["mean"] == 1.0 and doc["calibration"]["labelled"] == 0
    )
    rows = c.get(f"/api/runs/{rid}/answers").json()
    qid = rows[0]["query_id"]
    assert set(rows[0]["modes"]) == {"retrieved", "oracle"}
    one = c.get(f"/api/runs/{rid}/answers/{qid}").json()
    assert one["modes"]["oracle"]["pages"][0]["grade"] == 2 and one["next"] == rows[1]["query_id"]
    r = c.post(
        "/api/audits/judge",
        json={"run_id": rid, "query_id": qid, "mode": "oracle", "correct": False},
    )
    assert r.status_code == 200 and r.json()["agrees"] is False
    assert c.get(f"/api/runs/{rid}").json()["calibration"] == {
        "labelled": 1,
        "agree": 0,
        "target": 40,
    }
    r = c.post(
        "/api/audits/reference",
        json={"dataset": "finance_en", "query_id": qid, "status": "disputed"},
    )
    assert r.status_code == 200
    assert c.get(f"/api/runs/{rid}/answers").json()[0]["reference"] == "disputed"
    assert (
        c.post(
            "/api/audits/reference", json={"dataset": "finance_en", "query_id": qid, "status": "?"}
        ).status_code
        == 422
    )
    live = c.post(
        "/api/answer", json={"dataset": "finance_en", "query_id": qid, "corpus_ids": [1000 + qid]}
    ).json()
    assert live["judge"]["correct"] is True and live["reference"] == f"answer {qid}"


def test_every_run_has_the_same_setup_sections(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vidore_loop.eval import setup
    from vidore_loop.tracking import mlflow_log

    monkeypatch.setattr(mlflow_log, "run_url", lambda rid: None)
    rid = answers.execute(answers.AnswerConfig("finance_en", "train"), log=False)
    c = TestClient(app_module.create_app())
    ans = c.get(f"/api/runs/{rid}").json()["setup"]
    src_id = runs.get(rid)["config"]["source_run"]
    ret = setup.describe({**runs.get(src_id), "models": {"visual": "v", "visual_dim": 128}})
    sections = ["data", "retrieval", "answer", "judge", "provenance"]
    assert [s["section"] for s in ans] == sections == [s["section"] for s in ret]
    rows = {r["k"]: r for s in ans for r in s["rows"]}
    assert rows["source run"]["link"] == f"/runs/{src_id}"
    assert "top 5" in rows["pages given"]["v"] and rows["reranker"]["v"] == "not used"
    assert {r["k"]: r["v"] for r in ret[2]["rows"]} == {"answerer": "—, retrieval only"}


def test_answers_are_reused_when_claude_would_see_the_same_pages(fake: Path) -> None:
    first = answers.execute(
        answers.AnswerConfig("finance_en", "train", modes=("oracle",)), log=False
    )
    rid = answers.execute(answers.AnswerConfig("finance_en", "train"), log=False)
    t = runs.get(rid)["totals"]
    assert t["reused"] == 10 and t["calls"] == 10 * 2  # only `retrieved` was asked
    recs = answers.records(rid)
    assert all(r["modes"]["oracle"]["cached_from"] == first for r in recs)
    assert all("cached_from" not in r["modes"]["retrieved"] for r in recs)
    assert runs.get(rid)["summary"]["oracle"]["raw"] == {"mean": 1.0, "n": 10}
    # a different answering model is a different answer: nothing reused
    other = answers.execute(
        answers.AnswerConfig("finance_en", "train", model="haiku"), log=False, reuse=True
    )
    assert runs.get(other)["totals"]["reused"] == 0
    fresh = answers.execute(answers.AnswerConfig("finance_en", "train"), log=False, reuse=False)
    assert runs.get(fresh)["totals"] == {**t, "calls": 40, "reused": 0, **_tokens(40)}


def _tokens(calls: int) -> dict[str, Any]:
    # the fake answer costs 10 in / 1 out / $0.01, the fake judge 5 in / 1 out / $0
    return {"input_tokens": 15 * calls // 2, "output_tokens": calls, "cost_usd": 0.01 * calls / 2}


def test_a_source_run_must_be_finished_full_and_cover_the_split(fake: Path) -> None:
    rid = runs.get(next(iter(sorted(p.name for p in (fake / "runs").iterdir()))))["run_id"]
    path = fake / "runs" / rid / "run.json"
    good = json.loads(path.read_text())
    cfg = answers.AnswerConfig("finance_en", "train", source_run=rid)
    answers.plan(cfg)

    path.write_text(json.dumps({**good, "status": "interrupted"}))
    with pytest.raises(ValueError, match="finished"):
        answers.plan(cfg)
    path.write_text(json.dumps({**good, "config": {**good["config"], "limit": 3}}))
    with pytest.raises(ValueError, match="finished"):
        answers.plan(cfg)

    path.write_text(json.dumps(good))
    q = fake / "runs" / rid / "queries.jsonl"
    q.write_text("\n".join(q.read_text().splitlines()[:4]) + "\n")
    with pytest.raises(ValueError, match="no ranking"):
        answers.plan(cfg)


def _install_query(monkeypatch: pytest.MonkeyPatch, replies: list[claude.Reply]) -> list[int]:
    calls: list[int] = []

    async def fake_query(*a: Any, **k: Any) -> claude.Reply:
        calls.append(1)
        return replies[min(len(calls), len(replies)) - 1]

    monkeypatch.setattr(claude, "_query", fake_query)
    monkeypatch.setattr(claude.time, "sleep", lambda s: None)
    return calls


def test_an_errored_reply_is_retried_not_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bad = claude.Reply("", error="error_max_turns: partial")
    calls = _install_query(monkeypatch, [bad, claude.Reply("fine")])
    assert claude.ask("s", []).text == "fine" and len(calls) == 2
    calls = _install_query(monkeypatch, [bad])
    with pytest.raises(claude.ClaudeError, match="error_max_turns"):
        claude.ask("s", [])
    assert len(calls) == claude.RETRIES


def test_session_limit_waits_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = _install_query(
        monkeypatch, [claude.Reply("", error="session limit reached, resets soon")]
    )
    with pytest.raises(claude.ClaudeError, match="session-limit"):
        claude.ask("s", [])
    assert len(calls) == claude.MAX_RESET_WAITS + 1
