"""Plan s07 offline: what question-side tuning learns from, its report folds, the gate, the
loss, the weight swap, and how a tuned encoder enters a run's fingerprint."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from vidore_loop.data import catalog, pages
from vidore_loop.eval import runs, split
from vidore_loop.retrieval import tune, visual

# three reports: A holds pages 1–10, B 11–20, C 21–30
GOLD = {
    1: {1: 2, 2: 1},  # A; page 1 is also a test gold page
    2: {3: 2, 11: 1},  # A and B: never held out
    3: {12: 2},  # B
    4: {21: 2, 5: 1},  # C and A
    5: {22: 2},  # C; its only page is a test gold page, so it has nothing to learn from
    10: {22: 2, 1: 1},  # test
    11: {30: 2},  # test
}
TRAIN, TEST = [1, 2, 3, 4, 5], [10, 11]
PAGES = {"A": set(range(1, 11)), "B": set(range(11, 21)), "C": set(range(21, 31))}


@pytest.fixture()
def world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        catalog, "queries", lambda key: [{"query_id": q, "query": f"question {q}"} for q in GOLD]
    )
    monkeypatch.setattr(
        catalog,
        "qrels",
        lambda key: {
            q: [{"corpus_id": c, "score": s} for c, s in g.items()] for q, g in GOLD.items()
        },
    )
    monkeypatch.setattr(split, "ids", lambda key, which: {"train": TRAIN, "test": TEST}[which])
    report = {c: "A" if c <= 10 else "B" if c <= 20 else "C" for c in range(1, 31)}
    monkeypatch.setattr(
        pages, "page_index", lambda key: {c: {"doc_id": r} for c, r in report.items()}
    )


def test_training_never_sees_a_test_gold_page(world: None) -> None:
    d = tune.data("finance_en")
    assert d.blocked == {22, 1, 30}
    assert [e.query_id for e in d.examples] == [1, 2, 3, 4]  # question 5 has nothing left
    assert d.examples[0].gold == {2: 1}
    assert (d.pairs_all, d.pairs_clean) == (8, 6)
    assert all(not set(e.gold) & d.blocked for e in d.examples)


def test_a_report_fold_never_trains_on_its_report(world: None) -> None:
    fs = {f.report: f for f in tune.folds("finance_en")}
    assert {r: f.held_out for r, f in fs.items()} == {"A": [1], "B": [3], "C": [5]}
    for r, f in fs.items():
        own = PAGES[r]
        assert own <= f.blocked and {22, 1, 30} <= f.blocked
        # no training question has any gold page (clean or not) in the held-out report
        assert all(not set(GOLD[e.query_id]) & own for e in f.train)
        assert not set(f.held_out) & {e.query_id for e in f.train}
    assert [e.query_id for e in fs["C"].train] == [1, 2, 3]


def test_the_gate_needs_two_points_and_an_interval_above_zero() -> None:
    base = [0.5] * 40
    clear = tune.paired_bootstrap(base, [0.55] * 40)
    assert clear["gain"] == pytest.approx(0.05) and clear["lo"] == pytest.approx(0.05)
    assert tune.passes(clear)
    small = tune.paired_bootstrap(base, [0.51] * 40)
    assert not tune.passes(small)  # real but under +2 points
    noisy = tune.paired_bootstrap(base, [0.5 + (0.4 if i % 2 else -0.34) for i in range(40)])
    assert noisy["gain"] >= tune.GATE and noisy["lo"] < 0 and not tune.passes(noisy)
    assert tune.paired_bootstrap(base, [0.55] * 40) == clear  # seeded


def test_fit_is_refused_unless_the_sweep_cleared_the_gate() -> None:
    cv = {"status": "finished", "best": {"setting": "head-1e-4", "epoch": 2, "passes": True}}
    assert tune.gate_allows(cv, "head-1e-4", 2)
    assert not tune.gate_allows(cv, "head-1e-4", 3)
    assert not tune.gate_allows({**cv, "best": {**cv["best"], "passes": False}}, "head-1e-4", 2)
    assert not tune.gate_allows({**cv, "status": "failed"}, "head-1e-4", 2)


def test_a_tuned_encoder_joins_the_fingerprint_only_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(visual, "active", lambda key: "evie")
    monkeypatch.setattr(split, "fingerprint", lambda key: "split")
    meta = {"name": "h", "setting": "head-1e-4", "epochs": 2, "sha256": "abc123"}
    monkeypatch.setattr(tune, "describe", lambda key, name: meta)
    plain = runs.RunConfig("finance_en", "test", stages=("text", "visual", "fused"), weights=(0, 1))
    tuned = replace(plain, question_encoder="h")
    assert set(plain.models()) == {
        "text",
        "visual",
        "visual_dim",
        "reranker",
        "reranker_max_tokens",
    }
    assert tuned.models()["visual_question"] == meta
    assert plain.fingerprint() != tuned.fingerprint()
    assert (plain.label(), tuned.label()) == ("w0-1_norr", "w0-1_norr_qh")


def test_compare_splits_test_questions_by_shared_gold_pages(
    world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def queries(rid: str) -> list[dict[str, Any]]:
        score = {"base": {10: 0.5, 11: 0.8}, "tuned": {10: 0.7, 11: 0.8}}[rid]
        return [
            {"query_id": q, "stages": {"visual": {"metrics": {"ndcg_10": v}}}}
            for q, v in score.items()
        ]

    monkeypatch.setattr(runs, "queries", queries)
    out = tune.compare("base", "tuned", "finance_en")
    assert out["shares a train gold page"]["n"] == 1  # q10 shares page 1 with train question 1
    assert out["shares a train gold page"]["gain"] == pytest.approx(0.2)
    assert out["unseen gold pages"]["n"] == 1 and out["unseen gold pages"]["gain"] == 0.0
    assert out["all"]["gain"] == pytest.approx(0.1)


# ---- torch: the loss, MaxSim and the weight swap ------------------------------------


def test_the_loss_ignores_blocked_pages_and_weighs_full_pages_twice() -> None:
    torch = pytest.importorskip("torch")
    s = torch.tensor([5.0, 4.0, 3.0, 2.0])
    keep = torch.tensor([True, True, True, False])
    loss = tune.info_nce(s, {1: 2}, keep)
    expected = -(4.0 + torch.log(torch.tensor(2.0)) - torch.logsumexp(s[:3], 0))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)
    blocked_moved = s.clone()
    blocked_moved[3] = 100.0
    assert float(tune.info_nce(blocked_moved, {1: 2}, keep)) == pytest.approx(float(loss))
    lifted = s.clone()
    lifted[1] = 6.0
    assert float(tune.info_nce(lifted, {1: 2}, keep)) < float(loss)
    # a full page among the positives counts twice a partial one
    both = tune.info_nce(s, {1: 2, 2: 1}, keep)
    manual = -(torch.logsumexp(torch.tensor([4.0 + 0.6931472, 3.0]), 0) - torch.logsumexp(s[:3], 0))
    assert float(both) == pytest.approx(float(manual), rel=1e-5)


def test_maxsim_takes_each_tokens_best_unmasked_patch_and_passes_gradients() -> None:
    torch = pytest.importorskip("torch")
    docs = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]], [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    scores = visual.maxsim(q, docs, mask)
    assert scores.tolist() == pytest.approx([2.0, 1.0])  # the masked [9, 9] never counts
    scores.sum().backward()
    assert q.grad is not None and float(q.grad.abs().sum()) > 0


def test_a_tuned_encoder_is_swapped_in_for_the_block_and_always_swapped_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    enc = torch.nn.Linear(2, 2)
    shipped = enc.weight.detach().clone()
    tuned = torch.full((2, 2), 7.0)
    monkeypatch.setattr(tune, "describe", lambda key, name: {"name": name})
    monkeypatch.setattr(tune, "_tensors", lambda key, name, device: {"weight": tuned})
    monkeypatch.setattr(visual, "device", lambda: "cpu")
    with tune.applied(enc, "finance_en", "evie", "h"):
        assert torch.equal(enc.weight, tuned)
    assert torch.equal(enc.weight, shipped)
    with pytest.raises(ZeroDivisionError), tune.applied(enc, "finance_en", "evie", "h"):
        raise ZeroDivisionError
    assert torch.equal(enc.weight, shipped)
    with tune.applied(enc, "finance_en", "evie", None):  # no encoder: nothing changes
        assert torch.equal(enc.weight, shipped)
    with pytest.raises(RuntimeError), tune.applied(enc, "finance_en", "tomoro", "h"):
        pass


def test_training_maxsim_matches_search_maxsim_and_passes_gradients() -> None:
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    docs = torch.nn.functional.normalize(torch.randn(5, 7, 4, generator=gen), dim=-1)
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[1, 3:] = False
    q = torch.nn.functional.normalize(torch.randn(3, 4, generator=gen), dim=-1).requires_grad_(True)
    a = tune.maxsim_train(q, docs, mask, chunk=2)
    assert torch.allclose(a, visual.maxsim(q.detach(), docs, mask), atol=1e-6)
    a.sum().backward()
    assert q.grad is not None and bool(torch.isfinite(q.grad).all())


def test_a_non_finite_question_gradient_is_skipped_not_added() -> None:
    torch = pytest.importorskip("torch")
    t = tune.Tuner.__new__(tune.Tuner)
    t.dim = 2
    params = [torch.zeros(4, 3, requires_grad=True), torch.zeros(2, requires_grad=True)]
    masters = [p.detach().clone().requires_grad_(True) for p in params]
    for m in masters:
        m.grad = torch.zeros_like(m)
    params[0].grad, params[1].grad = torch.ones(4, 3), torch.ones(2)
    assert t._accumulate(params, masters)
    assert all(p.grad is None for p in params)
    params[0].grad, params[1].grad = torch.ones(4, 3), torch.tensor([1.0, float("nan")])
    assert not t._accumulate(params, masters)
    assert masters[1].grad.tolist() == [1.0, 1.0]  # the NaN question added nothing
    assert all(p.grad is None for p in params)
    # a step takes the mean, zeroes the head's rows past the cut, and moves the weights
    opt = torch.optim.SGD(masters, lr=1.0)
    t._step(["1.linear.weight", "0.layers.31.w"], params, masters, opt, 1)
    assert params[0][:2].sum() < 0 and params[0][2:].abs().sum() == 0
