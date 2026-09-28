import math

import pytest

from vidore_loop.cli import SMOKE_QRELS, SMOKE_RANKING
from vidore_loop.eval.metrics import mean_over_queries, ndcg_at_k, recall_at_k


def test_ndcg_graded_linear_gain() -> None:
    # DCG = 2/log2(2) + 1/log2(4) = 2.5; IDCG = 2 + 1/log2(3) + 1/log2(4)
    ideal = 2 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(SMOKE_RANKING, SMOKE_QRELS) == pytest.approx(2.5 / ideal)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k([11, 12, 13], SMOKE_QRELS) == pytest.approx(1.0)
    assert ndcg_at_k([], SMOKE_QRELS) == 0.0
    assert ndcg_at_k([1, 2], {}) == 0.0


def test_ndcg_cuts_at_k_and_ignores_duplicates() -> None:
    assert ndcg_at_k([99, 11], {11: 1}, k=1) == 0.0
    assert ndcg_at_k([11, 11, 12], {11: 1, 12: 1}, k=2) == pytest.approx(1.0)


def test_recall() -> None:
    assert recall_at_k(SMOKE_RANKING, SMOKE_QRELS) == pytest.approx(2 / 3)
    assert recall_at_k(SMOKE_RANKING, SMOKE_QRELS, k=1) == pytest.approx(1 / 3)


def test_mean_over_queries_scores_missing_ranking_zero() -> None:
    qrels = {1: {10: 2}, 2: {20: 1}}
    assert mean_over_queries({1: [10]}, qrels) == pytest.approx(0.5)
    assert mean_over_queries({1: [10], 2: [20]}, qrels, metric="recall") == pytest.approx(1.0)
