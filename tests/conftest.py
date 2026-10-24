"""Shared offline guards for every test."""

from __future__ import annotations

import pytest

from vidore_loop.answer import claude


@pytest.fixture(autouse=True)
def _aliases_resolve_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aliases resolve to fixed fake ids, so no test probes the subscription."""
    monkeypatch.setattr(claude, "_RESOLVED", {a: f"claude-{a}-test" for a in claude.ALIASES})
