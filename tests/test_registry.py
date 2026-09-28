import pytest

from vidore_loop.data import registry


def test_totals_match_the_blog_post() -> None:
    # huggingface.co/blog/QuentinJG/introducing-vidore-v3, public datasets table
    assert len(registry.DATASETS) == 8
    assert sum(d.pages for d in registry.DATASETS) == 19256
    assert sum(d.queries for d in registry.DATASETS) == 2419


def test_every_dataset_is_pinned() -> None:
    for d in registry.DATASETS:
        assert len(d.revision) == 40 and len(d.paper_revision) == 40
        assert d.repo_id == f"vidore/vidore_v3_{d.key}"


def test_get_unknown_names_the_choices() -> None:
    assert registry.get("hr").pages == 1110
    with pytest.raises(KeyError, match="finance_en"):
        registry.get("nuclear")
