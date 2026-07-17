"""graph_recall: the shared edge-recall logic behind both graphs.

Pure-logic tests — no database, no embedding provider, no network.
"""
import pytest

from app.memory import graph_recall
from app.models.db_models import KgEdge, MemoryEdge

pytestmark = pytest.mark.unit


def test_edge_text_is_the_embedded_form():
    assert graph_recall.edge_text("User", "uses", "Java") == "User uses Java"


@pytest.mark.parametrize("model", [MemoryEdge, KgEdge])
def test_keyword_filter_searches_relation_not_just_endpoints(model):
    """The relation column was never searched, so "what do I use" could not match
    an edge whose relation is `uses`."""
    sql = str(graph_recall.keyword_filter(model, "what do I use")).lower()
    assert "relation" in sql
    assert "source" in sql
    assert "target" in sql


def test_keyword_filter_ignores_short_terms():
    # "do"/"I" are noise; only >=3-char terms are used.
    assert graph_recall.keyword_filter(MemoryEdge, "do I go") is None


def test_keyword_filter_none_when_no_usable_terms():
    assert graph_recall.keyword_filter(MemoryEdge, "") is None
    assert graph_recall.keyword_filter(MemoryEdge, "   ") is None


def test_cosine_basics():
    assert graph_recall._cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert graph_recall._cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert graph_recall._cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_is_safe_on_bad_input():
    """Recall must never raise — a zero/absent/mismatched vector scores 0."""
    assert graph_recall._cosine([], [1, 0]) == 0.0
    assert graph_recall._cosine([1, 0], [1, 0, 0]) == 0.0
    assert graph_recall._cosine([0, 0], [0, 0]) == 0.0


def test_threshold_comes_from_config():
    from app.config import get_settings
    assert graph_recall._threshold() == pytest.approx(
        get_settings().memory_graph_recall_threshold)


def test_threshold_falls_back_when_config_unreadable(monkeypatch):
    monkeypatch.setattr(graph_recall, "get_settings",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)
    # _threshold imports get_settings internally; simulate a broken settings read
    import app.config
    monkeypatch.setattr(app.config, "get_settings",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert graph_recall._threshold() == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_embed_edges_returns_triples_even_without_a_provider(monkeypatch):
    """An edge with no vector is still worth storing — it recalls via keyword."""
    async def _no_vectors(owner_id, texts, input_type):
        return []
    monkeypatch.setattr(graph_recall, "embed", _no_vectors)
    triples = [{"source": "A", "relation": "r", "target": "B"}]
    out = await graph_recall.embed_edges(1, triples)
    assert out == triples
    assert "embedding" not in out[0]


@pytest.mark.asyncio
async def test_embed_edges_attaches_vector_and_dim(monkeypatch):
    async def _vecs(owner_id, texts, input_type):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr(graph_recall, "embed", _vecs)
    out = await graph_recall.embed_edges(1, [{"source": "A", "relation": "r",
                                              "target": "B"}])
    assert out[0]["embedding"] == [0.1, 0.2, 0.3]
    assert out[0]["embedding_dim"] == 3


@pytest.mark.asyncio
async def test_embed_edges_handles_empty():
    assert await graph_recall.embed_edges(1, []) == []


@pytest.mark.asyncio
async def test_embed_never_raises(monkeypatch):
    def _boom(db, owner_id):
        raise RuntimeError("no provider")
    monkeypatch.setattr(graph_recall, "resolve_embedding_provider", _boom)
    assert await graph_recall.embed(1, ["x"], graph_recall.INPUT_QUERY) == []


@pytest.mark.asyncio
async def test_search_returns_empty_for_blank_query():
    assert await graph_recall.search(MemoryEdge, 1, [], "") == []
    assert await graph_recall.search(MemoryEdge, 1, [], "   ") == []
