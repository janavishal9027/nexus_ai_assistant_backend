"""Memory lifecycle: summary / export / purge scopes, and knowledge-graph parity.

Covers the guarantees a user relies on in Settings → Memory:
  - a purge scope deletes exactly its own layer and nothing else,
  - an unknown scope is rejected rather than silently meaning "all",
  - export carries every layer (portability),
  - the content graph reinforces repeats and decays, like the personal one.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.memory import data_lifecycle, memory_graph
from app.models.db_models import (KgEdge, MemoryChunk, MemoryEdge,
                                  MessageFeedback, Skill)
from app.rag import knowledge_graph

pytestmark = pytest.mark.integration


def _seed_all_layers(db, owner_id: int) -> None:
    db.add_all([
        MemoryChunk(owner_id=owner_id, text="an exchange"),
        Skill(owner_id=owner_id, kind="preference", content="likes brevity"),
        MessageFeedback(owner_id=owner_id, message_index=0, rating=1),
        MemoryEdge(owner_id=owner_id, source="User", relation="uses",
                   target="Flutter", support_count=1),
        KgEdge(owner_id=owner_id, source="App", relation="uses",
               target="Postgres", support_count=1),
    ])
    db.commit()


# ── summary / export ─────────────────────────────────────────────────────────

def test_summary_counts_every_layer(db, make_account):
    acct = make_account()
    _seed_all_layers(db, acct.id)
    s = data_lifecycle.summary(acct.id)
    assert s == {"episodic": 1, "skills": 1, "feedback": 1, "graph": 1,
                 "knowledge": 1}


def test_export_includes_every_layer(db, make_account):
    """Export is the portability promise — a layer missing here is data the user
    cannot get out."""
    acct = make_account()
    _seed_all_layers(db, acct.id)
    ex = data_lifecycle.export_memory(acct.id)
    for layer in ("episodic", "skills", "feedback", "graph", "knowledge"):
        assert len(ex[layer]) == 1, f"export missing {layer}"
    assert ex["counts"]["knowledge"] == 1
    assert ex["knowledge"][0]["support_count"] == 1


def test_export_is_owner_scoped(db, make_account):
    mine, theirs = make_account(), make_account()
    _seed_all_layers(db, theirs.id)
    ex = data_lifecycle.export_memory(mine.id)
    assert all(not ex[l] for l in ("episodic", "skills", "feedback", "graph",
                                   "knowledge"))


# ── purge ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scope,expected_empty", [
    ("episodic", "episodic"),
    ("skills", "skills"),
    ("feedback", "feedback"),
    ("graph", "graph"),
    ("knowledge", "knowledge"),
])
def test_purge_scope_deletes_only_its_own_layer(db, make_account, scope,
                                                expected_empty):
    acct = make_account()
    _seed_all_layers(db, acct.id)
    data_lifecycle.purge_memory(acct.id, scope)
    s = data_lifecycle.summary(acct.id)
    assert s[expected_empty] == 0, f"scope={scope} did not clear {expected_empty}"
    for layer, n in s.items():
        if layer != expected_empty:
            assert n == 1, f"scope={scope} also cleared {layer}"


def test_purge_all_clears_everything(db, make_account):
    acct = make_account()
    _seed_all_layers(db, acct.id)
    data_lifecycle.purge_memory(acct.id, "all")
    assert set(data_lifecycle.summary(acct.id).values()) == {0}


def test_purge_is_owner_scoped(db, make_account):
    mine, theirs = make_account(), make_account()
    _seed_all_layers(db, mine.id)
    _seed_all_layers(db, theirs.id)
    data_lifecycle.purge_memory(mine.id, "all")
    assert set(data_lifecycle.summary(theirs.id).values()) == {1}


def test_unknown_purge_scope_raises_instead_of_wiping(db, make_account):
    """A typo'd scope once meant "all" — a 200 that destroyed everything."""
    acct = make_account()
    _seed_all_layers(db, acct.id)
    with pytest.raises(ValueError):
        data_lifecycle.purge_memory(acct.id, "skill")     # note: not "skills"
    assert set(data_lifecycle.summary(acct.id).values()) == {1}, "data was wiped"


def test_unknown_purge_scope_is_422_over_http(client, make_account, auth_headers,
                                              db):
    acct = make_account()
    _seed_all_layers(db, acct.id)
    r = client.delete("/api/memory?scope=skill", headers=auth_headers(acct))
    assert r.status_code == 422
    assert set(data_lifecycle.summary(acct.id).values()) == {1}


def test_purge_http_scope_knowledge(client, make_account, auth_headers, db):
    acct = make_account()
    _seed_all_layers(db, acct.id)
    r = client.delete("/api/memory?scope=knowledge", headers=auth_headers(acct))
    assert r.status_code == 200
    assert data_lifecycle.summary(acct.id)["knowledge"] == 0
    assert data_lifecycle.summary(acct.id)["graph"] == 1


def test_scopes_constant_matches_reality():
    assert set(data_lifecycle.SCOPES) == {
        "all", "episodic", "skills", "feedback", "graph", "knowledge"}


# ── knowledge-graph parity: reinforcement + decay ────────────────────────────

def test_kg_reinforces_repeats_rather_than_discarding(db, make_account):
    """A fact restated is evidence it's real; support_count is what ranks recall
    and resists decay."""
    acct = make_account()
    triple = {"source": "Acme", "relation": "uses", "target": "Postgres",
              "source_type": "org", "target_type": "tech"}
    for _ in range(3):
        knowledge_graph._store(acct.id, None, 1, [dict(triple)])
    rows = db.query(KgEdge).filter(KgEdge.owner_id == acct.id).all()
    assert len(rows) == 1, "repeat inserted a duplicate row"
    assert rows[0].support_count == 3
    assert rows[0].updated_at is not None


def test_kg_decay_fades_then_drops_stale_edges(db, make_account):
    acct = make_account()
    knowledge_graph._store(acct.id, None, 1, [
        {"source": "Old", "relation": "was", "target": "Fact",
         "source_type": "concept", "target_type": "concept"}])
    stale = datetime.now(timezone.utc) - timedelta(days=999)
    db.query(KgEdge).filter(KgEdge.owner_id == acct.id).update(
        {KgEdge.updated_at: stale, KgEdge.support_count: 2},
        synchronize_session=False)
    db.commit()

    knowledge_graph.decay(days=180)          # 2 -> 1, still present
    db.expire_all()
    row = db.query(KgEdge).filter(KgEdge.owner_id == acct.id).first()
    assert row is not None and row.support_count == 1

    knowledge_graph.decay(days=180)          # weak + stale -> dropped
    db.expire_all()
    assert db.query(KgEdge).filter(KgEdge.owner_id == acct.id).count() == 0


def test_decay_preserves_updated_at_so_edges_keep_ageing(db, make_account):
    """A bulk update fires updated_at's onupdate, which would reset staleness and
    leave the edge decaying one point per sweep forever."""
    acct = make_account()
    knowledge_graph._store(acct.id, None, 1, [
        {"source": "Strong", "relation": "is", "target": "Fact",
         "source_type": "concept", "target_type": "concept"}])
    stale = datetime.now(timezone.utc) - timedelta(days=999)
    db.query(KgEdge).filter(KgEdge.owner_id == acct.id).update(
        {KgEdge.updated_at: stale, KgEdge.support_count: 5},
        synchronize_session=False)
    db.commit()
    knowledge_graph.decay(days=180)
    db.expire_all()
    row = db.query(KgEdge).filter(KgEdge.owner_id == acct.id).first()
    assert row.updated_at.replace(tzinfo=timezone.utc) < stale + timedelta(days=1)


def test_decay_is_disabled_by_zero(db, make_account):
    acct = make_account()
    knowledge_graph._store(acct.id, None, 1, [
        {"source": "Keep", "relation": "me", "target": "Please",
         "source_type": "c", "target_type": "c"}])
    db.query(KgEdge).filter(KgEdge.owner_id == acct.id).update(
        {KgEdge.updated_at: datetime.now(timezone.utc) - timedelta(days=999)},
        synchronize_session=False)
    db.commit()
    assert knowledge_graph.decay(days=0) == 0
    assert db.query(KgEdge).filter(KgEdge.owner_id == acct.id).count() == 1


def test_memory_graph_decay_matches(db, make_account):
    acct = make_account()
    memory_graph._store(acct.id, [
        {"source": "User", "relation": "used", "target": "OldTool",
         "source_type": "user", "target_type": "tool"}])
    db.query(MemoryEdge).filter(MemoryEdge.owner_id == acct.id).update(
        {MemoryEdge.updated_at: datetime.now(timezone.utc) - timedelta(days=999),
         MemoryEdge.support_count: 1}, synchronize_session=False)
    db.commit()
    memory_graph.decay(days=90)
    db.expire_all()
    assert db.query(MemoryEdge).filter(MemoryEdge.owner_id == acct.id).count() == 0


def test_retention_keeps_recent_and_drops_old(db, make_account):
    acct = make_account()
    db.add_all([
        MemoryChunk(owner_id=acct.id, text="recent"),
        MemoryChunk(owner_id=acct.id, text="ancient",
                    created_at=datetime.now(timezone.utc) - timedelta(days=999)),
    ])
    db.commit()
    data_lifecycle.apply_retention(365)
    db.expire_all()
    left = [c.text for c in db.query(MemoryChunk).filter(
        MemoryChunk.owner_id == acct.id).all()]
    assert left == ["recent"]


def test_retention_zero_keeps_everything(db, make_account):
    acct = make_account()
    db.add(MemoryChunk(owner_id=acct.id, text="ancient",
                       created_at=datetime.now(timezone.utc) - timedelta(days=999)))
    db.commit()
    assert data_lifecycle.apply_retention(0) == 0
    assert db.query(MemoryChunk).filter(MemoryChunk.owner_id == acct.id).count() == 1
