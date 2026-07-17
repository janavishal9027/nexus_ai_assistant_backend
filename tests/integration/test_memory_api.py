"""The /api/memory surface: prefs, the graph view, and per-edge deletion.

The ownership assertions here matter most: these endpoints take an id from the
client, so a missing owner filter would let any account read or delete another's
memory.
"""
import pytest

from app.memory import memory_prefs
from app.models.db_models import KgEdge, MemoryEdge

pytestmark = pytest.mark.integration


def _edge(owner_id, target="Flutter", support=1):
    return MemoryEdge(owner_id=owner_id, source="User", relation="uses",
                      target=target, source_type="user", target_type="tool",
                      support_count=support)


# ── prefs ────────────────────────────────────────────────────────────────────

def test_prefs_default_to_all_on(client, make_account, auth_headers):
    acct = make_account()
    r = client.get("/api/memory/prefs", headers=auth_headers(acct))
    assert r.status_code == 200
    assert r.json()["prefs"] == {
        "recall_enabled": True, "record_enabled": True,
        "reflect_enabled": True, "graph_enabled": True}


def test_prefs_patch_is_partial_and_persists(client, make_account, auth_headers):
    acct = make_account()
    h = auth_headers(acct)
    r = client.patch("/api/memory/prefs", json={"graph_enabled": False}, headers=h)
    assert r.status_code == 200
    p = r.json()["prefs"]
    assert p["graph_enabled"] is False
    assert p["recall_enabled"] is True, "patch clobbered an untouched switch"
    assert client.get("/api/memory/prefs", headers=h).json()["prefs"] == p


def test_prefs_empty_patch_is_422(client, make_account, auth_headers):
    acct = make_account()
    assert client.patch("/api/memory/prefs", json={},
                        headers=auth_headers(acct)).status_code == 422


def test_prefs_are_owner_scoped(client, make_account, auth_headers):
    a, b = make_account(), make_account()
    client.patch("/api/memory/prefs", json={"recall_enabled": False},
                 headers=auth_headers(a))
    assert client.get("/api/memory/prefs",
                      headers=auth_headers(b)).json()["prefs"]["recall_enabled"] is True


def test_recall_off_disables_semantic_recall_too(db, make_account):
    """recall_enabled is the master switch: turning it off must also stop the
    'About the user' block, not just episodic."""
    acct = make_account()
    memory_prefs.set_prefs(acct.id, {"recall_enabled": False})
    eff = memory_prefs.effective(acct.id)
    assert eff["recall_enabled"] is False
    assert eff["semantic_recall_enabled"] is False


def test_user_switch_cannot_override_operator_flag(db, make_account, monkeypatch):
    """Effective = user AND operator. A user may turn a layer off, never on."""
    from app.config import get_settings
    acct = make_account()
    s = get_settings()
    monkeypatch.setattr(s, "memory_graph_enabled", False, raising=False)
    memory_prefs.set_prefs(acct.id, {"graph_enabled": True})
    assert memory_prefs.effective(acct.id)["graph_enabled"] is False


def test_prefs_read_fails_open_to_on(db, make_account, monkeypatch):
    """A prefs hiccup must not silently disable someone's memory."""
    acct = make_account()
    monkeypatch.setattr(memory_prefs, "_row",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert memory_prefs.get_prefs(acct.id) == {
        "recall_enabled": True, "record_enabled": True,
        "reflect_enabled": True, "graph_enabled": True}


# ── graph view ───────────────────────────────────────────────────────────────

def test_graph_returns_nodes_edges_and_ids(db, client, make_account, auth_headers):
    acct = make_account()
    db.add(_edge(acct.id))
    db.commit()
    body = client.get("/api/memory/graph", headers=auth_headers(acct)).json()
    assert len(body["edges"]) == 1
    assert body["edges"][0]["id"] is not None, "UI needs the id to delete an edge"
    assert {n["id"] for n in body["nodes"]} == {"User", "Flutter"}


def test_graph_is_strongest_first(db, client, make_account, auth_headers):
    acct = make_account()
    db.add_all([_edge(acct.id, "Weak", 1), _edge(acct.id, "Strong", 9)])
    db.commit()
    edges = client.get("/api/memory/graph", headers=auth_headers(acct)).json()["edges"]
    assert [e["target"] for e in edges] == ["Strong", "Weak"]


def test_graph_is_owner_scoped(db, client, make_account, auth_headers):
    mine, theirs = make_account(), make_account()
    db.add(_edge(theirs.id, "Secret"))
    db.commit()
    assert client.get("/api/memory/graph",
                      headers=auth_headers(mine)).json()["edges"] == []


def test_graph_requires_auth(client):
    assert client.get("/api/memory/graph").status_code == 401


# ── per-edge delete ──────────────────────────────────────────────────────────

def test_delete_own_edge(db, client, make_account, auth_headers):
    acct = make_account()
    db.add(_edge(acct.id, "Wrong"))
    db.commit()
    eid = db.query(MemoryEdge.id).filter(MemoryEdge.owner_id == acct.id).scalar()
    r = client.delete(f"/api/memory/graph/{eid}", headers=auth_headers(acct))
    assert r.status_code == 200
    assert db.query(MemoryEdge.id).filter(MemoryEdge.id == eid).scalar() is None


def test_cannot_delete_another_accounts_edge(db, client, make_account, auth_headers):
    """A guessed id must not delete someone else's fact — and the 404 must not
    reveal that the edge exists."""
    attacker, victim = make_account(), make_account()
    db.add(_edge(victim.id, "VictimSecret"))
    db.commit()
    eid = db.query(MemoryEdge.id).filter(MemoryEdge.owner_id == victim.id).scalar()
    r = client.delete(f"/api/memory/graph/{eid}", headers=auth_headers(attacker))
    assert r.status_code == 404
    assert db.query(MemoryEdge.id).filter(MemoryEdge.id == eid).scalar() is not None


def test_delete_unknown_edge_is_404(client, make_account, auth_headers):
    acct = make_account()
    assert client.delete("/api/memory/graph/99999999",
                         headers=auth_headers(acct)).status_code == 404


# ── summary ──────────────────────────────────────────────────────────────────

def test_summary_carries_prefs_and_knowledge_count(db, client, make_account,
                                                   auth_headers):
    acct = make_account()
    db.add(KgEdge(owner_id=acct.id, source="A", relation="r", target="B",
                  support_count=1))
    db.commit()
    body = client.get("/api/memory", headers=auth_headers(acct)).json()
    assert body["counts"]["knowledge"] == 1
    assert body["prefs"]["recall_enabled"] is True
