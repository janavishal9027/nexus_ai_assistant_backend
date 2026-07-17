"""DELETE /api/auth/me must erase every owner-scoped row.

Account ids are sequential, so anything left behind is inherited by the next
account to take that id — a deleted user's memories surfacing in a new user's
prompts. This suite is the regression net for that: if someone adds an
owner-scoped table and forgets `_OWNER_SCOPED` in routes/auth.py,
`test_no_owner_scoped_table_is_missed` fails.
"""
import pytest

from app.models.db_models import (Account, ApiKey, Conversation, KgEdge,
                                  MemoryChunk, MemoryEdge, MemoryPrefs, Message,
                                  MessageFeedback, Project, ProjectBrainEntry,
                                  Skill)
from app.models.rag_models import (Document, DocumentChunk, IngestionJob,
                                   KnowledgeBase)

pytestmark = pytest.mark.integration

# Every table keyed by owner_id. Mirrors routes/auth.py::_OWNER_SCOPED —
# test_no_owner_scoped_table_is_missed keeps both honest against the ORM.
OWNER_SCOPED = (MemoryChunk, Skill, MessageFeedback, MemoryEdge, KgEdge,
                ProjectBrainEntry, Project, MemoryPrefs, ApiKey,
                DocumentChunk, IngestionJob, Document, KnowledgeBase)


def _seed_rag(db, owner_id: int, tag: str) -> None:
    """The RAG chain: knowledge base → document → chunk + job, all owner-scoped."""
    kb = KnowledgeBase(owner_id=owner_id, name=f"{tag} kb")
    db.add(kb)
    db.commit()
    doc = Document(owner_id=owner_id, knowledge_base_id=kb.id,
                   filename=f"{tag}.pdf")
    db.add(doc)
    db.commit()
    db.add_all([
        DocumentChunk(owner_id=owner_id, document_id=doc.id,
                      knowledge_base_id=kb.id, ordinal=0, text=f"{tag} chunk"),
        IngestionJob(owner_id=owner_id, document_id=doc.id,
                     knowledge_base_id=kb.id),
    ])
    db.commit()


def _seed(db, owner_id: int, tag: str) -> int:
    """One row in every owner-scoped table. Returns the conversation id."""
    conv = Conversation(title=f"{tag} chat", owner_id=owner_id)
    db.add(conv)
    db.commit()
    db.add(Message(conversation_id=conv.id, role="user", content=f"{tag} msg"))
    db.add_all([
        MemoryChunk(owner_id=owner_id, conversation_id=conv.id, text=f"{tag} chunk"),
        # No conversation: this row is only reachable by owner_id. Deleting
        # chunks by conversation_id (as the code once did) would strand it.
        MemoryChunk(owner_id=owner_id, conversation_id=None, text=f"{tag} orphan"),
        Skill(owner_id=owner_id, kind="preference", content=f"{tag} skill"),
        MessageFeedback(owner_id=owner_id, conversation_id=conv.id,
                        message_index=0, rating=1),
        MemoryEdge(owner_id=owner_id, source="User", relation="uses",
                   target=f"{tag}Tool"),
        KgEdge(owner_id=owner_id, source="Doc", relation="mentions",
               target=f"{tag}Thing"),
        ProjectBrainEntry(owner_id=owner_id, project_id=1, kind="fact",
                          content=f"{tag} fact"),
        Project(owner_id=owner_id, name=f"{tag} project"),
        MemoryPrefs(owner_id=owner_id, recall_enabled=False),
        ApiKey(owner_id=owner_id, platform="openrouter", api_key=f"{tag}-key"),
    ])
    db.commit()
    _seed_rag(db, owner_id, tag)
    return conv.id


def _counts(db, owner_id: int) -> dict:
    out = {m.__name__: db.query(m).filter(m.owner_id == owner_id).count()
           for m in OWNER_SCOPED}
    out["Conversation"] = db.query(Conversation).filter(
        Conversation.owner_id == owner_id).count()
    return out


def test_deleting_an_account_leaves_nothing_behind(db, client, make_account, auth_headers):
    acct = make_account()
    oid = acct.id                 # capture: `acct` is unusable once its row is gone
    headers = auth_headers(acct)
    _seed(db, oid, "victim")
    seeded = _counts(db, oid)
    assert all(v > 0 for v in seeded.values()), f"seed failed: {seeded}"

    r = client.delete("/api/auth/me", headers=headers)
    assert r.status_code == 200, r.text

    db.expire_all()
    leaked = {k: v for k, v in _counts(db, oid).items() if v}
    assert not leaked, f"owner-scoped rows survived account deletion: {leaked}"
    assert db.query(Account).filter(Account.id == oid).first() is None


def test_deleting_an_account_does_not_touch_other_accounts(db, client, make_account,
                                                           auth_headers):
    victim, bystander = make_account(), make_account()
    victim_headers = auth_headers(victim)
    victim_id, bystander_id = victim.id, bystander.id
    _seed(db, victim_id, "victim")
    _seed(db, bystander_id, "bystander")
    before = _counts(db, bystander_id)

    r = client.delete("/api/auth/me", headers=victim_headers)
    assert r.status_code == 200

    db.expire_all()
    assert _counts(db, bystander_id) == before, "deletion crossed account boundaries"
    assert db.query(Account).filter(Account.id == bystander_id).first() is not None


def test_no_owner_scoped_table_is_missed(db):
    """Fails when a model gains an owner_id but isn't wired into account
    deletion — the failure mode is invisible at runtime, so it's asserted here."""
    from app.routes.auth import _OWNER_SCOPED
    from app.database import Base

    covered = {m.__name__ for m, _ in _OWNER_SCOPED} | {"Conversation", "Account"}
    with_owner = {
        m.class_.__name__
        for m in Base.registry.mappers
        if any(c.name == "owner_id" for c in m.local_table.columns)
    }
    missed = with_owner - covered
    assert not missed, (
        f"owner-scoped table(s) not deleted with the account: {sorted(missed)}. "
        f"Add them to _OWNER_SCOPED in app/routes/auth.py.")


def test_delete_requires_auth(client):
    assert client.delete("/api/auth/me").status_code == 401
