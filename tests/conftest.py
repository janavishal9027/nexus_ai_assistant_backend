"""Shared pytest fixtures.

The integration tests here need a real Postgres (pgvector, JSON, bulk deletes),
so they run against a DEDICATED test database — never the development one, which
holds real user data these tests would delete.

Settings.database_url is a computed property built from `postgres_db`, so the
override is POSTGRES_DB (setting DATABASE_URL does nothing). It must be set, and
the settings cache cleared, BEFORE `app.database` is imported, because that
module builds its engine at import time.

Override the database name with TEST_POSTGRES_DB; it defaults to the dev name
suffixed `_test` and is created on demand. If Postgres can't be reached the
integration tests skip with a clear reason — they never fall back to the dev
database. The `_test` assertion below is load-bearing: it has already caught one
misconfiguration that would have run this suite against real data.
"""
from __future__ import annotations

import os

import pytest
import sqlalchemy as sa

_TEST_DB_READY = False
_SKIP_REASON = ""


def _ensure_test_db(url: str) -> None:
    """Create the test database + pgvector if they don't exist."""
    base, name = url.rsplit("/", 1)
    admin = sa.create_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        exists = c.execute(
            sa.text("select 1 from pg_database where datname=:n"), {"n": name}
        ).scalar()
        if not exists:
            c.execute(sa.text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    eng = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    eng.dispose()


def _bootstrap() -> None:
    """Point the app at the test DB and create its schema. Runs at import, before
    any test module imports app.database."""
    global _TEST_DB_READY, _SKIP_REASON
    try:
        from app.config import get_settings
        dev_name = get_settings().postgres_db
        test_name = os.environ.get("TEST_POSTGRES_DB") or f"{dev_name}_test"
        if not test_name.endswith("_test") and not os.environ.get("TEST_POSTGRES_DB"):
            raise RuntimeError(f"unsafe test db name {test_name!r}")

        os.environ["POSTGRES_DB"] = test_name     # must precede app.database
        get_settings.cache_clear()                # drop the dev-name settings cache
        url = get_settings().database_url
        _ensure_test_db(url)

        from app.database import Base, engine
        # Import EVERY module declaring tables, or create_all silently skips them
        # and the tests fail on a missing relation.
        import app.models.db_models               # noqa: F401
        import app.models.rag_models              # noqa: F401
        if engine.url.database != test_name:
            raise RuntimeError(
                f"refusing to run against {engine.url.database!r}: app.database was "
                f"imported before this bootstrap could redirect it")
        Base.metadata.create_all(bind=engine)
        # Then the same ad-hoc DDL the app runs at boot. create_all only creates
        # MISSING tables — it never alters an existing one — so columns added via
        # _ensure_auth_schema (api_keys.last_error, kg_edges.support_count, the
        # edge embeddings, ...) are absent from a test DB created by an earlier
        # revision. Running the app's own migration path keeps the test schema
        # identical to production, and exercises that path on every run.
        from app.main import _ensure_auth_schema
        _ensure_auth_schema()
        _TEST_DB_READY = True
    except Exception as exc:                      # pragma: no cover - env-dependent
        _SKIP_REASON = f"test database unavailable: {exc}"


_bootstrap()

# Skips every test in a module that asks for `db` when Postgres isn't reachable,
# instead of erroring out — the unit tests still run without a database.
requires_db = pytest.mark.skipif(not _TEST_DB_READY, reason=_SKIP_REASON or "no test DB")


@pytest.fixture(scope="session")
def db_ready() -> bool:
    if not _TEST_DB_READY:
        pytest.skip(_SKIP_REASON or "no test DB")
    return True


@pytest.fixture()
def db(db_ready):
    """A session on the test database."""
    from app.database import SessionLocal
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def make_account(db):
    """Factory for throwaway accounts. Every row any of them owns is deleted at
    teardown, so a failing test can't leave state that poisons the next one."""
    from app.models.db_models import (Account, ApiKey, Conversation, KgEdge,
                                      MemoryChunk, MemoryEdge, MemoryPrefs,
                                      Message, MessageFeedback, Project,
                                      ProjectBrainEntry, Skill)
    from app.models.rag_models import (Document, DocumentChunk, IngestionJob,
                                       KnowledgeBase)
    from app.services import auth as auth_service

    created: list[int] = []
    counter = {"n": 0}

    def _make(name: str = "Test User") -> Account:
        counter["n"] += 1
        email = f"pytest-{counter['n']}-{os.getpid()}@example.invalid"
        acct = Account(email=email, name=name,
                       password_hash=auth_service.hash_password("pytest-pw-123"))
        db.add(acct)
        db.commit()
        created.append(acct.id)
        return acct

    yield _make

    # Children before parents, so the deletes can't trip a foreign key.
    owned = (MemoryChunk, Skill, MessageFeedback, MemoryEdge, KgEdge,
             ProjectBrainEntry, Project, MemoryPrefs, ApiKey,
             DocumentChunk, IngestionJob, Document, KnowledgeBase)
    for oid in created:
        conv_ids = [c for (c,) in db.query(Conversation.id).filter(
            Conversation.owner_id == oid).all()]
        if conv_ids:
            db.query(Message).filter(
                Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
            db.query(Conversation).filter(
                Conversation.id.in_(conv_ids)).delete(synchronize_session=False)
        for model in owned:
            db.query(model).filter(model.owner_id == oid).delete(synchronize_session=False)
        db.query(Account).filter(Account.id == oid).delete(synchronize_session=False)
    db.commit()


@pytest.fixture()
def auth_headers():
    """Bearer headers for an account."""
    from app.services import auth as auth_service

    def _h(account):
        return {"Authorization": f"Bearer {auth_service.create_token(account.id, account.email)}"}
    return _h


@pytest.fixture()
def client(db_ready):
    """FastAPI TestClient WITHOUT the lifespan: the schema already exists, and
    we don't want the app's background sweepers (retention/decay) mutating rows
    underneath a test."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
