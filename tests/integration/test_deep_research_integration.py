"""Integration tests for Intellectual Routing (intellectual-routing spec task 8.3).

Drives the real HTTP endpoints `/api/chat/send` and `/api/chat/stream` with
`deep_research=true` end-to-end (route → agent orchestrator → fallback router →
relay) against an in-memory SQLite DB, a stubbed account, and a scripted provider
that truncates then rate-limits so the relay must hand off to a second model.

Asserts the core spec guarantees: the answer is assembled across both models with
the models-used footer, and the user is never asked to type "continue".
"""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import get_db, Base
from app.services.auth import get_current_account
from app.models.db_models import ChatModel, ApiKey
from app.services import fallback_router as fr


# ─── In-memory DB wired into the app ─────────────────────────────────────────

@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = TestingSession()

    # Two large (>=400B) models on different platforms so a hand-off is possible.
    db.add_all([
        ChatModel(id=1, platform="nvidia", model_id="llama-3.1-405b-instruct",
                  display_name="Llama-3.1-405B", size_label="Frontier",
                  enabled=True, priority=0),
        ChatModel(id=2, platform="openrouter", model_id="deepseek-r1",
                  display_name="DeepSeek-R1", size_label="Frontier",
                  enabled=True, priority=1),
        ApiKey(id=1, platform="nvidia", api_key="nv-key", enabled=True,
               status="healthy", owner_id=None),
        ApiKey(id=2, platform="openrouter", api_key="or-key", enabled=True,
               status="healthy", owner_id=None),
    ])
    db.commit()
    yield db, TestingSession
    db.close()


class _StubAccount:
    id = 1
    email = "test@example.com"


class _RelayProvider:
    """The pool is ordered largest-first, so DeepSeek-R1 (671B) runs before
    Llama-3.1-405B. DeepSeek truncates once (finish_reason=length) then
    rate-limits; Llama-405B receives the work-so-far and finishes. Exercises
    same-model continuation + cross-model hand-off in one relay."""
    def __init__(self):
        self.state = {}

    async def stream_chat_completion_ex(self, api_key, messages, model_id,
                                        temperature=None, max_tokens=None):
        n = self.state.get(model_id, 0)
        self.state[model_id] = n + 1
        if model_id == "deepseek-r1":
            if n == 0:
                yield {"type": "content", "text": "## Project\nfile1 done. "}
                yield {"type": "finish", "reason": "length"}   # truncated
            else:
                raise RuntimeError("429 too many requests")     # now rate-limited
        else:  # llama-3.1-405b continuation
            yield {"type": "content", "text": "file2 done. All complete."}
            yield {"type": "finish", "reason": "stop"}


@pytest.fixture()
def client(db_session, monkeypatch):
    db, TestingSession = db_session

    def _override_get_db():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_account] = lambda: _StubAccount()

    provider = _RelayProvider()
    monkeypatch.setattr(fr.provider_registry, "get", lambda platform: provider)
    monkeypatch.setattr(fr.rate_limit, "is_on_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(fr.rate_limit, "record_success", lambda *a, **k: None)
    monkeypatch.setattr(fr.rate_limit, "set_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(fr.rate_limit, "record_rate_limit_hit", lambda *a, **k: None)
    monkeypatch.setattr(fr.rate_limit, "is_retryable_error", lambda e: True)
    # Deep Research otherwise fires a live web search; stub it out.
    import app.services.agent as agent_mod

    async def _no_web(*a, **k):
        return None
    monkeypatch.setattr(agent_mod, "web_search", _no_web)

    # TestClient() without a context manager does not run the lifespan (no model
    # seeding / network sync) — exactly what we want for an isolated test.
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_send_deep_research_relays_across_models(client):
    resp = client.post("/api/chat/send",
                       json={"message": "Build a full project", "deep_research": True})
    assert resp.status_code == 200
    body = resp.json()
    content = body["content"]
    # Both segments present → the relay continued across models.
    assert "file1 done." in content and "file2 done. All complete." in content
    # Models-used footer names both contributors in order.
    assert "DeepSeek-R1 → Llama-3.1-405B" in content
    assert "2 models" in content
    # The user was never asked to type "continue".
    assert "type continue" not in content.lower()
    assert body["model"] == "Deep Research"


def test_stream_deep_research_relays_across_models(client):
    with client.stream("POST", "/api/chat/stream",
                       json={"message": "Build a full project", "deep_research": True}) as resp:
        assert resp.status_code == 200
        content = ""
        for line in resp.iter_lines():
            if not line:
                continue
            payload = line[len("data: "):] if line.startswith("data: ") else line
            try:
                evt = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if evt.get("content"):
                content += evt["content"]

    assert "file1 done." in content and "file2 done. All complete." in content
    assert "DeepSeek-R1 → Llama-3.1-405B" in content        # footer
    assert "Continuing with **Llama-3.1-405B**" in content      # visible hand-off
