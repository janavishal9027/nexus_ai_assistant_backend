"""Provider health classification.

Getting this wrong is expensive in both directions: too eager and a good key is
benched (the router skips the whole provider); too lax and a dead provider keeps
eating the 20s fallback budget on every request.
"""
import pytest

from app.services import provider_health
from app.services.fallback_router import _NON_CHAT_HINTS, _is_chatty

pytestmark = pytest.mark.unit


class _FakeModel:
    def __init__(self, model_id, display_name=""):
        self.model_id = model_id
        self.display_name = display_name or model_id


# ── classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("error", [
    # The one that started it all: OpenRouter answers 402, which matched none of
    # the router's original auth checks ("credit card", "payment required"), so
    # it fell through to a per-model retry and walked all 351 models.
    'openrouter API error 402: {"error":{"message":"This request requires more '
    'credits, or fewer max_tokens. You requested up to 450 tokens, but can only '
    'afford 31."}}',
    'vercel API error 403: {"error":{"message":"AI Gateway requires a valid '
    'credit card on file to service requests."}}',
    'huggingface API error 403: {"error":"This authentication method does not '
    'have sufficient permissions"}',
    "google API error 403: Your API key was reported as leaked.",
    "openai API error 401: invalid api key",
    "some provider: unauthorized",
])
def test_key_problems_are_errors(error):
    assert provider_health.classify(error) == provider_health.ERROR


@pytest.mark.parametrize("error", [
    'Google API error 429: {"error":{"code":429,"message":"You exceeded your '
    'current quota, please check your plan and billing details."}}',
    'zai API error 429: {"error":{"code":"1113","message":"Insufficient balance"}}',
    "provider: rate limit exceeded",
    "429 too many requests",
    "RESOURCE_EXHAUSTED",
])
def test_rate_limits_are_limited_not_errors(error):
    """A 429 must not bench the provider — it's transient and often resets in
    seconds. Note Google's 429 text also contains 'billing', which would read as
    fatal if the order of the checks were reversed."""
    assert provider_health.classify(error) == provider_health.LIMITED


@pytest.mark.parametrize("error", [
    'mistral API error 400: {"message":"Invalid model: voxtral-mini-2602"}',
    "cerebras returned no content",
    "provider API error 500: internal server error",
    "read timeout",
    "",
    None,
])
def test_uninformative_failures_say_nothing_about_the_key(error):
    """A bad model id, an empty reply or a provider 5xx must not condemn a
    working key."""
    assert provider_health.classify(error) is None


# ── persistence ──────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_record_failure_and_recovery(db, make_account):
    from app.models.db_models import ApiKey
    acct = make_account()
    key = ApiKey(owner_id=acct.id, platform="openrouter", api_key="sk-x",
                 status="unknown")
    db.add(key)
    db.commit()

    assert provider_health.record_failure(
        db, key.id, "openrouter API error 402: requires more credits") == "error"
    db.refresh(key)
    assert key.status == "error"
    assert "more credits" in key.last_error
    assert key.last_checked_at is not None

    # A later success clears the error — the user topped up.
    provider_health.record_success(db, key.id)
    db.refresh(key)
    assert key.status == "healthy"
    assert key.last_error is None


@pytest.mark.integration
def test_uninformative_failure_does_not_change_status(db, make_account):
    from app.models.db_models import ApiKey
    acct = make_account()
    key = ApiKey(owner_id=acct.id, platform="mistral", api_key="sk-x",
                 status="healthy")
    db.add(key)
    db.commit()
    assert provider_health.record_failure(
        db, key.id, "mistral API error 400: Invalid model: voxtral-mini-2602") is None
    db.refresh(key)
    assert key.status == "healthy", "a bad model id benched a good key"


@pytest.mark.integration
def test_record_failure_never_raises_on_missing_key(db):
    assert provider_health.record_failure(db, 99999999, "401 unauthorized") is None
    provider_health.record_success(db, 99999999)      # must not raise


@pytest.mark.integration
def test_usable_filter_skips_errored_keys_then_lets_them_retry(db, make_account):
    """An errored key is benched — but not forever, or a blip (or a top-up the
    app can't see) would lock a provider out permanently."""
    from datetime import datetime, timedelta, timezone
    from app.models.db_models import ApiKey
    acct = make_account()
    fresh = ApiKey(owner_id=acct.id, platform="p1", api_key="k1", status="error",
                   last_checked_at=datetime.now(timezone.utc))
    stale = ApiKey(owner_id=acct.id, platform="p2", api_key="k2", status="error",
                   last_checked_at=datetime.now(timezone.utc) - timedelta(hours=5))
    healthy = ApiKey(owner_id=acct.id, platform="p3", api_key="k3",
                     status="healthy")
    limited = ApiKey(owner_id=acct.id, platform="p4", api_key="k4",
                     status="limited")
    db.add_all([fresh, stale, healthy, limited])
    db.commit()

    usable = {k.platform for k in db.query(ApiKey).filter(
        ApiKey.owner_id == acct.id, provider_health.usable_filter(30)).all()}
    assert "p1" not in usable, "a just-failed key should be skipped"
    assert "p2" in usable, "an old failure should get another chance"
    assert "p3" in usable
    assert "p4" in usable, "rate-limited is transient; keep trying it"


@pytest.mark.integration
def test_summary_never_exposes_the_key(db, make_account):
    from app.models.db_models import ApiKey
    acct = make_account()
    db.add(ApiKey(owner_id=acct.id, platform="openrouter",
                  api_key="sk-super-secret", status="healthy"))
    db.commit()
    rows = provider_health.summary(db, owner_id=acct.id)
    assert rows and "sk-super-secret" not in str(rows)


# ── non-chat model filtering ─────────────────────────────────────────────────

@pytest.mark.parametrize("model_id", [
    # Audio. Only the "-tts-" variants matched a hint before, so plain
    # voxtral-mini-2602 was routed chat and 400'd, burning fallback budget.
    "voxtral-mini-2602",
    "voxtral-mini-latest",
    "voxtral-small-2507",
    "voxtral-mini-transcribe-realtime-2602",
    "voxtral-mini-tts-2603",
    "voxtral-mini-asr-streaming-mellon-greek-2606-solutions",
    # Non-chat families that were already handled.
    "mistral-embed", "codestral-embed", "mistral-ocr-latest",
    "mistral-moderation-2603", "text-embedding-3-small", "whisper-large-v3",
])
def test_non_chat_models_are_excluded(model_id):
    assert not _is_chatty(_FakeModel(model_id)), f"{model_id} would be tried for chat"


@pytest.mark.parametrize("model_id", [
    "mistral-large-latest", "mistral-medium-latest", "magistral-medium-2509",
    "ministral-8b-latest", "open-mistral-nemo", "gpt-oss-120b",
    "qwen/qwen3-vl-235b-a22b-instruct", "nvidia/nemotron-3-super-120b-a12b",
    "codestral-latest", "devstral-latest",
])
def test_real_chat_models_are_kept(model_id):
    assert _is_chatty(_FakeModel(model_id)), f"{model_id} was wrongly filtered out"
