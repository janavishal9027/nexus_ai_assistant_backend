import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text

from .database import engine, SessionLocal, Base
from .routes import chat, conversations, keys, models, config
from .routes import agent as agent_routes
from .routes import auth as auth_routes
from .routes import knowledge as knowledge_routes
from .routes import projects as projects_routes
# Import RAG models so their tables are registered with Base.metadata before
# create_all runs (knowledge_bases, documents, document_chunks, ingestion_jobs).
from .models import rag_models  # noqa: F401
from .services.model_seeder import seed_models
from .services.feature_flags import get_agent_features
from .services.ws_manager import ws_manager
from .config import get_settings

# Import tools to trigger tool_registry registration via decorators
from . import tools  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_auth_schema() -> None:
    """Add conversations.owner_id to pre-auth databases (create_all does not
    alter existing tables). Idempotent."""
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS owner_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_owner_id ON conversations (owner_id)"))
            conn.execute(text("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS owner_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_api_keys_owner_id ON api_keys (owner_id)"))
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_parent_id ON conversations (parent_id)"))
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS project_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_project_id ON conversations (project_id)"))
        logger.info("[Auth] Ensured conversations.owner_id and api_keys.owner_id columns exist")
    except Exception as exc:
        logger.warning(f"[Auth] Could not ensure owner_id column: {exc}")


def _ensure_vector_extension() -> None:
    """Enable pgvector BEFORE create_all so document_chunks.embedding (a VECTOR
    column) can be created. Idempotent; best-effort so boot still proceeds (with
    the JSON fallback) where the extension is unavailable."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("[RAG] pgvector extension ensured")
    except Exception as exc:
        logger.warning(f"[RAG] Could not enable pgvector extension: {exc}")


def _ensure_rag_schema() -> None:
    """RAG schema bits create_all can't do: the KB tag on conversations, and the
    functional GIN index that powers keyword (full-text) search over chunks."""
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS knowledge_base_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_kb_id ON conversations (knowledge_base_id)"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_fts "
                "ON document_chunks USING gin (to_tsvector('english', text))"
            ))
            # Per-conversation RAG (A.3): chat-attached documents live on a
            # conversation instead of a KB.
            conn.execute(text("ALTER TABLE documents ALTER COLUMN knowledge_base_id DROP NOT NULL"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS conversation_id INTEGER"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_platform VARCHAR(64)"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128)"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_dim INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_documents_conversation_id ON documents (conversation_id)"))
            conn.execute(text("ALTER TABLE document_chunks ALTER COLUMN knowledge_base_id DROP NOT NULL"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS conversation_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chunks_conversation_id ON document_chunks (conversation_id)"))
        logger.info("[RAG] Ensured RAG schema (KB + per-conversation columns + FTS index)")
    except Exception as exc:
        logger.warning(f"[RAG] Could not ensure RAG schema: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, seed models, sync from providers
    logger.info("Starting ChatApp backend...")
    _ensure_vector_extension()          # must precede create_all (VECTOR columns)
    Base.metadata.create_all(bind=engine)
    _ensure_auth_schema()
    _ensure_rag_schema()

    db = SessionLocal()
    try:
        seed_models(db)
    finally:
        db.close()

    # Auto-sync free models from providers on startup
    from .services.model_sync import sync_all_providers
    db = SessionLocal()
    try:
        result = await sync_all_providers(db)
        logger.info(f"Model sync: {result}")
    except Exception as e:
        logger.warning(f"Model sync skipped: {e}")
    finally:
        db.close()

    # ─── Agent orchestration feature startup (flag-gated; req 15.6, 15.8) ───
    await _start_agent_features()

    logger.info("ChatApp backend ready on port 8080")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await _stop_agent_features()


def _warn_if_default(flag_name: str, env_var: str) -> None:
    if not os.environ.get(env_var):
        logger.warning(
            f"[Agent] {flag_name} enabled but {env_var} is not set; using the documented default"
        )


async def _start_agent_features() -> None:
    """Initialize agent infrastructure for each enabled feature. Fails fast when
    a flag-enabled service is unreachable at its configured/default address."""
    flags = get_agent_features()
    if not flags.any_enabled():
        logger.info("[Agent] No AGENT_FEATURES enabled; running legacy pipeline only")
        return
    settings = get_settings()

    if flags.redis_cache:
        _warn_if_default("redis_cache", "REDIS_URL")
        from .services import redis_cache as _rc
        _rc.redis_cache = _rc.RedisCache(settings.redis_url)
        if not await _rc.redis_cache.ping():
            raise RuntimeError("Redis unreachable at startup (redis_cache feature enabled)")
        logger.info("[Agent] Redis cache started")

    if flags.kafka:
        _warn_if_default("kafka", "KAFKA_BOOTSTRAP_SERVERS")
        from .services import kafka_producer as _kp
        _kp.kafka_producer = _kp.KafkaProducer(settings.kafka_bootstrap_servers)
        await _kp.kafka_producer.start()          # raises if broker unreachable
        from .services import kafka_consumer as _kc
        _kc.kafka_consumer = _kc.KafkaConsumer(
            settings.kafka_bootstrap_servers, ws_manager, _kc.event_buffer
        )
        await _kc.kafka_consumer.start()
        logger.info("[Agent] Kafka producer/consumer started")

    if flags.websocket:
        await ws_manager.start()
        logger.info("[Agent] WebSocket manager started")

    if flags.fcm:
        if not settings.fcm_credentials_path:
            raise RuntimeError("fcm feature enabled but FCM_CREDENTIALS_PATH is empty")
        from .services import fcm_notifier as _fcm
        _fcm.fcm_notifier = _fcm.FCMNotifier(settings.fcm_credentials_path)
        logger.info("[Agent] FCM notifier started")


async def _stop_agent_features() -> None:
    flags = get_agent_features()
    if flags.websocket:
        await ws_manager.stop()
    if flags.kafka:
        from .services import kafka_producer as _kp, kafka_consumer as _kc
        if _kp.kafka_producer:
            await _kp.kafka_producer.stop()
        if _kc.kafka_consumer:
            await _kc.kafka_consumer.stop()
    if flags.redis_cache:
        from .services import redis_cache as _rc
        if _rc.redis_cache:
            await _rc.redis_cache.close()


app = FastAPI(
    title="Nexus AI Backend",
    description="ChatGPT-like backend with agent orchestration, multi-provider LLM routing, and fallback",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(config.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(keys.router)
app.include_router(models.router)
app.include_router(agent_routes.router)
app.include_router(auth_routes.router)
app.include_router(knowledge_routes.router)
app.include_router(projects_routes.router)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/health")
def health():
    """Liveness + dependency health (chat-module A.6). Reachable-but-a-dependency-
    down returns 200 with status='degraded' so the client can tell a degraded
    backend apart from an unreachable one."""
    components = {}
    status = "healthy"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        components["database"] = "healthy"
    except Exception as exc:
        logger.warning(f"[Health] database check failed: {exc}")
        components["database"] = "down"
        status = "degraded"
    return {"status": status, "components": components}
