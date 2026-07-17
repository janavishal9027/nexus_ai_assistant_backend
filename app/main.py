import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text

from .database import engine, SessionLocal, Base
from .models.db_models import HAS_PGVECTOR
from .routes import chat, conversations, keys, models, config
from .routes import agent as agent_routes
from .routes import auth as auth_routes
from .routes import knowledge as knowledge_routes
from .routes import projects as projects_routes
from .routes import memory as memory_routes
from .routes import appearance as appearance_routes
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
            # Why a key is unhealthy ("out of credits", "key reported as
            # leaked"), so Settings can explain a failure instead of showing a
            # status that was never written at all.
            conn.execute(text("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS last_error TEXT"))
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_parent_id ON conversations (parent_id)"))
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS project_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_project_id ON conversations (project_id)"))
            # messages.stopped: mark turns stopped mid-stream (partial); such
            # answers are not offered as a downloadable document.
            conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS stopped BOOLEAN DEFAULT FALSE"))
            # kg_edges support_count/updated_at: the content graph gained
            # reinforcement + decay, matching the personal graph. Existing rows
            # backfill to support 1 / now, so they age from this boot rather
            # than being swept immediately as infinitely stale.
            conn.execute(text("ALTER TABLE kg_edges ADD COLUMN IF NOT EXISTS support_count INTEGER DEFAULT 1"))
            conn.execute(text("ALTER TABLE kg_edges ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"))
            conn.execute(text("UPDATE kg_edges SET support_count = 1 WHERE support_count IS NULL"))
            conn.execute(text("UPDATE kg_edges SET updated_at = COALESCE(created_at, NOW()) WHERE updated_at IS NULL"))
            # Edge embeddings, so graph recall matches on meaning rather than
            # shared substrings. Pre-existing edges keep a NULL vector and are
            # still reachable via the keyword fallback.
            _emb_type = "vector" if HAS_PGVECTOR else "json"
            for _tbl in ("kg_edges", "memory_edges"):
                conn.execute(text(
                    f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS embedding {_emb_type}"))
                conn.execute(text(
                    f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS embedding_dim INTEGER"))
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
            # Semantic-embedding enhancements: dedup hash + per-chunk metadata,
            # parent/child links, and per-chunk embedding provenance.
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents (content_hash)"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key VARCHAR(512)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chunks_content_hash ON document_chunks (content_hash)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section VARCHAR(512)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS char_start INTEGER"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS char_end INTEGER"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_chunk_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chunks_parent_id ON document_chunks (parent_chunk_id)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS is_parent BOOLEAN DEFAULT FALSE"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chunks_is_parent ON document_chunks (is_parent)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(160)"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(64)"))
            # Phase 3: fixed-dim ANN mirror column (the HNSW index is built by
            # _ensure_ann_index in its own transaction below).
            from .models.rag_models import HAS_PGVECTOR as _HP, ANN_DIM as _AD
            if _HP:
                conn.execute(text(
                    f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_ann vector({_AD})"))
        logger.info("[RAG] Ensured RAG schema (KB + per-conversation columns + FTS index + chunk metadata)")
    except Exception as exc:
        logger.warning(f"[RAG] Could not ensure RAG schema: {exc}")
    _ensure_ann_index()


def _ensure_memory_schema() -> None:
    """Part D episodic memory: per-user owner scoping + an unbounded embedding
    column (so real embeddings of any dim work), and a one-time clear of the old
    fake SHA-256 vectors (they were never real embeddings). Idempotent."""
    try:
        from .models.db_models import HAS_PGVECTOR
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS owner_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_chunks_owner_id ON memory_chunks (owner_id)"))
            conn.execute(text("ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS embedding_dim INTEGER"))
            if HAS_PGVECTOR:
                # Convert the embedding column to an UNBOUNDED vector once — the DB
                # may have it as legacy JSON (pgvector-absent fallback) or a fixed
                # vector(1536). Guard on the current type so we don't rewrite every
                # boot. The old vectors were fake hashes, so we clear them first
                # (making the type cast trivial).
                cur = conn.execute(text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = 'memory_chunks'::regclass AND attname = 'embedding'"
                )).scalar()
                if cur and cur != "vector":
                    conn.execute(text("DROP INDEX IF EXISTS memory_chunks_embedding_idx"))
                    conn.execute(text("ALTER TABLE memory_chunks ALTER COLUMN embedding DROP NOT NULL"))
                    conn.execute(text("UPDATE memory_chunks SET embedding = NULL"))
                    conn.execute(text(
                        "ALTER TABLE memory_chunks ALTER COLUMN embedding "
                        "TYPE vector USING embedding::text::vector"))
        logger.info("[Memory] Ensured episodic schema (owner scoping + unbounded embedding)")
    except Exception as exc:
        logger.warning(f"[Memory] Could not ensure memory schema: {exc}")


def _ensure_ann_index() -> None:
    """Build the HNSW index on the fixed-dim ANN mirror column. Best-effort in its
    own transaction so a build failure never aborts the rest of the schema."""
    try:
        from .models.rag_models import HAS_PGVECTOR
        if not HAS_PGVECTOR:
            return
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_ann_hnsw "
                "ON document_chunks USING hnsw (embedding_ann vector_cosine_ops)"))
        logger.info("[RAG] Ensured HNSW ANN index on document_chunks.embedding_ann")
    except Exception as exc:
        logger.warning(f"[RAG] HNSW ANN index skipped ({exc}); ANN mirror uses seq scan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, seed models, sync from providers
    logger.info("Starting ChatApp backend...")
    _ensure_vector_extension()          # must precede create_all (VECTOR columns)
    Base.metadata.create_all(bind=engine)
    _ensure_auth_schema()
    _ensure_rag_schema()
    _ensure_memory_schema()

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
    # ─── RAG scale-out (Phase 4): object store + event indexing + OTel ──────
    await _start_rag_scaleout()
    # ─── Memory lifecycle (Part D Phase 3): retention sweeper (gated) ───────
    _start_memory_lifecycle()

    logger.info("ChatApp backend ready on port 8080")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await _stop_agent_features()
    await _stop_rag_scaleout()
    await _stop_memory_lifecycle()


async def _start_rag_scaleout() -> None:
    """Phase 4 opt-in scale-out backends. All default-off; the app runs the
    in-process pipeline (BackgroundTasks + in-DB bytes) when they're disabled."""
    s = get_settings()
    if (getattr(s, "rag_object_store", "db") or "db").lower() == "s3":
        from .services.object_store import get_object_store
        get_object_store()      # fail fast if boto3 / MinIO is misconfigured
        logger.info("[RAG] S3/MinIO object store ready")
    if getattr(s, "rag_otel_enabled", False):
        try:
            from .services.rag_observability import init_otel
            init_otel(s.rag_otel_endpoint)
            logger.info("[RAG] OpenTelemetry span export enabled")
        except Exception as exc:
            logger.warning(f"[RAG] OTel init skipped ({exc})")
    if getattr(s, "rag_kafka_indexing", False):
        from .services import rag_events
        await rag_events.start()            # raises if the broker is unreachable
        from .workers import rag_index_worker
        rag_index_worker.start_in_process()
        logger.info("[RAG] Kafka event-driven indexing enabled (producer + in-process worker)")


async def _stop_rag_scaleout() -> None:
    if not getattr(get_settings(), "rag_kafka_indexing", False):
        return
    try:
        from .workers import rag_index_worker
        await rag_index_worker.stop_in_process()
        from .services import rag_events
        await rag_events.stop()
    except Exception:
        pass


_retention_task = None


def _start_memory_lifecycle() -> None:
    """Background sweeps (Part D): episodic retention (Phase 3), personal
    memory-graph decay (Phase 5), and content knowledge-graph decay (Phase 4).
    Each is independently config-gated; the sweeper starts if any is enabled."""
    s = get_settings()
    days = getattr(s, "memory_retention_days", 0)
    decay_days = getattr(s, "memory_graph_decay_days", 0)
    kg_decay_days = getattr(s, "memory_kg_decay_days", 0)
    if days <= 0 and decay_days <= 0 and kg_decay_days <= 0:
        return
    hours = max(1, getattr(s, "memory_retention_sweep_hours", 24))

    async def _loop() -> None:
        from .memory import data_lifecycle, memory_graph
        from .rag import knowledge_graph
        while True:
            if days > 0:
                try:
                    await asyncio.to_thread(data_lifecycle.apply_retention, days)
                except Exception as exc:
                    logger.warning(f"[Lifecycle] retention sweep failed: {exc}")
            if decay_days > 0:
                try:
                    await asyncio.to_thread(memory_graph.decay)
                except Exception as exc:
                    logger.warning(f"[Lifecycle] graph decay failed: {exc}")
            if kg_decay_days > 0:
                try:
                    await asyncio.to_thread(knowledge_graph.decay)
                except Exception as exc:
                    logger.warning(f"[Lifecycle] kg decay failed: {exc}")
            await asyncio.sleep(hours * 3600)

    global _retention_task
    _retention_task = asyncio.get_event_loop().create_task(_loop())
    logger.info(f"[Lifecycle] sweeper started (retention={days}d, "
                f"graph-decay={decay_days}d, kg-decay={kg_decay_days}d, "
                f"every {hours}h)")


async def _stop_memory_lifecycle() -> None:
    global _retention_task
    if _retention_task is not None:
        _retention_task.cancel()
        try:
            await _retention_task
        except Exception:
            pass
        _retention_task = None


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
app.include_router(memory_routes.router)
app.include_router(appearance_routes.router)


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
