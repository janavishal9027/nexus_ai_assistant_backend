from pydantic_settings import BaseSettings
from functools import lru_cache
from urllib.parse import quote_plus


class Settings(BaseSettings):
    # PostgreSQL config
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "chatapp"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Provider API keys (can also be added dynamically via API)
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    nvidia_api_key: str = ""
    huggingface_api_key: str = ""
    google_api_key: str = ""

    # Real-time web search. Optional: with no key, a keyless DuckDuckGo
    # fallback is used. A free Tavily key gives cleaner, LLM-optimized results.
    tavily_api_key: str = ""

    # Fallback config
    fallback_max_retries: int = 10
    cooldown_duration_ms: int = 90000

    # ─── Agent Orchestration Infrastructure (full-stack-agent-orchestration) ───
    # Redis (live session state + fast current-state cache). Req 15.1
    redis_url: str = "redis://localhost:6379"
    # Kafka (agent event streaming + inter-service commands). Req 15.2
    kafka_bootstrap_servers: str = "localhost:9092"
    # Firebase Cloud Messaging service-account credentials path. Req 15.4
    fcm_credentials_path: str = ""
    # Embedding model for pgvector semantic memory. Req 15.5
    embedding_model: str = "text-embedding-3-small"
    # Cosine similarity cut-off for memory_search. Req 8.9 / 15
    memory_similarity_threshold: float = 0.7
    # ── Part D Phase 2: semantic memory (skills/prefs/lessons) + Reflector ────
    memory_semantic_recall_enabled: bool = True   # inject "About the user" at turn start
    memory_skill_recall_top_k: int = 3
    memory_reflect_enabled: bool = True           # distil episodes+feedback → skills
    memory_reflect_every_turns: int = 6           # reflect once per N stored turns/conversation
    memory_skill_dedup_threshold: float = 0.88    # merge skills more similar than this
    # ── Part D Phase 3: lifecycle (retention / export / purge) ────────────────
    # Episodic memories older than this are purged by the background sweep. 0 =
    # keep forever. Semantic skills (distilled + durable) never expire.
    memory_retention_days: int = 0
    memory_retention_sweep_hours: int = 24
    # ── Part D Phase 4: project brain + content knowledge graph ───────────────
    memory_project_brain_enabled: bool = True     # per-project auto-brain + ledger
    memory_project_reflect_every_turns: int = 8
    memory_brain_dedup_threshold: float = 0.9
    memory_kg_enabled: bool = True                # entity/relation extraction
    memory_kg_every_turns: int = 4
    # Bounded per-topic ring-buffer size for the Real-Time Events tool. Req 18.4
    realtime_event_buffer_size: int = 500
    # Comma-separated agent feature flags: planner,redis_cache,kafka,fcm,websocket. Req 15.7
    agent_features: str = ""

    # ─── Authentication ─────────────────────────────────────────────────────
    # HS256 signing secret for JWTs. MUST be overridden in production via env.
    jwt_secret: str = "dev-insecure-jwt-secret-change-me"
    # Token lifetime in hours (default 30 days).
    jwt_expires_hours: int = 720

    # ─── RAG / Knowledge Base pipeline ──────────────────────────────────────
    # Chunking: target size and overlap in characters (~4 chars per token, so
    # ~1200 chars ≈ 300 tokens). Overlap preserves context across chunk edges.
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    # Max upload size per document (bytes). Default 25 MB.
    rag_max_upload_bytes: int = 25 * 1024 * 1024
    # Hybrid retrieval fan-out and fusion (per the recommended sequence):
    #   semantic top-N  +  keyword top-N  →  RRF  →  top-K  →  rerank → final
    rag_semantic_top_n: int = 20
    rag_keyword_top_n: int = 20
    rag_fusion_top_k: int = 10
    rag_final_top_k: int = 6
    # Reciprocal Rank Fusion damping constant (higher = flatter weighting).
    rag_rrf_k: int = 60
    # Preferred embedding platforms, best first. The first one the user holds a
    # key for is auto-selected; "hash" is a keyless local fallback (dev only).
    rag_embedding_preference: str = "mistral,openai,vercel,nvidia,google,hash"

    # ── Semantic-embedding enhancements (see docs/semantic-embedding/) ────────
    # Parent/child chunking: search the small child for precision, feed the
    # larger parent (its whole section) to the LLM for context. 0 disables.
    rag_child_chunk_size: int = 1200      # chars (~300 tok) — the searchable unit
    rag_parent_chunk_size: int = 4000     # chars (~1000 tok) — returned to the LLM
    rag_structure_aware: bool = True      # detect markdown headings → section paths
    # Retrieve more candidates than the final answer needs so the reranker has
    # something to work with (retrieve N → rerank → keep rag_final_top_k).
    rag_candidate_top_k: int = 30
    # Reranking. Preference order of rerank backends; the first available wins.
    #   cohere/jina/voyage → provider cross-encoder (needs that key)
    #   llm                → rerank with the user's own chat model (keyless)
    #   heuristic          → cheap lexical+score reranker (keyless, always works)
    rag_rerank_enabled: bool = True
    rag_rerank_preference: str = "cohere,jina,voyage,llm,heuristic"
    # Query understanding: rewrite follow-ups into standalone queries using the
    # conversation history (improves the query embedding). Uses the user's model.
    rag_query_rewrite: bool = True
    # Embedding service hardening.
    rag_embed_max_retries: int = 3        # per batch, exponential backoff
    rag_embed_cache_size: int = 4096      # in-process LRU of (model,text)→vector
    rag_embed_normalize: bool = False     # L2-normalize vectors (cosine-invariant)
    # Skip re-embedding when an uploaded file's content hash is unchanged.
    rag_dedup_by_hash: bool = True

    # ── Phase 3: scale (docs/semantic-embedding/11-implementation-roadmap) ─────
    # HNSW ANN index for the primary embedding dimension. Chunks whose vector
    # matches RAG_ANN_DIM (env, default 1024 = mistral-embed) are mirrored into an
    # indexed fixed-dim column for fast approximate search; other dims fall back
    # to the exact <=> scan. Robust: any ANN error falls back to exact.
    rag_ann_enabled: bool = True
    # Semantic retrieval cache: reuse the retrieved chunk-set for a repeated /
    # semantically-equivalent query within a scope (KB or conversation). Only the
    # chunk SELECTION is cached — the LLM still re-grounds a fresh answer — and it
    # is invalidated whenever a document is (re)ingested into that scope.
    rag_semantic_cache_enabled: bool = True
    rag_semantic_cache_threshold: float = 0.97   # cosine sim to count as a hit
    rag_semantic_cache_size: int = 64            # entries per scope
    rag_semantic_cache_ttl_s: int = 900          # 15 minutes

    # ── Phase 4: event-driven scale-out platform (opt-in; default off) ────────
    # Object storage for uploaded file bytes: "db" (BYTEA in documents.raw,
    # default) or "s3" (MinIO / S3 — needs boto3 + the rag_s3_* settings). The
    # in-DB path needs no extra infra; S3 offloads large blobs out of Postgres.
    rag_object_store: str = "db"
    rag_s3_endpoint: str = ""            # e.g. http://localhost:9000 (MinIO)
    rag_s3_bucket: str = "nexus-documents"
    rag_s3_access_key: str = ""
    rag_s3_secret_key: str = ""
    rag_s3_region: str = "us-east-1"
    # Kafka event-driven indexing: on upload, publish a `nexus.document.uploaded`
    # event and let a consumer worker run ingestion (instead of in-process
    # BackgroundTasks). Needs a reachable broker; falls back to BackgroundTasks
    # whenever off or the producer is unavailable — so uploads never fail.
    rag_kafka_indexing: bool = False
    rag_kafka_bootstrap: str = ""        # blank → reuse kafka_bootstrap_servers
    rag_kafka_index_retries: int = 3     # per-document retries before the DLQ
    # OpenTelemetry export of RagTrace spans (needs opentelemetry-sdk + an OTLP
    # collector). Off by default; the structured log line is always emitted.
    rag_otel_enabled: bool = False
    rag_otel_endpoint: str = "http://localhost:4317"

    @property
    def database_url(self) -> str:
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
