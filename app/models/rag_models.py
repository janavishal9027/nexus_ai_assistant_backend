"""RAG (Retrieval-Augmented Generation) data model.

Four additive tables — all owner-scoped like the rest of the app:

  knowledge_bases   a named collection a user chats against
  documents         an uploaded file (original bytes + ingestion status)
  document_chunks   cleaned text slices + their embedding (pgvector)
  ingestion_jobs    progress/status of the parse→chunk→embed pipeline

Embeddings use an UNBOUNDED pgvector column (``vector`` with no fixed
dimension) so the embedding model can be auto-detected at first ingest without
a schema migration; the dimension a KB settled on is recorded on the KB row so
all of its chunks stay comparable. When pgvector is unavailable the column
degrades to JSON and cosine similarity is computed in Python (see
``services/rag_retrieval.py``).
"""
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON,
    LargeBinary,
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from ..database import Base

# Reuse the same pgvector detection the semantic-memory model uses so the whole
# app agrees on whether real vector columns are available.
try:
    from pgvector.sqlalchemy import Vector as _Vector
    HAS_PGVECTOR = True

    def _vector_column():
        # No dimension → an unbounded ``vector`` column that accepts whatever
        # the auto-detected embedding model produces (mistral 1024, gemini 768,
        # openai 1536, …). Cannot carry an ANN index, but exact ``<=>`` search
        # is fast at personal-KB scale.
        return Column(_Vector(), nullable=True)
except Exception:  # pragma: no cover - only when pgvector is absent
    HAS_PGVECTOR = False

    def _vector_column():
        return Column(JSON, nullable=True)


def _now():
    return datetime.now(timezone.utc)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, index=True, nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # The embedding model this KB committed to at first successful ingest. Every
    # chunk in the KB is embedded with the same model so their vectors live in
    # one space; retrieval only ever compares within a single KB.
    embedding_platform = Column(String(64), nullable=True)
    embedding_model = Column(String(128), nullable=True)
    embedding_dim = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    documents = relationship(
        "Document", back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    owner_id = Column(Integer, index=True, nullable=True)

    filename = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=True)
    size_bytes = Column(Integer, nullable=True)

    # pending → processing → completed | failed
    status = Column(String(32), default="pending", index=True)
    error = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)

    # Original file bytes, kept so the source can be re-viewed / re-ingested.
    raw = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Denormalized so semantic/keyword search can filter by KB in one predicate.
    knowledge_base_id = Column(Integer, nullable=False, index=True)
    owner_id = Column(Integer, index=True, nullable=True)

    ordinal = Column(Integer, nullable=False)   # position within the document
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding = _vector_column()

    created_at = Column(DateTime, default=_now)

    document = relationship("Document", back_populates="chunks")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    knowledge_base_id = Column(Integer, nullable=False, index=True)
    owner_id = Column(Integer, index=True, nullable=True)

    # pending → parsing → chunking → embedding → completed | failed
    status = Column(String(32), default="pending", index=True)
    stage = Column(String(64), nullable=True)          # human-readable label
    progress = Column(Integer, default=0)              # 0–100
    total_chunks = Column(Integer, default=0)
    embedded_chunks = Column(Integer, default=0)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
