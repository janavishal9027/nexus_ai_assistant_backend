"""add agent orchestration tables

Additive migration for the full-stack-agent-orchestration feature (req 17.5).
Enables the pgvector extension, creates the new tables, and builds an IVFFlat
cosine index on memory_chunks.embedding. Existing tables are not modified.

Revision ID: 001_add_agent_tables
Revises:
Create Date: 2026-07-03
"""
from alembic import op
import sqlalchemy as sa

revision = "001_add_agent_tables"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    # 1. pgvector extension (req 8.1)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    from pgvector.sqlalchemy import Vector

    # 2. users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("role", sa.String(64), server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 3. tasks
    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", sa.String(64), server_default="pending"),
        sa.Column("assignee_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("priority", sa.String(32), server_default="medium"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_tasks_status", "tasks", ["status"])
    op.create_index("ix_tasks_assignee_id", "tasks", ["assignee_id"])

    # 4. memory_chunks
    op.create_table(
        "memory_chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("conversation_id", sa.Integer, sa.ForeignKey("conversations.id"), nullable=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_memory_chunks_conversation_id", "memory_chunks", ["conversation_id"])
    op.create_index("ix_memory_chunks_user_id", "memory_chunks", ["user_id"])

    # 5. IVFFlat cosine index (req 8.1)
    op.execute(
        "CREATE INDEX memory_chunks_embedding_idx ON memory_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # 6. fcm_tokens
    op.create_table(
        "fcm_tokens",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("device_token", sa.String(512), nullable=False, unique=True),
        sa.Column("device_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_fcm_tokens_user_id", "fcm_tokens", ["user_id"])

    # 7. audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("correlation_id", sa.String(36), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=False),
        sa.Column("acting_user_id", sa.Integer, nullable=True),
        sa.Column("target_resource", sa.String(255), nullable=True),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_logs_correlation_id", "audit_logs", ["correlation_id"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("fcm_tokens")
    op.execute("DROP INDEX IF EXISTS memory_chunks_embedding_idx")
    op.drop_table("memory_chunks")
    op.drop_table("tasks")
    op.drop_table("users")
    # The vector extension is left installed; other objects may depend on it.
