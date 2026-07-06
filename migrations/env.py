"""Alembic environment.

Resolves the database URL from the application's Settings and targets the
SQLAlchemy metadata so `alembic upgrade head` applies the additive agent tables.
Run from the backend/ directory.
"""
from alembic import context

from app.database import Base, engine
import app.models.db_models  # noqa: F401  (ensure all models are imported)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=str(engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
