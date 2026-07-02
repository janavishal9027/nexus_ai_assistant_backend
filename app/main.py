import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, SessionLocal, Base
from .routes import chat, conversations, keys, models, config
from .services.model_seeder import seed_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, seed models, sync from providers
    logger.info("Starting ChatApp backend...")
    Base.metadata.create_all(bind=engine)

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

    logger.info("ChatApp backend ready on port 8080")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="ChatApp Backend",
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


@app.get("/api/ping")
def ping():
    return {"status": "ok"}
