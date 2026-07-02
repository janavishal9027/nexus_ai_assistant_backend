import logging
from sqlalchemy.orm import Session
from ..models.db_models import ChatModel

logger = logging.getLogger(__name__)


def seed_models(db: Session):
    """Seed the model catalog from providers_config.json."""
    from .agent import get_config

    config = get_config()
    providers = config.get("providers", [])

    priority = 1
    added = 0
    for provider in providers:
        platform = provider["id"]
        for model in provider.get("models", []):
            existing = db.query(ChatModel).filter(
                ChatModel.platform == platform,
                ChatModel.model_id == model["id"],
            ).first()
            if not existing:
                db.add(ChatModel(
                    platform=platform,
                    model_id=model["id"],
                    display_name=model["name"],
                    size_label=model.get("tier", "Medium"),
                    intelligence_rank=priority,
                    speed_rank=priority,
                    context_window=model.get("context"),
                    supports_vision=model.get("vision", False),
                    supports_tools=model.get("tools", True),
                    enabled=True,
                    priority=priority,
                ))
                added += 1
            priority += 1

    if added > 0:
        db.commit()
        logger.info(f"Seeded {added} new models from config")

    total = db.query(ChatModel).count()
    logger.info(f"Model catalog: {total} models across {len(providers)} providers")
