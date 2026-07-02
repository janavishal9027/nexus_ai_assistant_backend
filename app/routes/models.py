from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import ChatModel

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/")
def get_all(db: Session = Depends(get_db)):
    models = db.query(ChatModel).order_by(ChatModel.priority.asc()).all()
    return [_to_dict(m) for m in models]


@router.get("/enabled")
def get_enabled(db: Session = Depends(get_db)):
    models = (
        db.query(ChatModel)
        .filter(ChatModel.enabled == True)
        .order_by(ChatModel.priority.asc())
        .all()
    )
    return [_to_dict(m) for m in models]


@router.get("/platform/{platform}")
def get_by_platform(platform: str, db: Session = Depends(get_db)):
    models = (
        db.query(ChatModel)
        .filter(ChatModel.platform == platform, ChatModel.enabled == True)
        .order_by(ChatModel.priority.asc())
        .all()
    )
    return [_to_dict(m) for m in models]


def _to_dict(m: ChatModel) -> dict:
    return {
        "id": m.id,
        "platform": m.platform,
        "modelId": m.model_id,
        "displayName": m.display_name,
        "sizeLabel": m.size_label,
        "intelligenceRank": m.intelligence_rank,
        "speedRank": m.speed_rank,
        "contextWindow": m.context_window,
        "enabled": m.enabled,
        "supportsVision": m.supports_vision,
        "supportsTools": m.supports_tools,
        "priority": m.priority,
    }
