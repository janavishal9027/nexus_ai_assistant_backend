"""Project endpoints (chat-module A.7): group conversations, give them shared
standing instructions, and delete with a keep-or-remove choice."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Project, Conversation, Account
from ..models.schemas import ProjectCreate, ProjectUpdate, ProjectDto
from ..services.auth import get_current_account

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _owned(db: Session, project_id: int, account_id: int) -> Project:
    p = db.query(Project).filter(Project.id == project_id).first()
    if p is None or (p.owner_id is not None and p.owner_id != account_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return p


def _count(db: Session, project_id: int) -> int:
    return int(db.query(func.count(Conversation.id))
               .filter(Conversation.project_id == project_id).scalar() or 0)


def _dto(db: Session, p: Project) -> ProjectDto:
    return ProjectDto(
        id=p.id, name=p.name, instructions=p.instructions,
        conversation_count=_count(db, p.id),
        created_at=p.created_at, updated_at=p.updated_at,
    )


@router.post("", response_model=ProjectDto)
@router.post("/", response_model=ProjectDto)
def create_project(body: ProjectCreate, db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    p = Project(name=name[:255], instructions=(body.instructions or None),
                owner_id=account.id)
    db.add(p)
    db.commit()
    db.refresh(p)
    return _dto(db, p)


@router.get("", response_model=list[ProjectDto])
@router.get("/", response_model=list[ProjectDto])
def list_projects(db: Session = Depends(get_db),
                  account: Account = Depends(get_current_account)):
    projects = (db.query(Project).filter(Project.owner_id == account.id)
                .order_by(Project.created_at.asc()).all())
    return [_dto(db, p) for p in projects]


@router.patch("/{project_id}", response_model=ProjectDto)
def update_project(project_id: int, body: ProjectUpdate, db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    p = _owned(db, project_id, account.id)
    if body.name is not None and body.name.strip():
        p.name = body.name.strip()[:255]
    if body.instructions is not None:
        p.instructions = body.instructions or None
    db.commit()
    db.refresh(p)
    return _dto(db, p)


@router.delete("/{project_id}")
def delete_project(project_id: int,
                   delete_conversations: bool = Query(False),
                   db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    """Delete a project. With delete_conversations=false (default) its chats are
    moved to ungrouped; with true they're deleted too."""
    _owned(db, project_id, account.id)
    convo_ids = [cid for (cid,) in db.query(Conversation.id)
                 .filter(Conversation.project_id == project_id).all()]

    if delete_conversations and convo_ids:
        # Clean agent-feature rows that FK to these conversations first.
        try:
            from ..models.db_models import MemoryChunk
            db.query(MemoryChunk).filter(
                MemoryChunk.conversation_id.in_(convo_ids)
            ).delete(synchronize_session=False)
        except Exception:
            db.rollback()
        for c in db.query(Conversation).filter(Conversation.id.in_(convo_ids)).all():
            db.delete(c)  # messages cascade via the ORM relationship
    else:
        db.query(Conversation).filter(Conversation.project_id == project_id).update(
            {Conversation.project_id: None}, synchronize_session=False)

    db.query(Project).filter(Project.id == project_id).delete()
    db.commit()
    return {"success": True, "deleted_conversations": delete_conversations,
            "affected": convo_ids}
