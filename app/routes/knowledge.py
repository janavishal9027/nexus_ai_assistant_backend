"""Knowledge Base (RAG) endpoints: KB CRUD, document upload/ingestion, hybrid
search, and grounded streaming chat with citations.

All routes are owner-scoped via the JWT account, mirroring the rest of the app.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks,
)
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..config import get_settings
from ..models.db_models import Account, Conversation, Message, ChatModel
from ..models.rag_models import KnowledgeBase, Document, DocumentChunk, IngestionJob
from ..models.schemas import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeBaseDto, DocumentDto,
    IngestionJobDto, DocumentUploadResponse, KbSearchRequest, KbSearchResponse,
    SourceChunkDto, KbChatRequest, ConversationDto, MessageDto,
)
from ..services.auth import get_current_account
from ..services import request_context
from ..services import rag_ingestion
from ..services.rag_chunking import is_supported, SUPPORTED_EXTS
from ..services.rag_retrieval import retrieve, build_grounded_messages
from ..services.fallback_router import route_stream_chat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["knowledge"])


# ─── Helpers ────────────────────────────────────────────────────────────────

def _owned_kb(db: Session, kb_id: int, account_id: int) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if kb is None or (kb.owner_id is not None and kb.owner_id != account_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _counts(db: Session, kb_id: int) -> tuple[int, int]:
    docs = db.query(func.count(Document.id)).filter(
        Document.knowledge_base_id == kb_id).scalar() or 0
    chunks = db.query(func.count(DocumentChunk.id)).filter(
        DocumentChunk.knowledge_base_id == kb_id).scalar() or 0
    return int(docs), int(chunks)


def _kb_dto(db: Session, kb: KnowledgeBase) -> KnowledgeBaseDto:
    doc_count, chunk_count = _counts(db, kb.id)
    return KnowledgeBaseDto(
        id=kb.id, name=kb.name, description=kb.description,
        embedding_platform=kb.embedding_platform, embedding_model=kb.embedding_model,
        embedding_dim=kb.embedding_dim, document_count=doc_count,
        chunk_count=chunk_count, created_at=kb.created_at, updated_at=kb.updated_at,
    )


def _doc_dto(doc: Document) -> DocumentDto:
    return DocumentDto(
        id=doc.id, knowledge_base_id=doc.knowledge_base_id, filename=doc.filename,
        content_type=doc.content_type, size_bytes=doc.size_bytes, status=doc.status,
        error=doc.error, chunk_count=doc.chunk_count or 0, created_at=doc.created_at,
    )


# ─── Knowledge Base CRUD ────────────────────────────────────────────────────

@router.post("", response_model=KnowledgeBaseDto)
@router.post("/", response_model=KnowledgeBaseDto)
def create_kb(body: KnowledgeBaseCreate, db: Session = Depends(get_db),
              account: Account = Depends(get_current_account)):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    kb = KnowledgeBase(name=name[:255], description=(body.description or None),
                       owner_id=account.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _kb_dto(db, kb)


@router.get("", response_model=list[KnowledgeBaseDto])
@router.get("/", response_model=list[KnowledgeBaseDto])
def list_kbs(db: Session = Depends(get_db),
             account: Account = Depends(get_current_account)):
    kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.owner_id == account.id)
        .order_by(KnowledgeBase.updated_at.desc())
        .all()
    )
    return [_kb_dto(db, kb) for kb in kbs]


@router.get("/{kb_id}", response_model=KnowledgeBaseDto)
def get_kb(kb_id: int, db: Session = Depends(get_db),
           account: Account = Depends(get_current_account)):
    return _kb_dto(db, _owned_kb(db, kb_id, account.id))


@router.patch("/{kb_id}", response_model=KnowledgeBaseDto)
def update_kb(kb_id: int, body: KnowledgeBaseUpdate, db: Session = Depends(get_db),
              account: Account = Depends(get_current_account)):
    kb = _owned_kb(db, kb_id, account.id)
    if body.name is not None and body.name.strip():
        kb.name = body.name.strip()[:255]
    if body.description is not None:
        kb.description = body.description or None
    db.commit()
    db.refresh(kb)
    return _kb_dto(db, kb)


@router.delete("/{kb_id}")
def delete_kb(kb_id: int, db: Session = Depends(get_db),
              account: Account = Depends(get_current_account)):
    kb = _owned_kb(db, kb_id, account.id)
    # Chunks/jobs FK to documents (ON DELETE CASCADE); documents cascade via the
    # KB relationship. Detach any KB chat conversations so they aren't orphaned.
    db.query(Conversation).filter(Conversation.knowledge_base_id == kb_id).update(
        {Conversation.knowledge_base_id: None}, synchronize_session=False)
    db.delete(kb)
    db.commit()
    return {"success": True}


# ─── Documents ──────────────────────────────────────────────────────────────

@router.post("/{kb_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(kb_id: int, background: BackgroundTasks,
                          file: UploadFile = File(...), db: Session = Depends(get_db),
                          account: Account = Depends(get_current_account)):
    kb = _owned_kb(db, kb_id, account.id)
    filename = file.filename or "upload"
    if not is_supported(filename):
        raise HTTPException(
            status_code=415,
            detail=(f"Unsupported file type. Supported: "
                    f"{', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTS))}"),
        )
    content = await file.read()
    settings = get_settings()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(content) > settings.rag_max_upload_bytes:
        mb = settings.rag_max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds the {mb} MB limit")

    doc = Document(
        knowledge_base_id=kb.id, owner_id=account.id, filename=filename[:512],
        content_type=file.content_type, size_bytes=len(content),
        status="pending",
    )
    # Store bytes via the object-store seam (DB BYTEA by default, or S3/MinIO).
    from ..services.object_store import store_document_bytes
    store_document_bytes(doc, content)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    job = rag_ingestion.create_job(db, doc)
    # Dispatch ingestion: publish a Kafka event when event-driven indexing is on
    # and the broker is reachable, else run in-process via BackgroundTasks. Owner
    # id is passed explicitly (contextvars don't propagate into background tasks).
    await rag_ingestion.dispatch_ingestion(background, doc.id, account.id)

    return DocumentUploadResponse(document=_doc_dto(doc), job_id=job.id)


@router.get("/{kb_id}/documents", response_model=list[DocumentDto])
def list_documents(kb_id: int, db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    docs = (
        db.query(Document)
        .filter(Document.knowledge_base_id == kb_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [_doc_dto(d) for d in docs]


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentDto)
def get_document(kb_id: int, doc_id: int, db: Session = Depends(get_db),
                 account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.knowledge_base_id == kb_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_dto(doc)


@router.get("/{kb_id}/documents/{doc_id}/job", response_model=IngestionJobDto)
def get_document_job(kb_id: int, doc_id: int, db: Session = Depends(get_db),
                     account: Account = Depends(get_current_account)):
    """Latest ingestion job for a document — polled by the client for progress."""
    _owned_kb(db, kb_id, account.id)
    job = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == doc_id,
                IngestionJob.knowledge_base_id == kb_id)
        .order_by(IngestionJob.id.desc())
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="No ingestion job found")
    return IngestionJobDto.model_validate(job)


@router.post("/{kb_id}/documents/{doc_id}/reingest", response_model=DocumentUploadResponse)
async def reingest_document(kb_id: int, doc_id: int, background: BackgroundTasks,
                            db: Session = Depends(get_db),
                            account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.knowledge_base_id == kb_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not (doc.raw or doc.storage_key):
        raise HTTPException(status_code=400, detail="Original file is no longer available")
    doc.status = "pending"
    doc.error = None
    db.commit()
    job = rag_ingestion.create_job(db, doc)
    await rag_ingestion.dispatch_ingestion(background, doc.id, account.id)
    return DocumentUploadResponse(document=_doc_dto(doc), job_id=job.id)


@router.delete("/{kb_id}/documents/{doc_id}")
def delete_document(kb_id: int, doc_id: int, db: Session = Depends(get_db),
                    account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.knowledge_base_id == kb_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    from ..services.object_store import delete_document_bytes
    delete_document_bytes(doc)   # remove the S3 object (no-op in db mode)
    db.delete(doc)  # chunks + jobs cascade via FK ON DELETE CASCADE
    db.commit()
    return {"success": True}


# ─── Search (retrieval preview) ─────────────────────────────────────────────

@router.post("/{kb_id}/search", response_model=KbSearchResponse)
async def search_kb(kb_id: int, body: KbSearchRequest, db: Session = Depends(get_db),
                    account: Account = Depends(get_current_account)):
    kb = _owned_kb(db, kb_id, account.id)
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    request_context.set_owner_id(account.id)
    sources = await retrieve(db, kb, query, account.id)
    if body.top_k:
        sources = sources[:body.top_k]
    return KbSearchResponse(query=query, sources=[SourceChunkDto(**s) for s in sources])


# ─── Grounded streaming chat ────────────────────────────────────────────────

def _sources_footer(sources: list[dict]) -> str:
    if not sources:
        return ""
    seen, lines = set(), []
    for s in sources:
        name = s["document_name"]
        if name in seen:
            continue
        seen.add(name)
        lines.append(f"- {name}")
    return "\n\n---\n**Sources**\n" + "\n".join(lines)


@router.post("/{kb_id}/chat/stream")
async def kb_chat_stream(kb_id: int, body: KbChatRequest, db: Session = Depends(get_db),
                         account: Account = Depends(get_current_account)):
    kb = _owned_kb(db, kb_id, account.id)
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    request_context.set_owner_id(account.id)

    # Retrieve grounding context up front (so failures surface before streaming).
    try:
        sources = await retrieve(db, kb, query, account.id, history=body.history)
    except Exception as e:
        logger.error(f"[KB/Chat] Retrieval failed: {e}", exc_info=True)
        return JSONResponse(status_code=503, content={"error": f"Retrieval failed: {e}"})

    messages = build_grounded_messages(query, sources, body.history)

    # Get or create the KB-scoped conversation.
    if body.conversation_id is not None:
        conv = db.query(Conversation).filter(
            Conversation.id == body.conversation_id,
            Conversation.owner_id == account.id).first()
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = Conversation(title=query[:60], owner_id=account.id,
                            knowledge_base_id=kb.id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
    conversation_id = conv.id

    db.add(Message(conversation_id=conversation_id, role="user", content=query))
    db.commit()

    try:
        result = await route_stream_chat(db, messages, requested_model=body.model)
    except Exception as e:
        logger.error(f"[KB/Chat] Routing failed: {e}", exc_info=True)
        return JSONResponse(status_code=503, content={"error": str(e)})

    source_payload = [SourceChunkDto(**s).model_dump() for s in sources]

    async def event_generator():
        full = ""
        try:
            # Emit citations first so the UI can render the source panel.
            yield json.dumps({"sources": source_payload,
                              "conversationId": conversation_id, "done": False})
            async for chunk in result.stream:
                full += chunk
                yield json.dumps({
                    "content": chunk, "model": result.display_name,
                    "platform": result.platform, "done": False,
                })

            stored = full + _sources_footer(sources)
            db.add(Message(
                conversation_id=conversation_id, role="assistant", content=stored,
                model_used=result.model_id, platform_used=result.platform,
            ))
            conv_row = db.query(Conversation).filter(
                Conversation.id == conversation_id).first()
            if conv_row:
                conv_row.updated_at = datetime.now(timezone.utc)
            db.commit()

            yield json.dumps({
                "content": "", "model": result.display_name, "platform": result.platform,
                "conversationId": conversation_id, "done": True,
            })
        except asyncio.CancelledError:
            logger.warning("[KB/Chat] Client disconnected mid-stream")
            return
        except Exception as e:
            logger.error(f"[KB/Chat] Stream error: {e}", exc_info=True)
            yield json.dumps({"error": str(e), "done": True})

    return EventSourceResponse(event_generator())


# ─── KB conversations (grounded chat history) ───────────────────────────────

@router.get("/{kb_id}/conversations", response_model=list[ConversationDto])
def list_kb_conversations(kb_id: int, db: Session = Depends(get_db),
                          account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    convs = (
        db.query(Conversation)
        .filter(Conversation.owner_id == account.id,
                Conversation.knowledge_base_id == kb_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [ConversationDto(id=c.id, title=c.title, created_at=c.created_at,
                            updated_at=c.updated_at) for c in convs]


@router.get("/{kb_id}/conversations/{conversation_id}", response_model=ConversationDto)
def get_kb_conversation(kb_id: int, conversation_id: int, db: Session = Depends(get_db),
                        account: Account = Depends(get_current_account)):
    _owned_kb(db, kb_id, account.id)
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id, Conversation.owner_id == account.id,
        Conversation.knowledge_base_id == kb_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = (
        db.query(Message).filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc()).all()
    )
    name_by_id = {mid: dn for mid, dn in
                  db.query(ChatModel.model_id, ChatModel.display_name).all()}
    return ConversationDto(
        id=conv.id, title=conv.title, created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[MessageDto(role=m.role, content=m.content,
                             model=(name_by_id.get(m.model_used, m.model_used)
                                    if m.model_used else None),
                             platform=m.platform_used) for m in messages],
    )
