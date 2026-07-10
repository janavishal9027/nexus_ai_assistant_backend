"""Agent Gateway routes — /api/agent/* (req 1, 2, 14, 16).

- POST /api/agent/chat   : authenticated HTTP agent chat (60 s timeout, 503/504)
- WS   /api/agent/ws/{id}: real-time streaming chat
- GET  /api/agent/health : component health (always HTTP 200; req 16.3)
- GET  /api/agent/metrics: Prometheus text (req 16.6)

Auth here is a lightweight bearer/API-key presence check: it enforces the
observable contract (401 without a token, per-token rate limiting) and is the
seam where real session/JWT validation should be plugged in.
"""
import asyncio
import logging
import time
import uuid
from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models.schemas import AgentChatRequest
from ..services.ws_manager import ws_manager
from ..services.observability import observability
from ..services.auth import account_from_token, extract_bearer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

REQUEST_TIMEOUT_S = 60.0
RATE_LIMIT_PER_MIN = 60          # req 14.3
HEALTH_CHECK_TIMEOUT_S = 2.0     # req 16.7

# Simple in-memory sliding-window rate limiter keyed by auth token.
_rate_windows: dict[str, deque] = {}


# ─── Auth / rate limiting helpers ───────────────────────────────────────────
def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.headers.get("X-API-Key") or request.query_params.get("token")


def _require_auth(request: Request) -> str:
    token = _extract_token(request)
    if not token:
        client = request.client.host if request.client else "unknown"
        logger.warning(f"[AgentGateway] Auth failure from {client} on {request.url.path}")
        raise HTTPException(status_code=401, detail="Authentication required")
    return token


def _check_rate_limit(token: str) -> None:
    now = time.time()
    window = _rate_windows.setdefault(token, deque())
    while window and window[0] < now - 60.0:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        retry_after = int(max(1, window[0] + 60.0 - now))
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)


def _validate_session_token(token: str | None) -> bool:
    return bool(token)


# ─── HTTP chat endpoint ─────────────────────────────────────────────────────
@router.post("/chat")
async def agent_chat_endpoint(request: Request, body: AgentChatRequest, db: Session = Depends(get_db)):
    token = extract_bearer(request)
    account = account_from_token(token, db)      # verify JWT (req 14.2)
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    _check_rate_limit(token)                     # 429 (req 14.4)
    correlation_id = str(uuid.uuid4())           # req 1.8

    from ..services.agent import agent_chat_orchestrated
    try:
        result = await asyncio.wait_for(
            agent_chat_orchestrated(db, body, correlation_id, owner_id=account.id),
            timeout=REQUEST_TIMEOUT_S,
        )
        return JSONResponse(content=result, headers={"X-Correlation-ID": correlation_id})
    except asyncio.TimeoutError:
        logger.warning(f"[AgentGateway] Request timed out correlation_id={correlation_id}")
        return JSONResponse(status_code=504, content={"error": "Request timed out"},
                            headers={"X-Correlation-ID": correlation_id})
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[AgentGateway] Orchestrator error correlation_id={correlation_id}: {exc}", exc_info=True)
        return JSONResponse(status_code=503, content={"error": "Service temporarily unavailable"},
                            headers={"X-Correlation-ID": correlation_id})


# ─── WebSocket endpoint ─────────────────────────────────────────────────────
@router.websocket("/ws/{session_id}")
async def agent_ws_endpoint(websocket: WebSocket, session_id: str):
    token = websocket.query_params.get("token") \
        or websocket.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    # Verify the JWT and resolve the owning account before completing the handshake.
    _db0 = SessionLocal()
    try:
        account = account_from_token(token, _db0)
    finally:
        _db0.close()
    if account is None:
        await websocket.close(code=4001)          # req 14.6
        return
    owner_id = account.id
    await websocket.accept()
    try:
        await ws_manager.register(session_id, websocket)   # req 1.4
    except RuntimeError:
        await websocket.close(code=4003)          # capacity (req 2.7)
        return
    try:
        while True:
            data = await websocket.receive_json()
            ws_manager.touch(session_id)
            if isinstance(data, dict) and data.get("type") == "chat":
                asyncio.create_task(_handle_ws_chat(session_id, data, owner_id))
            # {"type": "pong"} and others just refresh activity via touch().
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"[AgentGateway] WS error for {session_id}: {exc}")
    finally:
        await ws_manager.unregister(session_id)   # req 1.5


async def _handle_ws_chat(session_id: str, data: dict, owner_id: int | None = None) -> None:
    """Run the orchestration pipeline over the WebSocket, emitting typed events:
    plan_created → tool_start/tool_end → token(s) → done (req 2, 3.10)."""
    from datetime import datetime, timezone
    from ..services.agent import agent_stream_chat, _publish_event
    from ..services.fallback_router import DeepResearchUnavailableError
    from ..services import request_context
    from ..services.feature_flags import get_agent_features
    from ..services.memory_manager import memory_manager
    from ..models.schemas import ChatRequest
    from ..models.db_models import Message, Conversation

    correlation_id = str(uuid.uuid4())
    message = str(data.get("message", ""))
    conversation_id_in = data.get("conversation_id")
    db = SessionLocal()
    try:
        request_context.set_request_context(correlation_id=correlation_id, session_id=session_id, owner_id=owner_id)
        await _publish_event("request_received", correlation_id, conversation_id_in, session_id,
                             {"message_preview": message[:100]})

        # Planner (feature-flagged) → plan_created event (req 3.10)
        flags = get_agent_features()
        if flags.planner:
            from ..services.planner import PlannerAgent
            from ..services.tool_registry import tool_registry
            plan = await PlannerAgent(tool_registry).classify_and_plan(db, message, correlation_id)
            if plan and plan.subtasks:
                summary = [{"index": s.index, "description": s.description} for s in plan.subtasks]
                await ws_manager.send(session_id, {"type": "plan_created",
                                                   "subtask_count": len(summary), "subtasks": summary})
                await _publish_event("plan_created", correlation_id, conversation_id_in, session_id,
                                     {"subtask_count": len(summary), "subtasks": summary})

        async def on_tool_event(ev: dict) -> None:
            await ws_manager.send(session_id, ev)

        req = ChatRequest(
            message=message,
            conversation_id=conversation_id_in,
            model=data.get("model"),
            deep_research=bool(data.get("deep_research", False)),
            web_search=bool(data.get("web_search", False)),
        )
        conversation_id, stream_result, citations = await agent_stream_chat(
            db, req, on_tool_event=on_tool_event, owner_id=owner_id)

        full = ""
        async for chunk in stream_result.stream:
            full += chunk
            await ws_manager.send(session_id, {"type": "token", "content": chunk})
        if citations:
            full += f"\n\n{citations}"
            await ws_manager.send(session_id, {"type": "token", "content": f"\n\n{citations}"})

        # Persist the assistant message (agent_stream_chat persists only the user turn).
        db.add(Message(conversation_id=conversation_id, role="assistant", content=full,
                       model_used=stream_result.model_id, platform_used=stream_result.platform))
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        await memory_manager.auto_store(conversation_id, message, full)
        await _publish_event("response_generated", correlation_id, conversation_id, session_id,
                             {"model": stream_result.display_name, "platform": stream_result.platform})

        await ws_manager.send(session_id, {
            "type": "done", "conversation_id": conversation_id,
            "model": stream_result.display_name, "platform": stream_result.platform,
        })
    except DeepResearchUnavailableError as exc:
        # Actionable message (which model to add) instead of the generic error.
        logger.warning(f"[AgentGateway] Deep Research unavailable for {session_id}: {exc}")
        await ws_manager.send_error_and_close(session_id, str(exc))
    except Exception as exc:
        logger.error(f"[AgentGateway] WS chat failed for {session_id}: {exc}", exc_info=True)
        await ws_manager.send_error_and_close(session_id, "Generation failed")  # req 2.6
    finally:
        db.close()


# ─── Health & metrics ───────────────────────────────────────────────────────
async def _check_redis() -> dict:
    from ..services.redis_cache import get_redis_cache
    cache = get_redis_cache()
    if cache is None:
        return {"status": "disabled", "reason": "redis_cache feature not enabled"}
    return {"status": "healthy"} if await cache.ping() else {"status": "degraded", "reason": "ping failed"}


async def _check_kafka() -> dict:
    from ..services import kafka_producer as _kp
    if _kp.kafka_producer is None:
        return {"status": "disabled", "reason": "kafka feature not enabled"}
    return {"status": "healthy"} if await _kp.kafka_producer.ping() else {"status": "degraded", "reason": "broker unreachable"}


async def _check_pgvector() -> dict:
    from ..models.db_models import HAS_PGVECTOR
    from ..services.tool_registry import tool_registry
    if tool_registry.get("memory_search") is None:
        return {"status": "disabled", "reason": "memory tools not registered"}
    return {"status": "healthy", "reason": "pgvector" if HAS_PGVECTOR else "json-fallback"}


async def _check_ws() -> dict:
    return {"status": "healthy", "reason": f"{ws_manager.count()} active sessions"}


@router.get("/health")
async def health_endpoint():
    checks = {
        "agent_gateway": lambda: _ok(),
        "redis_cache": _check_redis,
        "kafka_producer": _check_kafka,
        "memory_pgvector": _check_pgvector,
        "websocket_manager": _check_ws,
    }
    components: dict = {}
    overall = "healthy"
    for name, fn in checks.items():
        try:
            status = await asyncio.wait_for(fn(), timeout=HEALTH_CHECK_TIMEOUT_S)
        except asyncio.TimeoutError:
            status = {"status": "timeout", "reason": "health check timed out"}
        except Exception as exc:
            status = {"status": "degraded", "reason": str(exc)}
        if status.get("status") in ("degraded", "timeout"):
            overall = "degraded"
        components[name] = status
    # Always HTTP 200, even when degraded (req 16.3 / Q12).
    return JSONResponse(status_code=200, content={"status": overall, "components": components})


async def _ok() -> dict:
    return {"status": "healthy"}


@router.get("/metrics")
async def metrics_endpoint():
    return PlainTextResponse(observability.to_prometheus_text(),
                             media_type="text/plain; version=0.0.4")
