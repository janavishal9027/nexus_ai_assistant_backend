"""
Agent Orchestration Layer.

Manages the conversation context, system prompt, and structures the LLM
interaction. The agent:
1. Prepends the system prompt from config
2. Trims conversation history to fit context limits
3. Sends to the fallback router for model selection
4. Returns structured response with metadata
5. Orchestrates multi-step tool invocation with token budget management
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path
from sqlalchemy.orm import Session

from ..models.schemas import MessageDto, ChatRequest
from ..models.db_models import Conversation, Message
from .fallback_router import (
    route_chat, route_stream_chat, route_deep_research_stream,
    RouteResult, StreamRouteResult, DeepResearchUnavailableError,
)
from .web_search import needs_web_search, web_search
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Import tool system components
try:
    from .tool_registry import tool_registry
    from .decision import DecisionEngine
    from .tool_executor import ToolExecutor
    from .token_budget import TokenBudgetManager, TokenBudgetConfig
    from .citations import CitationTracker
    from .tool_models import ToolResult
    _tool_system_available = True
except ImportError as e:
    _tool_system_available = False
    logger.warning(f"[Config] Tool system not available: {e}. Tool orchestration will be disabled.")

# Initialize tool system components at module level
if _tool_system_available:
    decision_engine = DecisionEngine(tool_registry)
    tool_executor = ToolExecutor(tool_registry)
    # token_budget_manager initialized in get_config() based on configuration
    _token_budget_manager = None
else:
    decision_engine = None
    tool_executor = None
    _token_budget_manager = None

# Tool failure tracking for monitoring (requirement 10.8)
_tool_success_counts: dict[str, int] = {}
_tool_failure_counts: dict[str, int] = {}


# Fast pre-gate for the LLM-backed tool-decision loop. Ordinary chat/coding/
# writing prompts never need a tool, so invoking the DecisionEngine on every
# message adds a full LLM round-trip *before* the answer even starts. We only
# enter the loop when the message explicitly signals a tool need. Real-time
# questions are handled separately (and faster) by the web-search heuristic,
# which injects results straight into the prompt — so when that already fired
# (web_context is set) the loop is skipped too.
_TOOL_INTENT_SIGNALS = (
    "search the web", "search online", "web search", "look up", "look it up",
    "find online", "browse the", "on the internet", "google it",
    "my task", "my tasks", "list task", "list tasks", "create a task",
    "create task", "complete task", "mark task", "task list", "assign task",
    "list user", "create user", "find user", "update user", "user record",
    "query the database", "run a query", "sql query", "from the database",
    "remember that", "what did i say", "earlier i said", "recall the",
    "recent events", "live events", "current state of my",
)


def _should_run_tool_loop(message: str, web_context: str | None) -> bool:
    """True only when the message plausibly needs a tool. Skips the per-message
    DecisionEngine LLM call for ordinary prompts — the main time-to-first-token win."""
    if web_context:  # real-time data already injected by the web-search heuristic
        return False
    q = (message or "").lower()
    return any(sig in q for sig in _TOOL_INTENT_SIGNALS)


def get_tool_failure_rates() -> dict[str, dict]:
    """
    Get tool failure rates for monitoring (requirement 10.8, 12.6).
    
    Returns:
        Dictionary mapping tool names to their success/failure counts and rate
    """
    all_tools = set(_tool_success_counts.keys()) | set(_tool_failure_counts.keys())
    rates = {}
    for tool_name in all_tools:
        success = _tool_success_counts.get(tool_name, 0)
        failure = _tool_failure_counts.get(tool_name, 0)
        total = success + failure
        failure_rate = (failure / total * 100) if total > 0 else 0.0
        rates[tool_name] = {
            "success_count": success,
            "failure_count": failure,
            "total_calls": total,
            "failure_rate_percent": round(failure_rate, 2)
        }
    return rates

# Load config once
_config_path = Path(__file__).parent.parent / "providers_config.json"
_config: dict = {}


def _apply_tool_config(tools_config: dict) -> None:
    """
    Apply per-tool configuration from the tools section.
    
    Reads the tools section and applies enable/disable status and per-tool
    settings to the tool_registry.
    
    Args:
        tools_config: The 'tools' section from providers_config.json
    """
    if not _tool_system_available:
        logger.debug("[Config] Skipping tool configuration - tool system not available")
        return
    
    if not tools_config:
        logger.debug("[Config] No tools section in configuration")
        return
    
    logger.info(f"[Config] Applying configuration for {len(tools_config)} tools")
    
    for tool_name, tool_settings in tools_config.items():
        # Check if tool exists in registry
        tool_def = tool_registry.get(tool_name)
        if tool_def is None:
            logger.warning(f"[Config] Tool '{tool_name}' in config but not registered in tool_registry")
            continue
        
        # Apply enabled/disabled status
        enabled = tool_settings.get("enabled", True)
        if enabled:
            tool_registry.enable(tool_name)
            logger.info(f"[Config] Enabled tool: {tool_name}")
        else:
            tool_registry.disable(tool_name)
            logger.info(f"[Config] Disabled tool: {tool_name}")
        
        # Apply per-tool timeout if specified
        if "timeout_seconds" in tool_settings:
            timeout = tool_settings["timeout_seconds"]
            if timeout > 0:
                tool_def.timeout_seconds = float(timeout)
                logger.debug(f"[Config] Set timeout for {tool_name}: {timeout}s")
        
        # Apply per-tool max_results if specified (tool-specific setting)
        if "max_results" in tool_settings:
            # Store as metadata that the tool implementation can access
            if not hasattr(tool_def, 'config'):
                tool_def.config = {}
            tool_def.config['max_results'] = tool_settings["max_results"]
            logger.debug(f"[Config] Set max_results for {tool_name}: {tool_settings['max_results']}")


def get_config() -> dict:
    global _config, _token_budget_manager
    if not _config:
        logger.info(f"[Config] Loading config from {_config_path}")
        
        # Check if config file exists
        if not _config_path.exists():
            logger.error(f"[Config] Configuration file not found: {_config_path}")
            logger.error("[Config] Using default settings")
            _config = {
                "agent": {
                    "system_prompt": "You are a helpful assistant.",
                    "max_context_messages": 20,
                    "default_temperature": 0.7,
                    "default_max_tokens": 4096,
                    "web_search_enabled": True,
                    "tool_calling_enabled": False,
                    "max_tool_rounds": 3,
                    "max_concurrent_tools": 5,
                    "tool_timeout_seconds": 30,
                    "token_budget": 100000
                },
                "tools": {},
                "fallback": {
                    "max_retries": 10,
                    "cooldown_seconds": 90,
                    "escalated_cooldown_seconds": 600
                }
            }
            # Initialize token budget manager with defaults
            if _tool_system_available and _token_budget_manager is None:
                _token_budget_manager = TokenBudgetManager(TokenBudgetConfig(
                    enabled=True,
                    max_tokens=100000,
                    reserve_for_response=4096,
                    truncation_threshold=0.8,
                    chars_per_token=4.0
                ))
            return _config
        
        try:
            _config = json.loads(_config_path.read_text(encoding="utf-8"))
            
            # Validate and provide defaults for agent section
            agent_cfg = _config.get("agent", {})
            
            # Handle token_budget - can be int or dict
            token_budget = agent_cfg.get("token_budget", 100000)
            if not isinstance(token_budget, (int, dict)):
                token_budget = 100000
            
            _config["agent"] = {
                "system_prompt": agent_cfg.get("system_prompt", "You are a helpful assistant."),
                "max_context_messages": agent_cfg.get("max_context_messages", 20),
                "default_temperature": agent_cfg.get("default_temperature", 0.7),
                "default_max_tokens": agent_cfg.get("default_max_tokens", 4096),
                "web_search_enabled": agent_cfg.get("web_search_enabled", True),
                "tool_calling_enabled": agent_cfg.get("tool_calling_enabled", False),
                "max_tool_rounds": agent_cfg.get("max_tool_rounds", 3),
                "max_concurrent_tools": agent_cfg.get("max_concurrent_tools", 5),
                "tool_timeout_seconds": agent_cfg.get("tool_timeout_seconds", 30),
                "token_budget": token_budget
            }
            
            logger.info(f"[Config] Loaded: web_search_enabled={_config['agent']['web_search_enabled']}, "
                       f"tool_calling_enabled={_config['agent']['tool_calling_enabled']}, "
                       f"max_tool_rounds={_config['agent']['max_tool_rounds']}, "
                       f"max_concurrent_tools={_config['agent']['max_concurrent_tools']}, "
                       f"tool_timeout_seconds={_config['agent']['tool_timeout_seconds']}, "
                       f"token_budget={_config['agent']['token_budget']}")
            
            # Apply tool configuration
            tools_config = _config.get("tools", {})
            _apply_tool_config(tools_config)
            
            # Initialize token budget manager from config
            if _tool_system_available:
                _token_budget_manager = _build_token_budget_manager(_config["agent"]["token_budget"])
            
        except json.JSONDecodeError as e:
            logger.error(f"[Config] Invalid JSON in configuration file: {e}")
            logger.error("[Config] Using default settings")
            _config = {
                "agent": {
                    "system_prompt": "You are a helpful assistant.",
                    "max_context_messages": 20,
                    "default_temperature": 0.7,
                    "default_max_tokens": 4096,
                    "web_search_enabled": True,
                    "tool_calling_enabled": False,
                    "max_tool_rounds": 3,
                    "max_concurrent_tools": 5,
                    "tool_timeout_seconds": 30,
                    "token_budget": 100000
                },
                "tools": {},
                "fallback": {
                    "max_retries": 10,
                    "cooldown_seconds": 90,
                    "escalated_cooldown_seconds": 600
                }
            }
            # Initialize token budget manager with defaults
            if _tool_system_available and _token_budget_manager is None:
                _token_budget_manager = _build_token_budget_manager(100000)
        except Exception as e:
            logger.error(f"[Config] Error loading configuration: {e}")
            logger.error("[Config] Using default settings")
            _config = {
                "agent": {
                    "system_prompt": "You are a helpful assistant.",
                    "max_context_messages": 20,
                    "default_temperature": 0.7,
                    "default_max_tokens": 4096,
                    "web_search_enabled": True,
                    "tool_calling_enabled": False,
                    "max_tool_rounds": 3,
                    "max_concurrent_tools": 5,
                    "tool_timeout_seconds": 30,
                    "token_budget": 100000
                },
                "tools": {},
                "fallback": {
                    "max_retries": 10,
                    "cooldown_seconds": 90,
                    "escalated_cooldown_seconds": 600
                }
            }
            # Initialize token budget manager with defaults
            if _tool_system_available and _token_budget_manager is None:
                _token_budget_manager = _build_token_budget_manager(100000)
    
    return _config


def reload_config():
    """Reload config from disk (call after edits) and re-apply tool configuration."""
    global _config
    
    logger.info(f"[Config] Reloading config from {_config_path}")
    
    # Check if config file exists
    if not _config_path.exists():
        logger.error(f"[Config] Configuration file not found: {_config_path}")
        logger.error("[Config] Keeping current configuration")
        return _config
    
    try:
        _config = json.loads(_config_path.read_text(encoding="utf-8"))
        
        # Validate and provide defaults for agent section
        agent_cfg = _config.get("agent", {})
        
        # Handle token_budget - can be int or dict
        token_budget = agent_cfg.get("token_budget", 100000)
        if not isinstance(token_budget, (int, dict)):
            token_budget = 100000
        
        _config["agent"] = {
            "system_prompt": agent_cfg.get("system_prompt", "You are a helpful assistant."),
            "max_context_messages": agent_cfg.get("max_context_messages", 20),
            "default_temperature": agent_cfg.get("default_temperature", 0.7),
            "default_max_tokens": agent_cfg.get("default_max_tokens", 4096),
            "web_search_enabled": agent_cfg.get("web_search_enabled", True),
            "tool_calling_enabled": agent_cfg.get("tool_calling_enabled", False),
            "max_tool_rounds": agent_cfg.get("max_tool_rounds", 3),
            "max_concurrent_tools": agent_cfg.get("max_concurrent_tools", 5),
            "tool_timeout_seconds": agent_cfg.get("tool_timeout_seconds", 30),
            "token_budget": token_budget
        }
        
        logger.info(f"[Config] Reloaded: web_search_enabled={_config['agent']['web_search_enabled']}, "
                   f"tool_calling_enabled={_config['agent']['tool_calling_enabled']}, "
                   f"max_tool_rounds={_config['agent']['max_tool_rounds']}, "
                   f"max_concurrent_tools={_config['agent']['max_concurrent_tools']}, "
                   f"tool_timeout_seconds={_config['agent']['tool_timeout_seconds']}, "
                   f"token_budget={_config['agent']['token_budget']}")
        
        # Re-apply tool configuration
        tools_config = _config.get("tools", {})
        _apply_tool_config(tools_config)
        
        # Rebuild token budget manager from new config
        if _tool_system_available:
            _token_budget_manager = _build_token_budget_manager(_config["agent"]["token_budget"])
        
    except json.JSONDecodeError as e:
        logger.error(f"[Config] Invalid JSON in configuration file: {e}")
        logger.error("[Config] Keeping current configuration")
    except Exception as e:
        logger.error(f"[Config] Error reloading configuration: {e}")
        logger.error("[Config] Keeping current configuration")
    
    return _config


def _build_token_budget_manager(token_budget_cfg: int | dict) -> "TokenBudgetManager":
    """
    Build a TokenBudgetManager from the config value.
    
    Accepts either a plain int (treated as max_tokens with all other defaults)
    or a dict with explicit fields matching TokenBudgetConfig.
    
    Args:
        token_budget_cfg: int or dict from providers_config.json
        
    Returns:
        Configured TokenBudgetManager instance
    """
    if isinstance(token_budget_cfg, int):
        config = TokenBudgetConfig(
            enabled=True,
            max_tokens=token_budget_cfg,
            reserve_for_response=4096,
            truncation_threshold=0.8,
            chars_per_token=4.0,
        )
    else:
        config = TokenBudgetConfig(
            enabled=token_budget_cfg.get("enabled", True),
            max_tokens=token_budget_cfg.get("max_tokens", 100_000),
            reserve_for_response=token_budget_cfg.get("reserve_for_response", 4096),
            truncation_threshold=token_budget_cfg.get("truncation_threshold", 0.8),
            chars_per_token=token_budget_cfg.get("chars_per_token", 4.0),
        )
    return TokenBudgetManager(config)


# Sensitive field names that should be stripped from tool result data before
# adding to the LLM context (requirement 15.7).
_SENSITIVE_FIELD_NAMES = frozenset({
    "password", "password_hash", "passwd", "secret", "token", "api_key", "apikey",
    "access_token", "refresh_token", "auth", "authorization",
    "private_key", "client_secret", "session_id", "ssn",
    "credit_card", "card_number", "cvv", "pin",
    "api_secret", "auth_token", "bearer_token",
})


def _strip_sensitive_data(data: object) -> object:
    """
    Recursively strip sensitive fields from tool result data.
    
    Removes keys matching known sensitive field names from dict objects
    at any nesting depth. Non-dict objects are returned unchanged.
    
    Args:
        data: Tool result data (any type)
    
    Returns:
        Sanitized copy of the data with sensitive fields removed
    """
    if isinstance(data, dict):
        return {
            k: _strip_sensitive_data(v)
            for k, v in data.items()
            if k.lower() not in _SENSITIVE_FIELD_NAMES
        }
    if isinstance(data, list):
        return [_strip_sensitive_data(item) for item in data]
    return data


def _format_tool_result(result: "ToolResult") -> str:
    """
    Format a single ToolResult as a string for LLM context.
    
    Success format:
        [Tool: {name}] Status: success | Duration: {ms}ms
        {data_json}
    
    Error/timeout format:
        [Tool: {name}] Status: {status} | Error: {message}
    
    Args:
        result: ToolResult to format
    
    Returns:
        Formatted string representation
    """
    if result.status == "success":
        # Strip sensitive information before adding to context (requirement 15.7)
        safe_data = _strip_sensitive_data(result.data)
        if isinstance(safe_data, (dict, list)):
            data_str = json.dumps(safe_data, indent=2)
        else:
            data_str = str(safe_data) if safe_data is not None else ""
        return (
            f"[Tool: {result.tool_name}] Status: success | "
            f"Duration: {result.execution_time_ms:.0f}ms\n{data_str}"
        )
    else:
        return (
            f"[Tool: {result.tool_name}] Status: {result.status} | "
            f"Error: {result.error_message}"
        )


def _append_tool_results(
    messages: list[MessageDto],
    results: list["ToolResult"],
) -> list[MessageDto]:
    """
    Format and append ToolResults to the message list as role="tool" messages.
    
    Also tracks tool success/failure counts for monitoring (requirement 10.8).
    
    Args:
        messages: Current message list (modified in place and returned)
        results: ToolResult objects to format and append
    
    Returns:
        Updated message list
    """
    for result in results:
        content = _format_tool_result(result)
        messages.append(MessageDto(role="tool", content=content))
        
        # Track success/failure counts for monitoring
        if result.status == "success":
            _tool_success_counts[result.tool_name] = _tool_success_counts.get(result.tool_name, 0) + 1
        else:
            _tool_failure_counts[result.tool_name] = _tool_failure_counts.get(result.tool_name, 0) + 1
            logger.warning(
                f"[Agent/Monitor] Tool failure: {result.tool_name} - {result.status} - {result.error_message}"
            )
    
    return messages


def _build_agent_messages(
    db: Session,
    conversation_id: int,
    user_message: str,
    history: list[MessageDto] | None = None,
    web_context: str | None = None,
    deep_research: bool = False,
) -> list[MessageDto]:
    """Build the message list with system prompt and context window trimming."""
    config = get_config()
    agent_cfg = config.get("agent", {})
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful assistant.")
    max_context = agent_cfg.get("max_context_messages", 20)

    # Always give the model the real current date (it can't know it otherwise).
    today = datetime.now().strftime("%A, %B %d, %Y")
    system_prompt = f"{system_prompt}\n\nToday's date is {today}."

    # Deep Research mode: instruct the (large) model to produce a thorough,
    # structured, well-sourced answer rather than a quick reply.
    if deep_research:
        system_prompt += (
            "\n\n=== DEEP RESEARCH MODE ===\n"
            "You are operating as a deep-research analyst using a large, highly "
            "capable model. Produce a comprehensive, well-structured answer:\n"
            "• Break the topic into clear sections with headings.\n"
            "• Reason step by step and consider multiple perspectives / trade-offs.\n"
            "• Ground claims in the live web results when provided and cite sources "
            "with their URLs.\n"
            "• Distinguish established facts from uncertainty, and note gaps or "
            "conflicting evidence.\n"
            "• End with a concise takeaway or set of recommendations.\n"
            "Be rigorous and adaptive to the depth the question demands."
        )

    # Inject live web results for real-time questions.
    if web_context:
        system_prompt += (
            "\n\n⚠️ IMPORTANT: The user asked a question requiring REAL-TIME information. "
            "You have been given live web search results below. You MUST use these results "
            "as your primary source. Do NOT say you lack real-time access — you have it via "
            "the search results. Summarize the findings clearly and cite sources with URLs. "
            "If the results don't fully answer the question, say what was found and what wasn't.\n\n"
            f"=== LIVE WEB SEARCH RESULTS ===\n{web_context}\n=== END OF SEARCH RESULTS ==="
        )

    messages: list[MessageDto] = []

    # System prompt always first
    messages.append(MessageDto(role="system", content=system_prompt))

    # Load conversation history
    if history:
        context_messages = history
    else:
        db_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        context_messages = [MessageDto(role=m.role, content=m.content) for m in db_messages]

    # Trim to max context (keep most recent messages)
    if len(context_messages) > max_context:
        context_messages = context_messages[-max_context:]

    messages.extend(context_messages)
    return messages


async def agent_chat(db: Session, request: ChatRequest, owner_id: int | None = None) -> dict:
    """
    Main agent entry point for non-streaming chat.
    Orchestrates: conversation management → context building → tool orchestration → LLM routing → response persistence.

    owner_id: the authenticated account that owns any newly created conversation.
    """
    config = get_config()
    agent_cfg = config.get("agent", {})

    conversation_id = request.conversation_id

    # Create conversation if new
    if conversation_id is None:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(title=title or "New Chat", owner_id=owner_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    # Save user message
    user_msg = Message(conversation_id=conversation_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # Generate correlation ID for tracing multi-step tool sequences (requirement 12.8)
    correlation_id = str(uuid.uuid4())
    logger.info(
        f"[Agent] Processing chat request: conversation_id={conversation_id}, "
        f"correlation_id={correlation_id}"
    )

    # Build agent-managed messages
    # Real-time web search when the question likely needs current data. Deep
    # Research always gathers live context (it doesn't wait for the heuristic).
    deep_research = bool(getattr(request, "deep_research", False))
    web_context = None
    web_search_enabled = agent_cfg.get("web_search_enabled", True)
    should_search = deep_research or needs_web_search(request.message)
    logger.info(f"[Agent/Debug] web_search_enabled={web_search_enabled}, should_search={should_search}, deep_research={deep_research}, query='{request.message[:60]}'")

    if web_search_enabled and should_search:
        logger.info(f"[Agent] Triggering web search for: {request.message[:60]}")
        web_context = await web_search(request.message)
        if web_context:
            logger.info(f"[Agent] Web search succeeded, {len(web_context)} chars of context")
        else:
            logger.warning(f"[Agent] Web search returned no results")

    messages = _build_agent_messages(
        db, conversation_id, request.message, request.history, web_context,
        deep_research=deep_research,
    )

    # --- New tool orchestration loop ---
    tool_calling_enabled = agent_cfg.get("tool_calling_enabled", True)
    max_rounds = agent_cfg.get("max_tool_rounds", 3)
    max_concurrent = agent_cfg.get("max_concurrent_tools", 5)
    all_tool_results: list[ToolResult] = []
    tool_rounds_executed = 0
    citation_tracker = CitationTracker() if _tool_system_available else None

    if tool_calling_enabled and _tool_system_available and _should_run_tool_loop(request.message, web_context):
        logger.info(
            f"[Agent] Tool orchestration enabled, max_rounds={max_rounds}, "
            f"max_concurrent={max_concurrent}, conversation_id={conversation_id}, "
            f"correlation_id={correlation_id}"
        )
        
        for round_num in range(max_rounds):
            logger.info(
                f"[Agent] Tool round {round_num + 1}/{max_rounds}, "
                f"conversation_id={conversation_id}, correlation_id={correlation_id}"
            )
            
            # Get enabled tools from registry
            enabled_tools = tool_registry.get_enabled()
            if not enabled_tools:
                logger.info(f"[Agent] No enabled tools available, ending tool orchestration")
                break
            
            # Ask decision engine if tools are needed
            try:
                decision = await decision_engine.decide(
                    db=db,
                    messages=messages,
                    available_tools=enabled_tools,
                    requested_model=request.model,
                    temperature=request.temperature or agent_cfg.get("default_temperature"),
                    max_tokens=request.max_tokens or agent_cfg.get("default_max_tokens"),
                )
                
                logger.info(
                    f"[Agent] Decision: tool_calls={len(decision.tool_calls)}, "
                    f"proceed_without_tools={decision.proceed_without_tools}, "
                    f"reasoning='{decision.reasoning[:100]}'"
                )
                
                if not decision.tool_calls:
                    logger.info(f"[Agent] LLM decided no tools needed, ending orchestration")
                    break
                
            except Exception as e:
                logger.error(f"[Agent] Decision engine failed: {e}", exc_info=True)
                # Add error context to messages so LLM can acknowledge limitation (requirement 10.1, 10.2)
                error_context = (
                    "[System Note] Tool decision engine encountered an error. "
                    "The assistant will respond based on existing knowledge without real-time data access."
                )
                messages.append(MessageDto(role="tool", content=error_context))
                # Continue without tools on decision failure (requirement 10.6)
                break
            
            # Execute tool calls
            try:
                results = await tool_executor.execute_batch(
                    decision.tool_calls,
                    max_concurrent=max_concurrent
                )
                all_tool_results.extend(results)
                tool_rounds_executed += 1
                
                logger.info(
                    f"[Agent] Executed {len(results)} tool calls in round {round_num + 1}. "
                    f"Total tool calls: {len(all_tool_results)}"
                )
                
                # Ingest results into citation tracker
                if citation_tracker:
                    citation_tracker.ingest(results)
                
                # Check token budget before adding to context
                if _token_budget_manager and not _token_budget_manager.fits(results, messages):
                    logger.warning(
                        f"[Agent] Token budget exceeded, truncating {len(results)} tool results"
                    )
                    truncated_results = _token_budget_manager.truncate(results)
                    messages = _append_tool_results(messages, truncated_results)
                else:
                    messages = _append_tool_results(messages, results)
                
                # Check if all tools failed (requirement 10.6)
                all_failed = all(r.status != "success" for r in results)
                if all_failed and results:
                    logger.warning(
                        f"[Agent] All {len(results)} tool calls failed in round {round_num + 1}"
                    )
                    # Error context already added via _append_tool_results
                    # LLM will see the error ToolResults and respond accordingly
                
            except Exception as e:
                logger.error(f"[Agent] Tool execution failed: {e}", exc_info=True)
                # Add error context to messages (requirement 10.1, 10.2, 10.4)
                error_context = (
                    "[System Note] Tool execution encountered an unexpected error. "
                    "The assistant will respond based on existing knowledge and any previously retrieved information."
                )
                messages.append(MessageDto(role="tool", content=error_context))
                # Continue to final response generation even if tool execution fails (requirement 10.6)
                break
            
            # Next round: re-decide with updated context
            # If this was the last round, fall through to final response
        
        logger.info(
            f"[Agent] Tool orchestration complete. "
            f"Rounds: {tool_rounds_executed}, Total tool calls: {len(all_tool_results)}"
        )
        
        # Log tool failure rates for monitoring (requirement 10.8)
        if all_tool_results:
            failure_rates = get_tool_failure_rates()
            for tool_name, stats in failure_rates.items():
                if stats["failure_rate_percent"] > 0:
                    logger.info(
                        f"[Agent/Monitor] Tool stats: {tool_name} - "
                        f"{stats['success_count']} success, {stats['failure_count']} failure, "
                        f"{stats['failure_rate_percent']}% failure rate"
                    )
    else:
        if not tool_calling_enabled:
            logger.info("[Agent] Tool orchestration disabled by configuration")
        if not _tool_system_available:
            logger.info("[Agent] Tool system not available, skipping orchestration")

    # --- Final response generation (uses existing Fallback Router) ---
    # Guarantee response generation even if all previous steps failed (requirement 10.6)
    try:
        result: RouteResult = await route_chat(
            db=db,
            messages=messages,
            requested_model=request.model,
            temperature=request.temperature or agent_cfg.get("default_temperature"),
            max_tokens=request.max_tokens or (8192 if deep_research
                                              else agent_cfg.get("default_max_tokens")),
            deep_research=deep_research,
        )

        # Append citations to content if any sources were found
        citations_text = ""
        if citation_tracker:
            citations_text = citation_tracker.format_citations()

        final_content = result.content + (f"\n\n{citations_text}" if citations_text else "")

        model_used = result.model_id
        platform_used = result.platform
        fallback_attempts = result.attempts
        display_name = result.display_name

    except DeepResearchUnavailableError as e:
        # Surface the actionable guidance (which model to add) to the user.
        logger.warning(f"[Agent] Deep Research unavailable: {e}")
        final_content = str(e)
        model_used = "deep-research-unavailable"
        platform_used = "none"
        fallback_attempts = 0
        display_name = "Deep Research"
    except Exception as e:
        # Absolute fallback: never expose internal errors to user (requirement 10.3, 10.4)
        logger.error(f"[Agent] Final response generation failed: {e}", exc_info=True)
        final_content = (
            "I apologize, but I'm currently experiencing technical difficulties and "
            "cannot provide a complete response. Please try again in a moment."
        )
        model_used = "error-fallback"
        platform_used = "error"
        fallback_attempts = 0
        display_name = "Error Fallback"

    # Save assistant response
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=final_content,
        model_used=model_used,
        platform_used=platform_used,
    )
    db.add(assistant_msg)

    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "conversation_id": conversation_id,
        "content": final_content,
        "model": display_name,
        "platform": platform_used,
        "fallback_attempts": fallback_attempts,
        "tool_calls_made": len(all_tool_results),
        "tool_rounds": tool_rounds_executed,
    }


async def agent_stream_chat(db: Session, request: ChatRequest, on_tool_event=None, owner_id: int | None = None) -> tuple[int, StreamRouteResult, str]:
    """
    Main agent entry point for streaming chat.
    Returns (conversation_id, StreamRouteResult, citations_text) so the caller can handle SSE and append citations.

    Tool orchestration runs in non-streaming mode before streaming begins.
    Citations are returned separately for the route handler to append after the stream.

    on_tool_event: optional async callback invoked with {"type": "tool_start"|"tool_end", ...}
    around each tool execution, used by the WebSocket gateway to surface live tool
    activity in the UI. Defaults to None (the existing SSE path passes nothing).
    """
    config = get_config()
    agent_cfg = config.get("agent", {})

    conversation_id = request.conversation_id

    if conversation_id is None:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(title=title or "New Chat", owner_id=owner_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    user_msg = Message(conversation_id=conversation_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # Generate correlation ID for tracing multi-step tool sequences (requirement 12.8)
    correlation_id = str(uuid.uuid4())
    logger.info(
        f"[Agent/Stream] Processing stream chat request: conversation_id={conversation_id}, "
        f"correlation_id={correlation_id}"
    )

    # Real-time web search when the question likely needs current data. Deep
    # Research always gathers live context (it doesn't wait for the heuristic).
    deep_research = bool(getattr(request, "deep_research", False))
    web_context = None
    web_search_enabled = agent_cfg.get("web_search_enabled", True)
    should_search = deep_research or needs_web_search(request.message)
    logger.info(f"[Agent/Stream/Debug] web_search_enabled={web_search_enabled}, should_search={should_search}, deep_research={deep_research}, query='{request.message[:60]}'")

    if web_search_enabled and should_search:
        logger.info(f"[Agent/Stream] Triggering web search for: {request.message[:60]}")
        web_context = await web_search(request.message)
        if web_context:
            logger.info(f"[Agent/Stream] Web search succeeded, {len(web_context)} chars")
        else:
            logger.warning(f"[Agent/Stream] Web search returned no results")

    messages = _build_agent_messages(
        db, conversation_id, request.message, request.history, web_context,
        deep_research=deep_research,
    )

    # --- Tool orchestration loop (runs in non-streaming mode before streaming begins) ---
    tool_calling_enabled = agent_cfg.get("tool_calling_enabled", True)
    max_rounds = agent_cfg.get("max_tool_rounds", 3)
    max_concurrent = agent_cfg.get("max_concurrent_tools", 5)
    all_tool_results: list[ToolResult] = []
    tool_rounds_executed = 0
    citation_tracker = CitationTracker() if _tool_system_available else None

    if tool_calling_enabled and _tool_system_available and _should_run_tool_loop(request.message, web_context):
        logger.info(
            f"[Agent/Stream] Tool orchestration enabled, max_rounds={max_rounds}, "
            f"max_concurrent={max_concurrent}, conversation_id={conversation_id}, "
            f"correlation_id={correlation_id}"
        )
        
        try:
            for round_num in range(max_rounds):
                logger.info(
                    f"[Agent/Stream] Tool round {round_num + 1}/{max_rounds}, "
                    f"conversation_id={conversation_id}, correlation_id={correlation_id}"
                )
                
                # Get enabled tools from registry
                enabled_tools = tool_registry.get_enabled()
                if not enabled_tools:
                    logger.info(f"[Agent/Stream] No enabled tools available, ending tool orchestration")
                    break
                
                # Ask decision engine if tools are needed
                try:
                    decision = await decision_engine.decide(
                        db=db,
                        messages=messages,
                        available_tools=enabled_tools,
                        requested_model=request.model,
                        temperature=request.temperature or agent_cfg.get("default_temperature"),
                        max_tokens=request.max_tokens or agent_cfg.get("default_max_tokens"),
                    )
                    
                    logger.info(
                        f"[Agent/Stream] Decision: tool_calls={len(decision.tool_calls)}, "
                        f"proceed_without_tools={decision.proceed_without_tools}, "
                        f"reasoning='{decision.reasoning[:100]}'"
                    )
                    
                    if not decision.tool_calls:
                        logger.info(f"[Agent/Stream] LLM decided no tools needed, ending orchestration")
                        break
                    
                except Exception as e:
                    logger.error(f"[Agent/Stream] Decision engine failed: {e}", exc_info=True)
                    # Continue without tools on decision failure
                    break
                
                # Execute tool calls
                try:
                    if on_tool_event is not None:
                        for _tc in decision.tool_calls:
                            try:
                                await on_tool_event({"type": "tool_start", "tool_name": _tc.tool_name})
                            except Exception:
                                pass
                    results = await tool_executor.execute_batch(
                        decision.tool_calls,
                        max_concurrent=max_concurrent
                    )
                    all_tool_results.extend(results)
                    tool_rounds_executed += 1
                    if on_tool_event is not None:
                        for _r in results:
                            try:
                                await on_tool_event({
                                    "type": "tool_end",
                                    "tool_name": _r.tool_name,
                                    "duration_ms": round(_r.execution_time_ms, 1),
                                })
                            except Exception:
                                pass

                    logger.info(
                        f"[Agent/Stream] Executed {len(results)} tool calls in round {round_num + 1}. "
                        f"Total tool calls: {len(all_tool_results)}"
                    )
                    
                    # Ingest results into citation tracker
                    if citation_tracker:
                        citation_tracker.ingest(results)
                    
                    # Check token budget before adding to context
                    if _token_budget_manager and not _token_budget_manager.fits(results, messages):
                        logger.warning(
                            f"[Agent/Stream] Token budget exceeded, truncating {len(results)} tool results"
                        )
                        truncated_results = _token_budget_manager.truncate(results)
                        messages = _append_tool_results(messages, truncated_results)
                    else:
                        messages = _append_tool_results(messages, results)
                    
                except asyncio.CancelledError:
                    logger.warning(f"[Agent/Stream] Tool execution cancelled (client disconnect)")
                    raise  # Re-raise to propagate cancellation
                except Exception as e:
                    logger.error(f"[Agent/Stream] Tool execution failed: {e}", exc_info=True)
                    # Continue to final response generation even if tool execution fails
                    break
                
                # Next round: re-decide with updated context
                # If this was the last round, fall through to final response
            
            logger.info(
                f"[Agent/Stream] Tool orchestration complete. "
                f"Rounds: {tool_rounds_executed}, Total tool calls: {len(all_tool_results)}"
            )
        except asyncio.CancelledError:
            logger.warning(f"[Agent/Stream] Tool orchestration cancelled (client disconnect)")
            raise  # Re-raise to propagate cancellation to caller
    else:
        if not tool_calling_enabled:
            logger.info("[Agent/Stream] Tool orchestration disabled by configuration")
        if not _tool_system_available:
            logger.info("[Agent/Stream] Tool system not available, skipping orchestration")

    # --- Final streaming response generation (uses existing Fallback Router) ---
    try:
        if deep_research:
            # Deep Research: auto-continue across large (>=400B) models so the
            # user never has to type "continue"; ends with a models-used footer.
            result: StreamRouteResult = await route_deep_research_stream(
                db=db,
                messages=messages,
                temperature=request.temperature or agent_cfg.get("default_temperature"),
                max_tokens=request.max_tokens or 8192,
            )
        else:
            result: StreamRouteResult = await route_stream_chat(
                db=db,
                messages=messages,
                requested_model=request.model,
                temperature=request.temperature or agent_cfg.get("default_temperature"),
                max_tokens=request.max_tokens or agent_cfg.get("default_max_tokens"),
            )
    except asyncio.CancelledError:
        logger.warning(f"[Agent/Stream] Streaming response generation cancelled (client disconnect)")
        raise  # Re-raise to propagate cancellation

    # Format citations for the route handler to append after stream
    citations_text = ""
    if citation_tracker:
        citations_text = citation_tracker.format_citations()

    return conversation_id, result, citations_text


# ─── Full-Stack Agent Orchestration pipeline (additive) ─────────────────────
# Wraps the proven tool-orchestration loop above with the new layers: semantic
# memory (auto search/store), Redis session state, Kafka lifecycle events, the
# Planner Agent, live-data freshness context, and structured request logging.
# The existing agent_chat / agent_stream_chat remain untouched for backward
# compatibility (req 17.1).

async def _publish_event(event_type, correlation_id, conversation_id, session_id, payload):
    try:
        from . import kafka_producer as _kp
        if _kp.kafka_producer is not None:
            await _kp.kafka_producer.publish(event_type, correlation_id, conversation_id, session_id or "", payload)
    except Exception as exc:
        logger.warning(f"[Agent] Kafka publish skipped for {event_type}: {exc}")


async def _send_ws(session_id, payload):
    if not session_id:
        return
    try:
        from .ws_manager import ws_manager
        if ws_manager.is_active(session_id):
            await ws_manager.send(session_id, payload)
    except Exception as exc:
        logger.warning(f"[Agent] WS send skipped: {exc}")


async def _write_session(session_id, conversation_id):
    if not session_id:
        return
    try:
        from .redis_cache import get_redis_cache, RedisCache, SESSION_TTL
        cache = get_redis_cache()
        if cache is not None:
            await cache.set(
                RedisCache.session_key(session_id),
                {"conversation_id": conversation_id, "last_message_at": datetime.now(timezone.utc).isoformat()},
                ttl=SESSION_TTL,
            )
    except Exception as exc:
        logger.warning(f"[Agent] Redis session write skipped: {exc}")


async def _maybe_fcm_plan(session_id, subtask_count):
    """Push an FCM 'agent working' notification for 3+ step plans when the user
    has no active WebSocket session (req 11.3 / 11.6)."""
    try:
        from .ws_manager import ws_manager
        from . import fcm_notifier as _fcm
        if _fcm.fcm_notifier is None or (session_id and ws_manager.is_active(session_id)):
            return
        # Device-token lookup would resolve the target user's token here; skipped
        # when no user/token mapping is available in this deployment.
    except Exception as exc:
        logger.warning(f"[Agent] FCM plan notification skipped: {exc}")


async def agent_chat_orchestrated(db: Session, request, correlation_id: str, owner_id: int | None = None) -> dict:
    """Agent_Gateway entry point implementing the full orchestration pipeline.

    `request` is an AgentChatRequest. Returns an AgentChatResponse-shaped dict.
    owner_id: the authenticated account that owns any newly created conversation.
    """
    import time
    from .feature_flags import get_agent_features
    from . import request_context
    from .memory_manager import memory_manager
    from .planner import PlannerAgent

    flags = get_agent_features()
    session_id = request.session_id or ""
    request_context.set_request_context(
        correlation_id=correlation_id,
        requires_fresh_data=bool(getattr(request, "requires_fresh_data", False)),
        session_id=session_id,
        owner_id=owner_id,
    )
    started = time.perf_counter()

    await _publish_event("request_received", correlation_id, request.conversation_id, session_id,
                         {"message_preview": (request.message or "")[:100]})

    # Turn-start memory search → prepend ## Relevant Memory (req 8.8)
    history = list(request.history or [])
    if request.conversation_id is not None:
        mem_block = await memory_manager.auto_search(request.conversation_id, request.message)
        if mem_block:
            history = [MessageDto(role="system", content=mem_block)] + history

    # Planner: detect multi-step, emit events (req 3.10, 10.1, 11.3)
    planner_used = False
    subtask_count = 0
    if flags.planner:
        plan = await PlannerAgent(tool_registry).classify_and_plan(db, request.message, correlation_id)
        if plan and plan.subtasks:
            planner_used = True
            subtask_count = len(plan.subtasks)
            summary = [{"index": s.index, "description": s.description} for s in plan.subtasks]
            await _send_ws(session_id, {"type": "plan_created", "subtask_count": subtask_count, "subtasks": summary})
            await _publish_event("plan_created", correlation_id, request.conversation_id, session_id,
                                 {"subtask_count": subtask_count, "subtasks": summary})
            if subtask_count >= 3:
                await _maybe_fcm_plan(session_id, subtask_count)

    # Execute the tool-orchestration + LLM loop via the existing pipeline.
    chat_request = ChatRequest(
        message=request.message, conversation_id=request.conversation_id,
        model=request.model, temperature=request.temperature,
        max_tokens=request.max_tokens, history=history or None,
    )
    try:
        result = await agent_chat(db, chat_request, owner_id=owner_id)
    except Exception:
        await _publish_event("tool_failed", correlation_id, request.conversation_id, session_id,
                             {"error_message": "orchestration error"})
        raise

    conv_id = result.get("conversation_id")

    # Turn-end memory store (req 8.7)
    if conv_id is not None:
        await memory_manager.auto_store(conv_id, request.message, result.get("content", ""),
                                        user_id=request_context.get_acting_user_id())

    await _write_session(session_id, conv_id)

    duration_ms = (time.perf_counter() - started) * 1000
    await _publish_event("response_generated", correlation_id, conv_id, session_id,
                         {"model": result.get("model"), "platform": result.get("platform"),
                          "duration_ms": round(duration_ms, 2), "planner_used": planner_used,
                          "subtask_count": subtask_count})

    # Structured JSON request log (req 16.4)
    logger.info("[AgentMetrics] " + json.dumps({
        "correlation_id": correlation_id, "session_id": session_id, "conversation_id": conv_id,
        "planner_used": planner_used, "subtask_count": subtask_count,
        "tool_calls_made": result.get("tool_calls_made", 0),
        "total_duration_ms": round(duration_ms, 2), "llm_provider": result.get("platform"),
        "requires_fresh_data": bool(getattr(request, "requires_fresh_data", False)),
    }))

    return {
        "conversation_id": conv_id,
        "content": result.get("content", ""),
        "model": result.get("model"),
        "platform": result.get("platform"),
        "correlation_id": correlation_id,
        "planner_used": planner_used,
        "subtask_count": subtask_count,
        "tool_calls_made": result.get("tool_calls_made", 0),
        "requires_fresh_data": bool(getattr(request, "requires_fresh_data", False)),
    }
