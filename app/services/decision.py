"""
Decision Engine for the agent tool system.

This module provides the DecisionEngine class that uses an LLM to analyze
user queries and determine which tools (if any) should be invoked.
"""

import json
import logging
import uuid
import re
from datetime import datetime
from typing import Any
from sqlalchemy.orm import Session

from .tool_models import ToolCall, DecisionResult, ToolDefinition
from .tool_registry import ToolRegistry
from .fallback_router import route_chat
from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    LLM-based tool decision making component.
    
    Analyzes conversation context to determine if external tools are needed,
    and generates structured tool calls when appropriate.
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the decision engine.
        
        Args:
            registry: ToolRegistry instance for accessing available tools
        """
        self.registry = registry
    
    async def decide(
        self,
        db: Session,
        messages: list[MessageDto],
        available_tools: list[ToolDefinition],
        requested_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> DecisionResult:
        """
        Ask the LLM if tools are needed and which ones to invoke.
        
        Args:
            db: Database session (required by route_chat)
            messages: Conversation history
            available_tools: List of enabled tools to present to the LLM
            requested_model: Model to use for decision (optional)
            temperature: LLM temperature (optional)
            max_tokens: Max tokens for response (optional)
        
        Returns:
            DecisionResult with tool_calls (empty list = no tools needed)
        """
        # Handle case where no tools are available (requirement 2.3)
        if not available_tools:
            logger.info("[DecisionEngine] No tools available, proceeding without tools")
            return DecisionResult(
                tool_calls=[],
                reasoning="No tools available",
                proceed_without_tools=True
            )
        
        # Build the tool selection prompt
        prompt = self._build_prompt(messages, available_tools)
        
        # Create message list for LLM call
        decision_messages = [MessageDto(role="user", content=prompt)]
        
        # Make non-streaming LLM call (requirement 2.5)
        try:
            result = await route_chat(
                db=db,
                messages=decision_messages,
                requested_model=requested_model,
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens if max_tokens is not None else 2048,
            )
            
            response_text = result.content
            logger.info(f"[DecisionEngine] LLM decision response received ({len(response_text)} chars)")
            
        except Exception as e:
            logger.error(f"[DecisionEngine] LLM call failed: {e}")
            # Fall back to proceeding without tools (requirement 2.5)
            return DecisionResult(
                tool_calls=[],
                reasoning=f"LLM call failed: {str(e)}",
                proceed_without_tools=True
            )
        
        # Parse the JSON response
        parsed = self._parse_response(response_text)
        
        if parsed is None:
            # JSON parse failure (requirement 2.5)
            logger.warning("[DecisionEngine] Failed to parse LLM response as JSON")
            return DecisionResult(
                tool_calls=[],
                reasoning="Failed to parse LLM response",
                proceed_without_tools=True
            )
        
        # Extract reasoning
        reasoning = parsed.get("reasoning", "No reasoning provided")
        
        # Check if tools are needed
        tools_needed = parsed.get("tools_needed", False)
        raw_tool_calls = parsed.get("tool_calls", [])
        
        if not tools_needed or not raw_tool_calls:
            # LLM decided no tools needed (requirement 2.8)
            logger.info("[DecisionEngine] LLM decided no tools are needed")
            return DecisionResult(
                tool_calls=[],
                reasoning=reasoning,
                proceed_without_tools=False
            )
        
        # Map raw tool calls to ToolCall instances
        tool_calls = self._map_tool_calls(raw_tool_calls, available_tools)
        
        # Log decision details at DEBUG level (requirement 12.4)
        logger.debug(
            f"[DecisionEngine] Decision: tools_needed={tools_needed}, "
            f"reasoning='{reasoning}', tool_calls={len(tool_calls)}"
        )
        logger.info(f"[DecisionEngine] Generated {len(tool_calls)} valid tool calls")
        return DecisionResult(
            tool_calls=tool_calls,
            reasoning=reasoning,
            proceed_without_tools=False
        )
    
    def _build_prompt(
        self,
        messages: list[MessageDto],
        available_tools: list[ToolDefinition]
    ) -> str:
        """
        Build the tool selection prompt for the LLM.
        
        Includes current date/time, available tools with descriptions and schemas,
        and recent conversation history.
        
        Args:
            messages: Conversation history
            available_tools: List of tools to present
        
        Returns:
            Formatted prompt string
        """
        # Current date and time (requirement 2.9)
        current_datetime = datetime.now().strftime("%A, %B %d, %Y %H:%M UTC")
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in available_tools:
            # Format tool with description and parameters (requirement 2.2)
            params_json = json.dumps(tool.input_schema.get("properties", {}), indent=2)
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {params_json}"
            )
        
        tools_text = "\n".join(tool_descriptions)
        
        # Format recent conversation
        conversation_lines = []
        for msg in messages[-10:]:  # Last 10 messages for context
            conversation_lines.append(f"{msg.role}: {msg.content}")
        
        conversation_text = "\n".join(conversation_lines)
        
        # Build complete prompt (requirement 2.2)
        prompt = f"""You are a tool-calling assistant. Based on the conversation, determine if any tools are needed to answer the user's question.

Current date and time: {current_datetime}

Available tools:
{tools_text}

Conversation:
{conversation_text}

Respond with a JSON object in this exact format:
{{
  "tools_needed": true|false,
  "reasoning": "brief explanation",
  "tool_calls": [
    {{"tool": "tool_name", "parameters": {{...}}}}
  ]
}}

If no tools are needed, set tools_needed to false and tool_calls to [].
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> dict[str, Any] | None:
        """
        Parse the LLM's JSON response.
        
        Handles JSON wrapped in markdown code blocks (```json ... ```).
        
        Args:
            response_text: Raw LLM response
        
        Returns:
            Parsed dict or None if parsing fails
        """
        # Try to extract JSON from markdown code blocks first
        code_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        
        if matches:
            # Try parsing the first code block
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON-like content
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _map_tool_calls(
        self,
        raw_tool_calls: list[dict[str, Any]],
        available_tools: list[ToolDefinition]
    ) -> list[ToolCall]:
        """
        Map raw tool calls from LLM JSON to ToolCall instances.
        
        Validates tool names against available tools, logs and filters out
        unrecognized tools (requirement 2.6).
        
        Args:
            raw_tool_calls: List of dicts with 'tool' and 'parameters' keys
            available_tools: List of available tool definitions
        
        Returns:
            List of valid ToolCall instances with generated UUIDs
        """
        tool_names = {tool.name for tool in available_tools}
        valid_calls = []
        
        for raw_call in raw_tool_calls:
            tool_name = raw_call.get("tool")
            parameters = raw_call.get("parameters", {})
            
            if not tool_name:
                logger.warning("[DecisionEngine] Tool call missing 'tool' field, skipping")
                continue
            
            # Validate tool exists (requirement 2.6)
            if tool_name not in tool_names:
                logger.error(
                    f"[DecisionEngine] Invalid tool name '{tool_name}' requested, "
                    f"not in available tools: {tool_names}"
                )
                continue
            
            # Generate UUID for call_id
            call_id = str(uuid.uuid4())
            
            # Create ToolCall instance
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                call_id=call_id
            )
            
            valid_calls.append(tool_call)
            logger.info(
                f"[DecisionEngine] Created tool call: {tool_name} "
                f"(call_id: {call_id[:8]}...)"
            )
        
        return valid_calls
