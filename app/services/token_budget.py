"""
Token Budget Management for Agent Tool Operations.

This module provides token tracking and enforcement for tool operations,
ensuring context window limits are respected during multi-step tool orchestration.
"""

import json
import logging
from dataclasses import dataclass

from ..models.schemas import MessageDto
from .tool_models import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class TokenBudgetConfig:
    """
    Configuration for token budget management.
    
    Attributes:
        enabled: Whether token budget enforcement is active
        max_tokens: Maximum total tokens allowed in context
        reserve_for_response: Tokens reserved for final LLM response generation
        truncation_threshold: Fraction of max_tokens at which truncation occurs (0.0-1.0)
        chars_per_token: Character-to-token ratio for estimation (default 4.0)
    """
    enabled: bool = True
    max_tokens: int = 100_000
    reserve_for_response: int = 4096
    truncation_threshold: float = 0.8
    chars_per_token: float = 4.0


class TokenBudgetManager:
    """
    Tracks token consumption across tool descriptions, tool calls, and tool results.
    Enforces limits and truncates when necessary.
    
    Responsibilities:
        - Estimate tokens from text using character-to-token ratio
        - Check if adding tool results would exceed budget
        - Truncate tool results prioritizing recent results
        - Log warnings when usage exceeds threshold
    """
    
    def __init__(self, config: TokenBudgetConfig):
        """
        Initialize the token budget manager.
        
        Args:
            config: Token budget configuration
        """
        self.config = config
        logger.info(
            f"[TokenBudget] Initialized with max_tokens={config.max_tokens}, "
            f"threshold={config.truncation_threshold}, chars_per_token={config.chars_per_token}"
        )
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens from text using character-to-token ratio.
        
        Uses a conservative character-to-token ratio (default 4.0) suitable
        for typical English text. This avoids dependency on model-specific
        tokenizers and provides reasonable estimates across different models.
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)
    
    def fits(self, results: list[ToolResult], current_messages: list[MessageDto]) -> bool:
        """
        Check if adding tool results would exceed token budget.
        
        Calculates total tokens from current messages + new results + response reserve,
        then checks against the truncation threshold.
        
        Args:
            results: Tool results to be added to context
            current_messages: Current message list in context
            
        Returns:
            True if results fit within budget, False if truncation needed
        """
        if not self.config.enabled:
            return True
        
        # Estimate tokens from current messages
        current_tokens = sum(self.estimate_tokens(m.content) for m in current_messages)
        
        # Estimate tokens from tool results and log per-tool consumption (requirement 12.3)
        result_tokens = 0
        for result in results:
            if result.status == "success" and result.data:
                # Serialize data to JSON for accurate token estimation
                data_text = json.dumps(result.data) if isinstance(result.data, (dict, list)) else str(result.data)
                tool_tokens = self.estimate_tokens(data_text)
            elif result.error_message:
                tool_tokens = self.estimate_tokens(result.error_message)
            else:
                tool_tokens = 0
            result_tokens += tool_tokens
            # Log token consumption per individual tool call (requirement 12.3)
            logger.debug(
                f"[TokenBudget] Token consumption for tool '{result.tool_name}' "
                f"(call_id={result.call_id}): ~{tool_tokens} tokens"
            )
        
        # Total with response reserve
        total_tokens = current_tokens + result_tokens + self.config.reserve_for_response
        threshold_limit = int(self.config.max_tokens * self.config.truncation_threshold)
        
        # Log warning if exceeding 80% threshold
        usage_ratio = total_tokens / self.config.max_tokens
        if usage_ratio >= 0.8:
            logger.warning(
                f"[TokenBudget] Token usage at {usage_ratio:.1%} "
                f"({total_tokens}/{self.config.max_tokens} tokens). "
                f"Threshold: {self.config.truncation_threshold:.1%}"
            )
        
        fits = total_tokens < threshold_limit
        logger.debug(
            f"[TokenBudget] Check: current={current_tokens}, results={result_tokens}, "
            f"total={total_tokens}, threshold={threshold_limit}, fits={fits}"
        )
        
        return fits
    
    def truncate(self, results: list[ToolResult]) -> list[ToolResult]:
        """
        Truncate tool results to fit budget, prioritizing recent results.
        
        Strategy:
            1. Calculate available token budget (threshold - reserve)
            2. Keep as many recent results as possible within budget
            3. Add truncation notice to the first kept result
            4. If truncation notice itself doesn't fit, return all results unchanged
        
        Args:
            results: Tool results to truncate
            
        Returns:
            Truncated list of tool results with truncation notice, or original
            list if truncation notice doesn't fit
        """
        if not self.config.enabled or not results:
            return results
        
        # Calculate available budget
        available_tokens = int(self.config.max_tokens * self.config.truncation_threshold) - self.config.reserve_for_response
        
        truncation_notice = "[...earlier tool results truncated due to token budget...]"
        notice_tokens = self.estimate_tokens(truncation_notice)
        
        # Start with notice tokens in budget
        total_tokens = notice_tokens
        kept_results = []
        
        # Iterate from most recent (end) to oldest (start)
        for result in reversed(results):
            # Estimate tokens for this result
            if result.status == "success" and result.data:
                data_text = json.dumps(result.data) if isinstance(result.data, (dict, list)) else str(result.data)
                result_tokens = self.estimate_tokens(data_text)
            elif result.error_message:
                result_tokens = self.estimate_tokens(result.error_message)
            else:
                result_tokens = 0
            
            # Check if adding this result would exceed available budget
            if total_tokens + result_tokens > available_tokens:
                break
            
            # Add to kept results (insert at beginning to maintain order)
            kept_results.insert(0, result)
            total_tokens += result_tokens
        
        # Edge case: If we couldn't keep any results, return all unchanged
        if not kept_results:
            logger.warning(
                f"[TokenBudget] Truncation notice itself exceeds budget "
                f"({notice_tokens} tokens > {available_tokens} available). "
                f"Returning all {len(results)} results unchanged."
            )
            return results
        
        # Add truncation notice to the first kept result's data
        truncated_count = len(results) - len(kept_results)
        if truncated_count > 0:
            first_result = kept_results[0]
            
            # Add notice to the data field
            if first_result.status == "success" and isinstance(first_result.data, dict):
                first_result.data = {
                    "_truncation_notice": truncation_notice,
                    **first_result.data
                }
            
            logger.info(
                f"[TokenBudget] Truncated {truncated_count}/{len(results)} tool results. "
                f"Kept {len(kept_results)} most recent results ({total_tokens} tokens)."
            )
        
        return kept_results
