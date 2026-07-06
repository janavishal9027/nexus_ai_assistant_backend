"""Planner Agent — decomposes multi-step requests into an Execution_Plan (req 3, 12).

classify_and_plan() asks the LLM whether a request is multi-step; if so it
generates a plan, validates index uniqueness and forward-only dependencies, and
re-plans up to max_retries before falling back to single-step execution.
`_validate_plan` is pure (no registry / LLM) so it is safe to unit/property test.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .tool_registry import ToolRegistry
from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 10  # req 3.9


@dataclass
class Subtask:
    index: int                         # 1-based, unique (req 12.3)
    description: str
    required_tools: list[str] = field(default_factory=list)
    dependency_indices: list[int] = field(default_factory=list)  # all < index (req 12.4)
    success_criterion: str = ""
    status: str = "pending"            # pending | completed | skipped | failed
    output: Optional[dict] = None
    failure_reason: Optional[str] = None


@dataclass
class ExecutionPlan:
    subtasks: list[Subtask]
    correlation_id: str


class PlannerAgent:
    def __init__(self, registry: Optional[ToolRegistry] = None, max_retries: int = 2) -> None:
        self.registry = registry
        self.max_retries = max_retries

    async def classify_and_plan(self, db, user_message: str, correlation_id: str) -> Optional[ExecutionPlan]:
        """Return an ExecutionPlan for multi-step requests, else None (req 3.2)."""
        try:
            if not await self._is_multi_step(db, user_message):
                return None
        except Exception as exc:
            logger.warning(f"[Planner] multi-step classification failed: {exc}; single-step")
            return None

        for attempt in range(self.max_retries + 1):
            try:
                plan = await self._generate_plan(db, user_message, correlation_id)
            except Exception as exc:
                logger.warning(f"[Planner] plan generation failed (attempt {attempt + 1}): {exc}")
                continue
            errors = self._validate_plan(plan)
            if not errors:
                return plan
            logger.warning(f"[Planner] Plan validation failed (attempt {attempt + 1}): {errors}")

        logger.warning("[Planner] Falling back to single-step execution after retry exhaustion")
        return None

    async def _is_multi_step(self, db, message: str) -> bool:
        from .fallback_router import route_chat
        prompt = (
            "Determine if this request requires multiple distinct sequential steps "
            "where the output of one step feeds into the next.\n"
            f"Request: {message}\n"
            'Respond with {"multi_step": true} or {"multi_step": false} only.'
        )
        result = await route_chat(db=db, messages=[MessageDto(role="user", content=prompt)],
                                  temperature=0.0, max_tokens=50)
        try:
            return bool(json.loads(self._extract_json(result.content)).get("multi_step", False))
        except Exception:
            return False

    async def _generate_plan(self, db, message: str, correlation_id: str) -> ExecutionPlan:
        from .fallback_router import route_chat
        available_tools = [t.name for t in self.registry.get_enabled()] if self.registry else []
        prompt = self._build_planning_prompt(message, available_tools)
        result = await route_chat(db=db, messages=[MessageDto(role="user", content=prompt)],
                                  temperature=0.2, max_tokens=2048)
        raw = json.loads(self._extract_json(result.content)) if result.content else {"subtasks": []}
        all_subtasks = raw.get("subtasks", [])
        if len(all_subtasks) > MAX_SUBTASKS:
            logger.warning(f"[Planner] Plan had >{MAX_SUBTASKS} subtasks; truncated to {MAX_SUBTASKS}")
        subtasks = [
            Subtask(
                index=s.get("index", i + 1),
                description=s.get("description", ""),
                required_tools=s.get("required_tools", []) or [],
                dependency_indices=s.get("dependency_indices", []) or [],
                success_criterion=s.get("success_criterion", ""),
            )
            for i, s in enumerate(all_subtasks[:MAX_SUBTASKS])
        ]
        return ExecutionPlan(subtasks=subtasks, correlation_id=correlation_id)

    def _validate_plan(self, plan: ExecutionPlan) -> list[str]:
        """Pure validation: unique 1..N indices (req 12.3/12.7) + forward-only deps (req 12.4)."""
        errors: list[str] = []
        indices = [s.index for s in plan.subtasks]
        n = len(indices)
        if sorted(indices) != list(range(1, n + 1)):
            errors.append(f"Subtask indices {indices} are not unique consecutive integers starting at 1")
        for s in plan.subtasks:
            for dep in s.dependency_indices:
                if dep >= s.index:
                    errors.append(f"Subtask {s.index} has forward/self dependency on {dep}")
        return errors

    def _build_planning_prompt(self, message: str, tools: list[str]) -> str:
        tools_list = "\n".join(f"- {t}" for t in tools) or "- (no tools registered)"
        return (
            "Decompose the following request into sequential subtasks.\n"
            f"Available tools:\n{tools_list}\n\n"
            f"Request: {message}\n\n"
            "Rules: index starts at 1 and increments by 1; dependency_indices must all be "
            "lower than the subtask's own index; reference only tools from the list.\n"
            "Respond with JSON only, matching this schema exactly:\n"
            '{"subtasks": [{"index": 1, "description": "...", '
            '"required_tools": ["tool_name"], "dependency_indices": [], '
            '"success_criterion": "..."}]}'
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        if not text:
            return "{}"
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fall back to the first {...} block.
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        return brace.group(0) if brace else text.strip()
