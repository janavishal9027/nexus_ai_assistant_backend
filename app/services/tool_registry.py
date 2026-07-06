"""
Tool Registry for the agent tool system.

This module provides a centralized registry for managing tool definitions,
including registration, validation, and runtime enable/disable functionality.
"""

import logging
from typing import Callable
import jsonschema
from jsonschema.exceptions import SchemaError

from .tool_models import ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized catalog of available tools.
    
    Manages registration, validation, enable/disable, and metadata retrieval.
    All tool definitions are validated against JSON Schema Draft 7 on registration.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(self, definition: ToolDefinition) -> None:
        """
        Register a tool definition.
        
        Validates that:
        1. The tool name is unique (not already registered)
        2. The input_schema is a valid JSON Schema Draft 7
        3. The output_schema is a valid JSON Schema Draft 7
        
        Args:
            definition: ToolDefinition to register
        
        Raises:
            ValueError: If tool name is duplicate or schemas are invalid
        """
        # Check for duplicate name first (even before schema validation per requirement 1.4)
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        
        # Validate input schema
        try:
            self._validate_schema(definition.input_schema)
        except SchemaError as e:
            raise ValueError(f"Invalid input_schema for tool '{definition.name}': {e.message}")
        
        # Validate output schema
        try:
            self._validate_schema(definition.output_schema)
        except SchemaError as e:
            raise ValueError(f"Invalid output_schema for tool '{definition.name}': {e.message}")
        
        # All validations passed, register the tool
        self._tools[definition.name] = definition
        logger.info(f"Registered tool: {definition.name}")
    
    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        output_schema: dict,
        **kwargs
    ):
        """
        Decorator for registering a function as a tool.
        
        Usage:
            @tool_registry.tool(
                name="web_search",
                description="Search the web for information",
                input_schema={"type": "object", "properties": {...}},
                output_schema={"type": "object", "properties": {...}},
                timeout_seconds=15.0
            )
            async def web_search(query: str) -> dict:
                ...
        
        Args:
            name: Unique tool identifier
            description: Human-readable purpose for LLM prompt
            input_schema: JSON Schema for parameters
            output_schema: JSON Schema for results
            **kwargs: Additional ToolDefinition fields (enabled, timeout_seconds, etc.)
        
        Returns:
            Decorator function that registers the tool and returns the original function
        
        Raises:
            ValueError: If tool name is duplicate or schemas are invalid
        """
        def decorator(fn: Callable):
            definition = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                fn=fn,
                **kwargs
            )
            self.register(definition)
            return fn
        return decorator
    
    def get(self, name: str) -> ToolDefinition | None:
        """
        Get a tool definition by name.
        
        Args:
            name: Tool identifier
        
        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_enabled(self) -> list[ToolDefinition]:
        """
        Get all enabled tools.
        
        Returns:
            List of enabled ToolDefinitions
        """
        return [tool for tool in self._tools.values() if tool.enabled]
    
    def enable(self, name: str) -> None:
        """
        Enable a tool at runtime.
        
        Args:
            name: Tool identifier
        
        Raises:
            ValueError: If tool does not exist
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry")
        tool.enabled = True
        logger.info(f"Enabled tool: {name}")
    
    def disable(self, name: str) -> None:
        """
        Disable a tool at runtime.
        
        Args:
            name: Tool identifier
        
        Raises:
            ValueError: If tool does not exist
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry")
        tool.enabled = False
        logger.info(f"Disabled tool: {name}")
    
    def _validate_schema(self, schema: dict) -> None:
        """
        Validate a dict is a valid JSON Schema Draft 7.
        
        Args:
            schema: Dictionary to validate as JSON Schema
        
        Raises:
            jsonschema.SchemaError: If schema is invalid
        """
        jsonschema.Draft7Validator.check_schema(schema)


# Module-level singleton
tool_registry = ToolRegistry()
