"""
Tools module for agent-based real-time data access.

Import all tool modules here to ensure they register with the tool_registry.
All imports are additive; each module registers its tools via decorators at
import time (req 17.4).
"""

# Import all tools to trigger decorator registration
from . import web_search
from . import user_service      # user_get, user_list, user_create, user_update
from . import task_service      # task_get, task_list, task_create, task_update, task_complete
from . import database_tool     # query_database
from . import memory_tool       # memory_store, memory_search, memory_delete, memory_store_batch
from . import realtime_tool     # realtime_get_state, realtime_recent_events

__all__ = [
    "web_search",
    "user_service",
    "task_service",
    "database_tool",
    "memory_tool",
    "realtime_tool",
]
