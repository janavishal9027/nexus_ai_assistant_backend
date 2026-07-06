# Disabled-Tool Guard Flow Documentation

## Overview

This document illustrates how the disabled-tool guard system prevents unauthorized or disabled tool execution through multiple layers of protection.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Chat Request Received                     │
│                   (user query + context)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Load Configuration from JSON                    │
│        tool_calling_enabled = config['agent']['...']        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │ Tool Calling       │
                  │ Enabled?           │
                  └────────┬───────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                 NO                YES
                  │                 │
                  ▼                 ▼
     ┌──────────────────┐  ┌──────────────────────────┐
     │ Skip Tool Loop   │  │ Tool System Available?   │
     │ (go to response) │  └──────────┬───────────────┘
     └──────────────────┘             │
                                ┌─────┴─────┐
                                │           │
                               NO          YES
                                │           │
                                │           ▼
                                │  ┌─────────────────────────┐
                                │  │ Get Enabled Tools from  │
                                │  │    Tool Registry        │
                                │  └──────────┬──────────────┘
                                │             │
                                │             ▼
                                │   ┌────────────────────┐
                                │   │ Any Tools Enabled? │
                                │   └─────────┬──────────┘
                                │             │
                                │        ┌────┴────┐
                                │        │         │
                                │       NO        YES
                                │        │         │
                                ▼        ▼         ▼
                    ┌─────────────────────────────────────┐
                    │      Skip Tool Loop / Break         │
                    │   (proceed to final response)       │
                    └─────────────────────────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────┐
                              │   Decision Engine Called     │
                              │  (only sees enabled tools)   │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  LLM Decides Tool Calls      │
                              │  (can only request tools in  │
                              │   available_tools list)      │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │   Validate Tool Names        │
                              │  (in _map_tool_calls)        │
                              └──────────────┬───────────────┘
                                             │
                                      ┌──────┴──────┐
                                      │             │
                               INVALID          VALID
                                      │             │
                                      ▼             ▼
                          ┌──────────────┐  ┌────────────────┐
                          │ Log Error &  │  │ Execute Tool   │
                          │ Skip Tool    │  │ via Executor   │
                          └──────────────┘  └────────────────┘
```

## Guard Layers Explained

### Layer 1: Global Tool Calling Toggle

**File**: `app/services/agent.py`  
**Line**: 520

```python
tool_calling_enabled = agent_cfg.get("tool_calling_enabled", True)
```

**Purpose**: System-wide enable/disable switch
**Effect**: When `False`, entire tool system is bypassed
**Configuration**: `providers_config.json` → `agent.tool_calling_enabled`

### Layer 2: Tool System Availability Check

**File**: `app/services/agent.py`  
**Line**: 18-23

```python
try:
    from .tool_registry import tool_registry
    from .decision import DecisionEngine
    # ... other imports ...
    _tool_system_available = True
except ImportError as e:
    _tool_system_available = False
```

**Purpose**: Graceful degradation if tool modules missing
**Effect**: When `False`, tool system cannot be used

### Layer 3: Combined Guard Condition

**File**: `app/services/agent.py`  
**Line**: 528

```python
if tool_calling_enabled and _tool_system_available:
    # Tool orchestration loop
```

**Purpose**: Final gate before entering tool orchestration
**Effect**: Must pass BOTH checks to use tools

### Layer 4: Registry Filtering

**File**: `app/services/tool_registry.py`  
**Line**: 82-87

```python
def get_enabled(self) -> list[ToolDefinition]:
    """Get all enabled tools."""
    return [tool for tool in self._tools.values() if tool.enabled]
```

**Purpose**: Per-tool granular control
**Effect**: Only tools with `enabled=True` are returned
**Configuration**: `providers_config.json` → `tools.{tool_name}.enabled`

### Layer 5: Empty Tools Check

**File**: `app/services/agent.py`  
**Line**: 536-539

```python
enabled_tools = tool_registry.get_enabled()
if not enabled_tools:
    logger.info(f"[Agent] No enabled tools available, ending tool orchestration")
    break
```

**Purpose**: Prevent wasted LLM calls when no tools available
**Effect**: Breaks orchestration loop if all tools disabled

### Layer 6: Tool Name Validation

**File**: `app/services/decision.py`  
**Line**: 265-272

```python
if tool_name not in tool_names:
    logger.error(
        f"[DecisionEngine] Invalid tool name '{tool_name}' requested, "
        f"not in available tools: {tool_names}"
    )
    continue
```

**Purpose**: Validate LLM's tool requests
**Effect**: Filters out unrecognized/disabled tool names
**Handles**: Edge cases (tool disabled mid-conversation, config errors)

## Configuration Examples

### Disable All Tools

```json
{
  "agent": {
    "tool_calling_enabled": false
  }
}
```

**Result**: Entire tool orchestration loop skipped

### Disable Specific Tool

```json
{
  "agent": {
    "tool_calling_enabled": true
  },
  "tools": {
    "web_search": {
      "enabled": false
    }
  }
}
```

**Result**: `web_search` not included in `available_tools` list

### Enable All (Default)

```json
{
  "agent": {
    "tool_calling_enabled": true
  },
  "tools": {
    "web_search": {
      "enabled": true
    }
  }
}
```

**Result**: All registered and enabled tools available

## Logging Examples

### Tool Calling Disabled

```
[Agent] Tool orchestration disabled by configuration
```

### Tool System Unavailable

```
[Agent] Tool system not available, skipping orchestration
```

### No Enabled Tools

```
[Agent] No enabled tools available, ending tool orchestration
```

### Invalid Tool Requested

```
[DecisionEngine] Invalid tool name 'weather_api' requested, not in available tools: {'web_search'}
```

## Runtime Control

Tools can be enabled/disabled at runtime:

```python
from app.services.tool_registry import tool_registry

# Disable a tool
tool_registry.disable("web_search")

# Re-enable it
tool_registry.enable("web_search")
```

**Note**: Changes take effect immediately in the next conversation turn.

## Edge Cases Handled

### 1. Tool Disabled Mid-Conversation

**Scenario**: Tool is disabled between LLM decision and execution

**Handling**: 
- Decision engine won't see it in next round
- Tool executor validation catches it
- Error logged, execution skipped

### 2. All Tools Disabled

**Scenario**: User disables all tools via configuration

**Handling**:
- `get_enabled()` returns empty list
- Loop breaks immediately (line 537-539)
- Proceeds to final response generation

### 3. Configuration Error

**Scenario**: Config file specifies non-existent tool

**Handling**:
- Tool not in registry
- Validation catches it (line 265-272)
- Error logged, ignored

### 4. LLM Requests Invalid Tool

**Scenario**: LLM hallucinates a tool name not in available list

**Handling**:
- `_map_tool_calls()` validation
- Error logged with tool name and available tools
- Invalid call filtered out

## Security Considerations

1. **No Bypass Mechanism**: There's no way to invoke a disabled tool through the normal flow

2. **Configuration Validation**: Config errors default to safe behavior (skip tools)

3. **Multiple Validation Points**: Even if one layer fails, others catch issues

4. **Comprehensive Logging**: All guard decisions logged for audit trail

5. **Graceful Degradation**: System continues to function even when tools fail/disabled

## Performance Impact

The guard system has **minimal performance overhead**:

- Guard checks: O(1) boolean comparisons
- `get_enabled()`: O(n) filter operation where n = total registered tools (typically < 10)
- Validation: O(1) set membership check

**Expected overhead**: < 1ms per request

## Maintenance Notes

When adding new tools:

1. Register with `@tool_registry.tool()` decorator
2. Set `enabled=True` in default configuration
3. No changes to guard logic needed

When modifying configuration:

1. Update `providers_config.json`
2. Call `reload_config()` or restart server
3. Changes take effect immediately

## References

- Task: 13.2 Implement disabled-tool guard
- Requirements: 9.4, 9.5, 11.7
- Design Document: Section on Tool Configuration and Availability
