"""
Unit tests for agent configuration loading and tool registry integration.

Tests the enhanced get_config() and reload_config() functions in agent.py,
verifying that:
- Configuration loads from providers_config.json
- Missing/invalid configuration falls back to defaults
- Tool registry is configured with per-tool settings
- Configuration can be reloaded without restart
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.services import agent


@pytest.fixture
def sample_config():
    """Sample configuration matching the expected structure."""
    return {
        "agent": {
            "system_prompt": "Test prompt",
            "max_context_messages": 25,
            "default_temperature": 0.8,
            "default_max_tokens": 2048,
            "web_search_enabled": False,
            "tool_calling_enabled": True,
            "max_tool_rounds": 5,
            "max_concurrent_tools": 10,
            "tool_timeout_seconds": 45,
            "token_budget": 200000
        },
        "tools": {
            "web_search": {
                "enabled": False,
                "timeout_seconds": 20,
                "max_results": 10
            },
            "test_tool": {
                "enabled": True,
                "timeout_seconds": 30
            }
        },
        "fallback": {
            "max_retries": 5,
            "cooldown_seconds": 60,
            "escalated_cooldown_seconds": 300
        }
    }


@pytest.fixture
def minimal_config():
    """Minimal configuration with only required fields."""
    return {
        "agent": {
            "system_prompt": "Minimal prompt"
        }
    }


@pytest.fixture
def reset_config():
    """Reset the global _config before and after each test."""
    # Save original state
    original_config = agent._config.copy() if agent._config else {}
    
    # Save original tool timeouts
    original_tool_states = {}
    if agent._tool_registry_available:
        from app.services.tool_registry import tool_registry
        for tool_name in ["web_search"]:  # List tools that might be modified
            tool = tool_registry.get(tool_name)
            if tool:
                original_tool_states[tool_name] = {
                    'timeout_seconds': tool.timeout_seconds,
                    'enabled': tool.enabled
                }
    
    agent._config = {}
    yield
    
    # Restore original state
    agent._config = original_config
    
    # Restore original tool states
    if agent._tool_registry_available:
        from app.services.tool_registry import tool_registry
        for tool_name, state in original_tool_states.items():
            tool = tool_registry.get(tool_name)
            if tool:
                tool.timeout_seconds = state['timeout_seconds']
                tool.enabled = state['enabled']


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_get_config_loads_all_agent_settings(self, reset_config, sample_config, tmp_path):
        """Test that get_config loads all agent settings from the file."""
        # Create temporary config file
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file):
            config = agent.get_config()
        
        agent_cfg = config["agent"]
        assert agent_cfg["system_prompt"] == "Test prompt"
        assert agent_cfg["max_context_messages"] == 25
        assert agent_cfg["default_temperature"] == 0.8
        assert agent_cfg["default_max_tokens"] == 2048
        assert agent_cfg["web_search_enabled"] is False
        assert agent_cfg["tool_calling_enabled"] is True
        assert agent_cfg["max_tool_rounds"] == 5
        assert agent_cfg["max_concurrent_tools"] == 10
        assert agent_cfg["tool_timeout_seconds"] == 45
        assert agent_cfg["token_budget"] == 200000
    
    def test_get_config_applies_defaults_for_missing_fields(self, reset_config, minimal_config, tmp_path):
        """Test that get_config applies default values for missing fields."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(minimal_config))
        
        with patch.object(agent, '_config_path', config_file):
            config = agent.get_config()
        
        agent_cfg = config["agent"]
        assert agent_cfg["system_prompt"] == "Minimal prompt"
        assert agent_cfg["max_context_messages"] == 20  # default
        assert agent_cfg["default_temperature"] == 0.7  # default
        assert agent_cfg["default_max_tokens"] == 4096  # default
        assert agent_cfg["web_search_enabled"] is True  # default
        assert agent_cfg["tool_calling_enabled"] is False  # default
        assert agent_cfg["max_tool_rounds"] == 3  # default
        assert agent_cfg["max_concurrent_tools"] == 5  # default
        assert agent_cfg["tool_timeout_seconds"] == 30  # default
        assert agent_cfg["token_budget"] == 100000  # default
    
    def test_get_config_handles_missing_file(self, reset_config, tmp_path):
        """Test that get_config falls back to defaults when file is missing."""
        nonexistent_file = tmp_path / "nonexistent.json"
        
        with patch.object(agent, '_config_path', nonexistent_file):
            config = agent.get_config()
        
        agent_cfg = config["agent"]
        assert agent_cfg["system_prompt"] == "You are a helpful assistant."
        assert agent_cfg["tool_calling_enabled"] is False
        assert agent_cfg["max_tool_rounds"] == 3
        assert "tools" in config
        assert config["tools"] == {}
    
    def test_get_config_handles_invalid_json(self, reset_config, tmp_path):
        """Test that get_config falls back to defaults when JSON is invalid."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text("{ invalid json }")
        
        with patch.object(agent, '_config_path', config_file):
            config = agent.get_config()
        
        agent_cfg = config["agent"]
        assert agent_cfg["system_prompt"] == "You are a helpful assistant."
        assert agent_cfg["tool_calling_enabled"] is False
    
    def test_get_config_caches_result(self, reset_config, sample_config, tmp_path):
        """Test that get_config caches the result and doesn't reload."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file):
            config1 = agent.get_config()
            
            # Modify file
            modified_config = sample_config.copy()
            modified_config["agent"]["max_tool_rounds"] = 999
            config_file.write_text(json.dumps(modified_config))
            
            # Should return cached version
            config2 = agent.get_config()
            assert config2["agent"]["max_tool_rounds"] == 5  # original value


class TestToolRegistryIntegration:
    """Tests for tool registry integration."""
    
    def test_apply_tool_config_enables_and_disables_tools(self, reset_config, sample_config, tmp_path):
        """Test that tool configuration is applied to tool_registry."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        # Mock tool_registry
        mock_registry = MagicMock()
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()
        mock_registry.get.side_effect = lambda name: mock_tool1 if name == "web_search" else mock_tool2 if name == "test_tool" else None
        
        with patch.object(agent, '_config_path', config_file), \
             patch.object(agent, 'tool_registry', mock_registry), \
             patch.object(agent, '_tool_registry_available', True):
            agent.get_config()
        
        # Verify enable/disable calls
        assert mock_registry.disable.call_count == 1
        mock_registry.disable.assert_called_with("web_search")
        assert mock_registry.enable.call_count == 1
        mock_registry.enable.assert_called_with("test_tool")
    
    def test_apply_tool_config_sets_timeout(self, reset_config, sample_config, tmp_path):
        """Test that per-tool timeout is applied."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        # Create real objects to track attribute changes
        class MockTool:
            def __init__(self, name):
                self.name = name
                self.timeout_seconds = 30.0
        
        mock_registry = MagicMock()
        web_search_tool = MockTool("web_search")
        test_tool = MockTool("test_tool")
        
        def get_tool(name):
            if name == "web_search":
                return web_search_tool
            elif name == "test_tool":
                return test_tool
            return None
        
        mock_registry.get.side_effect = get_tool
        
        with patch.object(agent, '_config_path', config_file), \
             patch.object(agent, 'tool_registry', mock_registry), \
             patch.object(agent, '_tool_registry_available', True):
            agent.get_config()
        
        # Verify timeout was set for web_search (20s in config)
        assert web_search_tool.timeout_seconds == 20.0
        # Verify timeout was set for test_tool (30s in config)
        assert test_tool.timeout_seconds == 30.0
    
    def test_apply_tool_config_handles_unregistered_tools(self, reset_config, tmp_path):
        """Test that configuration for unregistered tools is logged but doesn't crash."""
        config = {
            "agent": {"system_prompt": "Test"},
            "tools": {
                "nonexistent_tool": {
                    "enabled": True
                }
            }
        }
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(config))
        
        # Mock tool_registry
        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # Tool not found
        
        with patch.object(agent, '_config_path', config_file), \
             patch.object(agent, 'tool_registry', mock_registry), \
             patch.object(agent, '_tool_registry_available', True):
            # Should not raise
            config_result = agent.get_config()
        
        assert config_result is not None
    
    def test_apply_tool_config_skips_when_registry_unavailable(self, reset_config, sample_config, tmp_path):
        """Test that tool configuration is skipped when tool_registry is not available."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file), \
             patch.object(agent, '_tool_registry_available', False):
            # Should not raise
            config = agent.get_config()
        
        assert config is not None


class TestConfigReloading:
    """Tests for configuration reloading."""
    
    def test_reload_config_updates_settings(self, reset_config, sample_config, tmp_path):
        """Test that reload_config updates settings from disk."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file):
            # Load initial config
            config1 = agent.get_config()
            assert config1["agent"]["max_tool_rounds"] == 5
            
            # Modify file
            modified_config = sample_config.copy()
            modified_config["agent"]["max_tool_rounds"] = 7
            config_file.write_text(json.dumps(modified_config))
            
            # Reload
            config2 = agent.reload_config()
            assert config2["agent"]["max_tool_rounds"] == 7
    
    def test_reload_config_reapplies_tool_config(self, reset_config, sample_config, tmp_path):
        """Test that reload_config re-applies tool registry configuration."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        # Mock tool_registry
        mock_registry = MagicMock()
        mock_tool = MagicMock()
        mock_registry.get.return_value = mock_tool
        
        with patch.object(agent, '_config_path', config_file), \
             patch.object(agent, 'tool_registry', mock_registry), \
             patch.object(agent, '_tool_registry_available', True):
            # Initial load
            agent.get_config()
            initial_enable_count = mock_registry.enable.call_count
            
            # Modify config - enable web_search
            modified_config = sample_config.copy()
            modified_config["tools"]["web_search"]["enabled"] = True
            config_file.write_text(json.dumps(modified_config))
            
            # Reload
            agent.reload_config()
            
            # Should have been called again
            assert mock_registry.enable.call_count > initial_enable_count
    
    def test_reload_config_handles_missing_file(self, reset_config, sample_config, tmp_path):
        """Test that reload_config handles missing file gracefully."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file):
            # Load initial config
            config1 = agent.get_config()
            initial_rounds = config1["agent"]["max_tool_rounds"]
            
            # Delete file
            config_file.unlink()
            
            # Reload should keep current config
            config2 = agent.reload_config()
            assert config2["agent"]["max_tool_rounds"] == initial_rounds
    
    def test_reload_config_handles_invalid_json(self, reset_config, sample_config, tmp_path):
        """Test that reload_config handles invalid JSON gracefully."""
        config_file = tmp_path / "providers_config.json"
        config_file.write_text(json.dumps(sample_config))
        
        with patch.object(agent, '_config_path', config_file):
            # Load initial config
            config1 = agent.get_config()
            initial_rounds = config1["agent"]["max_tool_rounds"]
            
            # Corrupt file
            config_file.write_text("{ invalid json }")
            
            # Reload should keep current config
            config2 = agent.reload_config()
            assert config2["agent"]["max_tool_rounds"] == initial_rounds
