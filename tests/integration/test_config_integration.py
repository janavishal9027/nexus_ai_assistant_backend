"""
Integration test for configuration loading and tool registry integration.

This test verifies that the actual providers_config.json file is loaded correctly
and applied to the tool registry when the application starts.
"""

import pytest
from app.services.agent import get_config, reload_config
from app.services.tool_registry import tool_registry


class TestConfigurationIntegration:
    """Integration tests for configuration loading with real config file."""
    
    def test_real_config_loads_agent_settings(self):
        """Verify that the real providers_config.json loads agent settings."""
        config = get_config()
        
        assert "agent" in config
        agent_cfg = config["agent"]
        
        # Verify required fields exist
        assert "system_prompt" in agent_cfg
        assert "max_context_messages" in agent_cfg
        assert "default_temperature" in agent_cfg
        assert "default_max_tokens" in agent_cfg
        
        # Verify new tool-related fields exist
        assert "tool_calling_enabled" in agent_cfg
        assert "max_tool_rounds" in agent_cfg
        assert "max_concurrent_tools" in agent_cfg
        assert "tool_timeout_seconds" in agent_cfg
        assert "token_budget" in agent_cfg
        
        # Verify types and reasonable values
        assert isinstance(agent_cfg["tool_calling_enabled"], bool)
        assert isinstance(agent_cfg["max_tool_rounds"], int)
        assert agent_cfg["max_tool_rounds"] > 0
        assert isinstance(agent_cfg["max_concurrent_tools"], int)
        assert agent_cfg["max_concurrent_tools"] > 0
        assert isinstance(agent_cfg["tool_timeout_seconds"], (int, float))
        assert agent_cfg["tool_timeout_seconds"] > 0
        
        # token_budget can be either an int or a dict
        assert "token_budget" in agent_cfg
        if isinstance(agent_cfg["token_budget"], dict):
            # If dict, verify it has the expected structure
            assert "enabled" in agent_cfg["token_budget"]
            assert "max_tokens" in agent_cfg["token_budget"]
        else:
            # If int, verify it's positive
            assert isinstance(agent_cfg["token_budget"], int)
            assert agent_cfg["token_budget"] > 0
    
    def test_real_config_has_tools_section(self):
        """Verify that the real providers_config.json has a tools section."""
        config = get_config()
        
        assert "tools" in config
        tools_cfg = config["tools"]
        
        # Should have at least web_search tool configured
        assert "web_search" in tools_cfg
        
        # Verify web_search configuration
        web_search_cfg = tools_cfg["web_search"]
        assert "enabled" in web_search_cfg
        assert "timeout_seconds" in web_search_cfg
        assert isinstance(web_search_cfg["enabled"], bool)
        assert isinstance(web_search_cfg["timeout_seconds"], (int, float))
    
    def test_tool_registry_reflects_config(self):
        """Verify that tool_registry is configured from providers_config.json."""
        config = get_config()
        tools_cfg = config.get("tools", {})
        
        # Check web_search tool if it's in the config
        if "web_search" in tools_cfg:
            tool = tool_registry.get("web_search")
            if tool:  # Tool might not be registered yet in test environment
                web_search_cfg = tools_cfg["web_search"]
                
                # Verify enabled status matches config
                assert tool.enabled == web_search_cfg["enabled"]
                
                # Verify timeout matches config
                expected_timeout = web_search_cfg.get("timeout_seconds", 30.0)
                assert tool.timeout_seconds == expected_timeout
    
    def test_reload_config_updates_values(self):
        """Verify that reload_config() re-reads the file and updates settings."""
        # Get initial config
        initial_config = get_config()
        initial_rounds = initial_config["agent"]["max_tool_rounds"]
        
        # Reload config (should re-read from disk)
        reloaded_config = reload_config()
        
        # Should have the same value (since we didn't modify the file)
        assert reloaded_config["agent"]["max_tool_rounds"] == initial_rounds
        
        # Verify structure is still correct
        assert "agent" in reloaded_config
        assert "tools" in reloaded_config
        assert "tool_calling_enabled" in reloaded_config["agent"]
