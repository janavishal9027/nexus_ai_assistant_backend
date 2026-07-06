"""
Unit tests for web_search tool implementation.

Tests the web_search tool's registration, Tavily API integration,
DuckDuckGo fallback, and source extraction for citation tracking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.tool_registry import tool_registry
from app.tools.web_search import web_search, _tavily_search, _ddg_search


class TestWebSearchToolRegistration:
    """Test that the web_search tool is properly registered."""
    
    def test_tool_is_registered(self):
        """Verify the web_search tool exists in the registry."""
        tool = tool_registry.get("web_search")
        assert tool is not None
        assert tool.name == "web_search"
    
    def test_tool_has_correct_metadata(self):
        """Verify the tool has correct description and schemas."""
        tool = tool_registry.get("web_search")
        
        # Check description mentions real-time information
        assert "real-time" in tool.description.lower()
        
        # Check input schema has query parameter
        assert "query" in tool.input_schema["properties"]
        assert "max_results" in tool.input_schema["properties"]
        assert "query" in tool.input_schema["required"]
        
        # Check output schema
        assert "results" in tool.output_schema["properties"]
        assert "total_found" in tool.output_schema["properties"]
    
    def test_tool_has_correct_timeout(self):
        """Verify the tool has 15 second timeout as specified."""
        tool = tool_registry.get("web_search")
        assert tool.timeout_seconds == 15.0


class TestTavilySearch:
    """Test Tavily API search functionality."""
    
    @pytest.mark.asyncio
    async def test_tavily_search_success(self):
        """Test successful Tavily API search with results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Example Article",
                    "content": "This is an example article content.",
                    "url": "https://example.com/article"
                },
                {
                    "title": "Another Article",
                    "content": "More content here.",
                    "url": "https://example.com/another"
                }
            ]
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await _tavily_search("test query", "fake_api_key", 5)
        
        # Check structure
        assert "results" in result
        assert "total_found" in result
        assert "_sources" in result
        
        # Check results content
        assert len(result["results"]) == 2
        assert result["total_found"] == 2
        assert result["results"][0]["title"] == "Example Article"
        assert result["results"][0]["url"] == "https://example.com/article"
        
        # Check sources for citation tracking
        assert len(result["_sources"]) == 2
        assert result["_sources"][0]["url"] == "https://example.com/article"
        assert result["_sources"][0]["title"] == "Example Article"
    
    @pytest.mark.asyncio
    async def test_tavily_search_empty_results(self):
        """Test Tavily API with no results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await _tavily_search("test query", "fake_api_key", 5)
        
        assert result["results"] == []
        assert result["total_found"] == 0
        assert result["_sources"] == []
    
    @pytest.mark.asyncio
    async def test_tavily_search_network_error(self):
        """Test Tavily API network error handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.RequestError("Network error"))
            mock_client_class.return_value = mock_client
            
            with pytest.raises(httpx.RequestError):
                await _tavily_search("test query", "fake_api_key", 5)
    
    @pytest.mark.asyncio
    async def test_tavily_search_snippet_truncation(self):
        """Test that long content is truncated to 500 chars."""
        long_content = "x" * 1000
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Long Article",
                    "content": long_content,
                    "url": "https://example.com/long"
                }
            ]
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await _tavily_search("test query", "fake_api_key", 5)
        
        # Snippet should be truncated to 500 chars
        assert len(result["results"][0]["snippet"]) == 500


class TestDuckDuckGoSearch:
    """Test DuckDuckGo fallback search functionality."""
    
    @pytest.mark.asyncio
    async def test_ddg_search_success(self):
        """Test successful DuckDuckGo search with results."""
        mock_ddg_results = [
            {
                "title": "DDG Result 1",
                "body": "Body content 1",
                "href": "https://ddg.example.com/1"
            },
            {
                "title": "DDG Result 2",
                "body": "Body content 2",
                "href": "https://ddg.example.com/2"
            }
        ]
        
        # Mock the DDGS class at import time within the function
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value = mock_ddgs
        mock_ddgs.__exit__.return_value = None
        mock_ddgs.text.return_value = iter(mock_ddg_results)
        
        mock_ddgs_class = MagicMock(return_value=mock_ddgs)
        
        # Patch where DDGS is imported (inside the function)
        with patch("ddgs.DDGS", mock_ddgs_class):
            result = await _ddg_search("test query", 5)
        
        # Check structure
        assert "results" in result
        assert "total_found" in result
        assert "_sources" in result
        
        # Check results content
        assert len(result["results"]) == 2
        assert result["total_found"] == 2
        assert result["results"][0]["title"] == "DDG Result 1"
        assert result["results"][0]["url"] == "https://ddg.example.com/1"
        
        # Check sources for citation tracking
        assert len(result["_sources"]) == 2
    
    @pytest.mark.asyncio
    async def test_ddg_search_empty_results(self):
        """Test DuckDuckGo search with no results."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value = mock_ddgs
        mock_ddgs.__exit__.return_value = None
        mock_ddgs.text.return_value = iter([])
        
        mock_ddgs_class = MagicMock(return_value=mock_ddgs)
        
        with patch("ddgs.DDGS", mock_ddgs_class):
            result = await _ddg_search("test query", 5)
        
        assert result["results"] == []
        assert result["total_found"] == 0
        assert result["_sources"] == []
    
    @pytest.mark.asyncio
    async def test_ddg_search_import_error(self):
        """Test DuckDuckGo search when ddgs package is not installed."""
        # Simulate ImportError by preventing the import
        import sys
        original_import = __builtins__.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("No module named 'ddgs'")
            return original_import(name, *args, **kwargs)
        
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="ddgs package required"):
                await _ddg_search("test query", 5)


class TestWebSearchIntegration:
    """Test the main web_search function with provider fallback."""
    
    @pytest.mark.asyncio
    async def test_web_search_uses_tavily_when_key_present(self):
        """Test that Tavily is used when API key is configured."""
        with patch("app.tools.web_search.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = "test_key"
            
            with patch("app.tools.web_search._tavily_search") as mock_tavily:
                mock_tavily.return_value = {
                    "results": [{"title": "Test", "snippet": "Content", "url": "http://test.com"}],
                    "total_found": 1,
                    "_sources": [{"url": "http://test.com", "title": "Test"}]
                }
                
                result = await web_search("test query", 5)
        
        # Verify Tavily was called
        mock_tavily.assert_called_once()
        assert result["total_found"] == 1
    
    @pytest.mark.asyncio
    async def test_web_search_falls_back_to_ddg_when_no_tavily_key(self):
        """Test that DuckDuckGo is used when no Tavily key is configured."""
        with patch("app.tools.web_search.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = ""
            
            with patch("app.tools.web_search._ddg_search") as mock_ddg:
                mock_ddg.return_value = {
                    "results": [{"title": "DDG", "snippet": "Content", "url": "http://ddg.com"}],
                    "total_found": 1,
                    "_sources": [{"url": "http://ddg.com", "title": "DDG"}]
                }
                
                result = await web_search("test query", 5)
        
        # Verify DuckDuckGo was called
        mock_ddg.assert_called_once()
        assert result["total_found"] == 1
    
    @pytest.mark.asyncio
    async def test_web_search_falls_back_to_ddg_on_tavily_failure(self):
        """Test that DuckDuckGo is used when Tavily fails."""
        with patch("app.tools.web_search.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = "test_key"
            
            with patch("app.tools.web_search._tavily_search") as mock_tavily:
                mock_tavily.side_effect = Exception("Tavily API error")
                
                with patch("app.tools.web_search._ddg_search") as mock_ddg:
                    mock_ddg.return_value = {
                        "results": [{"title": "DDG", "snippet": "Content", "url": "http://ddg.com"}],
                        "total_found": 1,
                        "_sources": [{"url": "http://ddg.com", "title": "DDG"}]
                    }
                    
                    result = await web_search("test query", 5)
        
        # Verify fallback occurred
        mock_tavily.assert_called_once()
        mock_ddg.assert_called_once()
        assert result["total_found"] == 1
    
    @pytest.mark.asyncio
    async def test_web_search_raises_when_both_providers_fail(self):
        """Test that an exception is raised when both providers fail."""
        with patch("app.tools.web_search.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = "test_key"
            
            with patch("app.tools.web_search._tavily_search") as mock_tavily:
                mock_tavily.side_effect = Exception("Tavily failed")
                
                with patch("app.tools.web_search._ddg_search") as mock_ddg:
                    mock_ddg.side_effect = Exception("DuckDuckGo failed")
                    
                    with pytest.raises(RuntimeError, match="Web search unavailable"):
                        await web_search("test query", 5)
    
    @pytest.mark.asyncio
    async def test_web_search_respects_max_results_parameter(self):
        """Test that max_results parameter is passed to search providers."""
        with patch("app.tools.web_search.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = ""
            
            with patch("app.tools.web_search._ddg_search") as mock_ddg:
                mock_ddg.return_value = {
                    "results": [],
                    "total_found": 0,
                    "_sources": []
                }
                
                await web_search("test query", max_results=10)
        
        # Verify max_results was passed
        mock_ddg.assert_called_once_with("test query", 10)
