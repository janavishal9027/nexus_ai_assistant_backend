"""
Unit tests for citations.py

Tests the Citation dataclass and CitationTracker class to ensure proper
extraction, deduplication, and formatting of citations from tool results.

Validates Requirements: 8.1, 8.2, 8.4, 8.6, 8.7, 8.8
"""
import pytest
from dataclasses import dataclass
from typing import Any

from app.services.citations import Citation, CitationTracker


# Mock ToolResult for testing
@dataclass
class MockToolResult:
    """Mock ToolResult object for testing."""
    tool_name: str
    status: str
    sources: list[dict] | None = None
    data: Any = None
    error_message: str | None = None


class TestCitationDataclass:
    """Tests for Citation dataclass."""
    
    def test_citation_initialization(self):
        """Test Citation dataclass initialization."""
        citation = Citation(
            url="https://example.com",
            title="Example Article",
            tool_name="web_search"
        )
        assert citation.url == "https://example.com"
        assert citation.title == "Example Article"
        assert citation.tool_name == "web_search"


class TestCitationTrackerInitialization:
    """Tests for CitationTracker initialization."""
    
    def test_init_empty_citations(self):
        """Test CitationTracker initializes with empty citations."""
        tracker = CitationTracker()
        assert len(tracker.citations) == 0
        assert tracker.get_count() == 0


class TestCitationIngestion:
    """Tests for citation ingestion from tool results."""
    
    def test_ingest_single_source(self):
        """Test ingesting a single tool result with one source (Req 8.1)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example.com", "title": "Example Article"}
                ]
            )
        ]
        
        tracker.ingest(results)
        assert tracker.get_count() == 1
        assert "https://example.com" in tracker.citations
        assert tracker.citations["https://example.com"].title == "Example Article"
        assert tracker.citations["https://example.com"].tool_name == "web_search"
    
    def test_ingest_multiple_sources(self):
        """Test ingesting a tool result with multiple sources (Req 8.1)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example1.com", "title": "Article 1"},
                    {"url": "https://example2.com", "title": "Article 2"},
                    {"url": "https://example3.com", "title": "Article 3"}
                ]
            )
        ]
        
        tracker.ingest(results)
        assert tracker.get_count() == 3
        assert "https://example1.com" in tracker.citations
        assert "https://example2.com" in tracker.citations
        assert "https://example3.com" in tracker.citations
    
    def test_deduplication_by_url(self):
        """Test that duplicate URLs are deduplicated (Req 8.6)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example.com", "title": "First Title"}
                ]
            ),
            MockToolResult(
                tool_name="api_tool",
                status="success",
                sources=[
                    {"url": "https://example.com", "title": "Second Title"}
                ]
            )
        ]
        
        tracker.ingest(results)
        # Should only have 1 citation despite 2 results with same URL
        assert tracker.get_count() == 1
        # First occurrence should be kept
        assert tracker.citations["https://example.com"].title == "First Title"
        assert tracker.citations["https://example.com"].tool_name == "web_search"
    
    def test_multiple_ingests_accumulate(self):
        """Test multiple ingest calls accumulate citations (Req 8.2)."""
        tracker = CitationTracker()
        
        # First ingest
        tracker.ingest([
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example1.com", "title": "Article 1"}
                ]
            )
        ])
        assert tracker.get_count() == 1
        
        # Second ingest
        tracker.ingest([
            MockToolResult(
                tool_name="news_api",
                status="success",
                sources=[
                    {"url": "https://example2.com", "title": "Article 2"}
                ]
            )
        ])
        assert tracker.get_count() == 2


class TestToolNamePreservation:
    """Tests for preserving tool name in citations."""
    
    def test_preserve_tool_name(self):
        """Test that tool_name is preserved for each citation (Req 8.7)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example1.com", "title": "Article 1"}
                ]
            ),
            MockToolResult(
                tool_name="news_api",
                status="success",
                sources=[
                    {"url": "https://example2.com", "title": "Article 2"}
                ]
            )
        ]
        
        tracker.ingest(results)
        assert tracker.citations["https://example1.com"].tool_name == "web_search"
        assert tracker.citations["https://example2.com"].tool_name == "news_api"


class TestMissingOrEmptySources:
    """Tests for handling missing or empty source fields."""
    
    def test_handle_missing_sources_field(self):
        """Test handling ToolResult without sources field (Req 8.8)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="calculation_tool",
                status="success",
                sources=None  # No sources field
            )
        ]
        
        # Should not raise an error
        tracker.ingest(results)
        assert tracker.get_count() == 0
    
    def test_handle_empty_sources_list(self):
        """Test handling ToolResult with empty sources list (Req 8.8)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[]  # Empty sources
            )
        ]
        
        tracker.ingest(results)
        assert tracker.get_count() == 0
    
    def test_handle_missing_url(self):
        """Test that sources without URL are skipped (Req 8.8)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"title": "Article Without URL"}  # No URL
                ]
            )
        ]
        
        tracker.ingest(results)
        assert tracker.get_count() == 0


class TestErrorAndTimeoutHandling:
    """Tests for handling error and timeout results."""
    
    def test_handle_error_status(self):
        """Test that error results are not processed (Req 8.8)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="error",
                error_message="Network failure",
                sources=[
                    {"url": "https://example.com", "title": "Article"}
                ]
            )
        ]
        
        tracker.ingest(results)
        # Error results should not contribute citations
        assert tracker.get_count() == 0
    
    def test_handle_timeout_status(self):
        """Test that timeout results are not processed (Req 8.8)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="timeout",
                error_message="Timeout exceeded",
                sources=[
                    {"url": "https://example.com", "title": "Article"}
                ]
            )
        ]
        
        tracker.ingest(results)
        # Timeout results should not contribute citations
        assert tracker.get_count() == 0


class TestCitationFormatting:
    """Tests for citation formatting."""
    
    def test_format_citations_empty(self):
        """Test formatting when no citations exist."""
        tracker = CitationTracker()
        assert tracker.format_citations() == ""
    
    def test_format_citations_single(self):
        """Test formatting a single citation (Req 8.4)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example.com", "title": "Example Article"}
                ]
            )
        ]
        
        tracker.ingest(results)
        output = tracker.format_citations()
        
        assert "## Sources" in output
        assert "1. [Example Article](https://example.com)" in output
    
    def test_format_citations_multiple(self):
        """Test formatting multiple citations with proper numbering (Req 8.4)."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example1.com", "title": "Article 1"},
                    {"url": "https://example2.com", "title": "Article 2"},
                    {"url": "https://example3.com", "title": "Article 3"}
                ]
            )
        ]
        
        tracker.ingest(results)
        output = tracker.format_citations()
        
        assert "## Sources" in output
        assert "1. [Article 1](https://example1.com)" in output
        assert "2. [Article 2](https://example2.com)" in output
        assert "3. [Article 3](https://example3.com)" in output


class TestEdgeCases:
    """Tests for edge cases and data quality."""
    
    def test_handle_missing_title(self):
        """Test that missing title defaults to URL."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example.com"}  # No title
                ]
            )
        ]
        
        tracker.ingest(results)
        citation = tracker.citations["https://example.com"]
        assert citation.title == "https://example.com"
    
    def test_mixed_results(self):
        """Test ingesting a mix of success, error, and results without sources."""
        tracker = CitationTracker()
        results = [
            MockToolResult(
                tool_name="web_search",
                status="success",
                sources=[
                    {"url": "https://example1.com", "title": "Article 1"}
                ]
            ),
            MockToolResult(
                tool_name="calculation",
                status="success",
                sources=None  # No sources
            ),
            MockToolResult(
                tool_name="broken_api",
                status="error",
                error_message="API failure",
                sources=[
                    {"url": "https://example2.com", "title": "Article 2"}
                ]
            ),
            MockToolResult(
                tool_name="news_api",
                status="success",
                sources=[
                    {"url": "https://example3.com", "title": "Article 3"}
                ]
            )
        ]
        
        tracker.ingest(results)
        # Should only have 2 citations (from successful results with sources)
        assert tracker.get_count() == 2
        assert "https://example1.com" in tracker.citations
        assert "https://example3.com" in tracker.citations
        assert "https://example2.com" not in tracker.citations  # Error result excluded
