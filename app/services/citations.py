"""
Citation Tracking for Agent Tool System.

Extracts source URLs from tool results, deduplicates by URL, and formats
citations as numbered Markdown references for appending to responses.

Validates Requirements: 8.1, 8.2, 8.4, 8.6, 8.7, 8.8
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    """A single citation extracted from a tool result."""
    url: str
    title: str
    tool_name: str


class CitationTracker:
    """
    Tracks and formats citations from tool execution results.
    
    Maintains a dictionary of unique citations keyed by URL. Provides
    methods to ingest tool results and format the accumulated citations
    as a numbered Markdown list.
    """

    def __init__(self):
        """Initialize an empty citation tracker."""
        self.citations: dict[str, Citation] = {}  # url -> Citation

    def ingest(self, results: list[Any]) -> None:
        """
        Extract citations from tool results. Deduplicates by URL.
        
        Handles ToolResult entries that may or may not have a 'sources' field.
        Preserves the tool_name that provided each citation.
        
        Args:
            results: List of ToolResult objects (must have attributes: status, 
                     tool_name, and optionally sources)
        
        Requirements:
            - 8.1: Extract source URLs from all Tool_Results
            - 8.6: Deduplicate citations by URL
            - 8.7: Preserve original tool name for each citation
            - 8.8: Handle Tool_Results without source URLs without error
        """
        for result in results:
            # Only process successful results
            if not hasattr(result, 'status') or result.status != "success":
                continue
            
            # Handle missing sources field gracefully (Req 8.8)
            if not hasattr(result, 'sources') or result.sources is None:
                continue
            
            # Extract citations from sources
            for source in result.sources:
                url = source.get("url")
                if url and url not in self.citations:
                    # Deduplicate by URL (Req 8.6)
                    self.citations[url] = Citation(
                        url=url,
                        title=source.get("title", url),
                        tool_name=result.tool_name  # Req 8.7
                    )

    def format_citations(self) -> str:
        """
        Format citations as numbered references in Markdown.
        
        Returns an empty string if no citations were collected.
        
        Returns:
            Markdown-formatted string with numbered list of citations,
            or empty string if no citations exist.
        
        Requirements:
            - 8.2: Maintain list of unique citations during conversation turn
            - 8.4: Format citations as numbered references with title and URL
        """
        if not self.citations:
            return ""

        lines = ["## Sources"]
        for i, (url, citation) in enumerate(self.citations.items(), start=1):
            lines.append(f"{i}. [{citation.title}]({url})")

        return "\n".join(lines)

    def get_count(self) -> int:
        """
        Get the number of unique citations tracked.
        
        Returns:
            Count of unique citations (by URL)
        """
        return len(self.citations)
