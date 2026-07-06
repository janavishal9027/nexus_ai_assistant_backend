"""Quick test script for web search functionality"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.services.web_search import needs_web_search, web_search


async def test_search():
    # Test 1: Detection
    test_queries = [
        "What's today's weather in madipakkam, chennai",
        "Hello, how are you?",  # Should not trigger
        "Latest news about AI",
        "Current Bitcoin price",
    ]
    
    print("=== Testing Web Search Detection ===")
    for query in test_queries:
        should_search = needs_web_search(query)
        print(f"{'✅' if should_search else '❌'} {query}")
    
    # Test 2: Actual search
    print("\n=== Testing Actual Web Search ===")
    query = "weather in chennai today"
    print(f"Searching for: {query}")
    result = await web_search(query, max_results=3)
    
    if result:
        print(f"\n✅ Search succeeded! ({len(result)} chars)")
        print("\n--- First 500 chars of result ---")
        print(result[:500])
        print("...")
    else:
        print("\n❌ Search failed or returned no results")


if __name__ == "__main__":
    asyncio.run(test_search())
