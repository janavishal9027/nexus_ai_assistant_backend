"""Test the chat API with web search directly"""
import httpx
import asyncio
import json


async def test_chat_api():
    """Send a test message that should trigger web search"""
    
    # Test message that should trigger web search
    payload = {
        "message": "What's today's weather in madipakkam, chennai?",
        "model": None,  # Use default
        "conversation_id": None,  # New conversation
        "history": [],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    print("=" * 60)
    print("Testing Chat API with Real-Time Query")
    print("=" * 60)
    print(f"\nQuery: {payload['message']}")
    print("\nSending request to http://localhost:8080/api/chat/send ...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:8080/api/chat/send",
                json=payload
            )
            
            print(f"\nStatus: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n" + "=" * 60)
                print("RESPONSE:")
                print("=" * 60)
                print(f"\nConversation ID: {data.get('conversation_id')}")
                print(f"Model: {data.get('model')}")
                print(f"Platform: {data.get('platform')}")
                print(f"\nContent:\n{data.get('content')}")
                print("\n" + "=" * 60)
                
                # Check if response indicates real-time access
                content_lower = data.get('content', '').lower()
                if 'not able to provide real-time' in content_lower or \
                   "don't have access" in content_lower or \
                   "can't access" in content_lower:
                    print("\n❌ WARNING: LLM claims no real-time access!")
                    print("Check server logs for web search activity.")
                elif any(word in content_lower for word in ['°c', '°f', 'temperature', 'humidity', 'forecast']):
                    print("\n✅ SUCCESS: Response contains weather data!")
                else:
                    print("\n⚠️ UNCLEAR: Response doesn't clearly show real-time data")
                    
            else:
                print(f"\n❌ Error: {response.text}")
                
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        print("\nMake sure the backend is running on http://localhost:8080")


if __name__ == "__main__":
    print("\nIMPORTANT: Watch the server terminal for these logs:")
    print("  [Agent] Triggering web search for: ...")
    print("  [WebSearch/DDG] Retrieved X results")
    print("  [Agent] Web search succeeded, XXXX chars of context\n")
    
    asyncio.run(test_chat_api())
