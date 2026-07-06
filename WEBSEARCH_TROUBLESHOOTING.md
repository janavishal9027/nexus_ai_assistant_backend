# Real-Time Web Search - Troubleshooting Guide

## ✅ What I Fixed

1. **Added explicit `web_search_enabled` flag** in `providers_config.json`
2. **Enhanced logging** - Now you can see exactly when web search triggers
3. **Improved error handling** - Better detection when search fails
4. **Strengthened system prompt** - LLM now forced to acknowledge and use search results
5. **Better DuckDuckGo integration** - More reliable with detailed logging

## 🧪 Test Web Search is Working

### Option 1: Run the test script
```bash
cd "v:\Project Worksapce\free llm api\chatapp\backend"
python test_websearch.py
```

You should see:
- ✅ Detection working for queries with real-time keywords
- ✅ Search succeeded with actual results

### Option 2: Check server logs
When you ask a real-time question through the app, watch the terminal. You should see:
```
INFO:app.services.agent:[Agent] Triggering web search for: What's today's weather...
INFO:app.services.web_search:[WebSearch] Starting search for: 'What's today's weather...'
INFO:app.services.web_search:[WebSearch] Using DuckDuckGo (no Tavily key)
INFO:app.services.web_search:[WebSearch/DDG] Searching for 'What's today's weather...', max=5
INFO:app.services.web_search:[WebSearch/DDG] Retrieved 5 results
INFO:app.services.web_search:[WebSearch/DDG] Formatted 2145 chars
INFO:app.services.agent:[Agent] Web search succeeded, 2145 chars of context
```

### Option 3: Test from your frontend
Ask these questions:
- ✅ "What's today's weather in Chennai?"
- ✅ "Latest news about AI"
- ✅ "Current Bitcoin price"
- ✅ "What happened this week?"

The AI should:
1. Provide current information (not training data)
2. Cite sources with URLs
3. NOT say "I don't have real-time access"

## 🔍 Troubleshooting

### Issue: AI still says it lacks real-time access

**Symptoms:**
- AI response: "I don't have access to real-time information..."
- No web search logs appear

**Check:**
1. Look for these logs in the server terminal:
   ```
   [Agent] Triggering web search for: ...
   ```
   
2. If you DON'T see that log, the query didn't trigger detection. Try adding more keywords:
   - "today", "current", "latest", "right now", "weather", "news", "price"

3. If you SEE the log but get no results:
   ```
   [WebSearch/DDG] No results found
   ```
   Try the test script to verify DuckDuckGo is accessible from your network

### Issue: DuckDuckGo blocked or slow

**Solution:** Get a free Tavily API key

1. Visit https://tavily.com and sign up (free, 1000 searches/month)
2. Add to `.env`:
   ```env
   TAVILY_API_KEY=tvly-your-key-here
   ```
3. Restart the backend - you'll see:
   ```
   [WebSearch] Using Tavily API
   ```

### Issue: Web search triggers but AI ignores results

**Check the system prompt is being enhanced:**
The logs should show search succeeded:
```
[Agent] Web search succeeded, XXXX chars of context
```

This means the search results ARE in the prompt. If the AI still ignores them:
1. Try a different model (some models follow instructions better)
2. The results might not contain relevant info - check the test script output
3. The LLM might be hallucinating - try asking more specifically

## 📊 Configuration Reference

### Trigger Keywords (in `web_search.py`)
These words in user queries auto-trigger web search:
```python
"today", "tonight", "right now", "currently", "current", "latest", 
"recent", "this week", "this month", "news", "weather", "price", 
"stock", "2024", "2025", "2026", "2027", etc.
```

### Agent Config (in `providers_config.json`)
```json
{
  "agent": {
    "web_search_enabled": true  // Set to false to disable
  }
}
```

### Search Providers (priority order)
1. **Tavily** (if `TAVILY_API_KEY` is set) - Best results, LLM-optimized
2. **DuckDuckGo** (fallback) - Free, no API key needed

## 🎯 Expected Behavior

### When Real-Time Question Asked:
1. ✅ System detects keywords
2. ✅ Performs web search (DDG or Tavily)
3. ✅ Injects results into system prompt
4. ✅ LLM uses results to answer with citations
5. ✅ User gets current information with source URLs

### When NOT a Real-Time Question:
1. ❌ No web search triggered (no overhead)
2. ✅ Normal LLM response from training data
3. ✅ Fast response time

## 📝 Log Monitoring

Watch for these patterns:

**✅ Success:**
```
[Agent] Triggering web search for: ...
[WebSearch] Starting search for: ...
[WebSearch/DDG] Retrieved X results
[Agent] Web search succeeded, XXXX chars of context
```

**⚠️ Warning (but not fatal):**
```
[WebSearch/DDG] No results found
[Agent] Web search returned no results
```

**❌ Error:**
```
[WebSearch] Error during search: ...
[WebSearch/DDG] Error during search: ...
```

## 🔧 Quick Fixes

### Restart the server:
1. Press CTRL+C in the terminal
2. Run: `python run.py`

### Reload config without restart:
The server has auto-reload enabled for code changes, but for config file changes, restart is recommended.

### Verify dependencies:
```bash
pip install ddgs httpx
```

### Check network access:
```bash
python test_websearch.py
```

---

## Need More Help?

Check the logs in detail - they now show every step of the web search process!
