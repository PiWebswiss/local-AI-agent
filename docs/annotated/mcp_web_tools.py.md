# mcp_web_tools.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: FastMCP server exposing `web_search` and `fetch_url` tools over stdio.

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import json
      ↳ Imports standard library modules: json.
  4 
      ↳ Blank line for readability.
  5 from fastmcp import FastMCP
      ↳ Imports FastMCP from the third-party module `fastmcp`.
  6 
      ↳ Blank line for readability.
  7 import web_tools
      ↳ Imports local project modules: web_tools.
  8 
      ↳ Blank line for readability.
  9 
      ↳ Blank line for readability.
 10 mcp = FastMCP("Local Web Tools")
      ↳ Assignment: sets `mcp`.
 11 
      ↳ Blank line for readability.
 12 
      ↳ Blank line for readability.
 13 @mcp.tool()
      ↳ Decorator line: modifies the behavior of the next function/method.
 14 def web_search(query: str, max_results: int = 5) -> str:
      ↳ Defines function `web_search()`.
 15     """
      ↳ Implementation detail: part of the surrounding logic.
 16     Search the web and return JSON.
      ↳ Implementation detail: part of the surrounding logic.
 17 
      ↳ Blank line for readability.
 18     Returns a JSON array of objects: [{"title": "...", "url": "...", "snippet": "..."}].
      ↳ Implementation detail: part of the surrounding logic.
 19     """
      ↳ Implementation detail: part of the surrounding logic.
 20     results = web_tools.web_search(query, max_results=max_results)
      ↳ Assignment: sets `results`.
 21     return json.dumps(results, ensure_ascii=False)
      ↳ Returns a value from the current function.
 22 
      ↳ Blank line for readability.
 23 
      ↳ Blank line for readability.
 24 @mcp.tool()
      ↳ Decorator line: modifies the behavior of the next function/method.
 25 def fetch_url(url: str, max_chars: int = 8_000) -> str:
      ↳ Defines function `fetch_url()`.
 26     """Fetch a URL and return readable text (truncated)."""
      ↳ Implementation detail: part of the surrounding logic.
 27     return web_tools.fetch_url(url, max_chars=max_chars)
      ↳ Returns a value from the current function.
 28 
      ↳ Blank line for readability.
 29 
      ↳ Blank line for readability.
 30 if __name__ == "__main__":
      ↳ Conditional branch: checks a condition and chooses a code path.
 31     mcp.run()
      ↳ Implementation detail: part of the surrounding logic.
 32 
      ↳ Blank line for readability.
```
