from __future__ import annotations

import json

from fastmcp import FastMCP

import web_tools


mcp = FastMCP("Local Web Tools")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return JSON.

    Returns a JSON array of objects: [{"title": "...", "url": "...", "snippet": "..."}].
    """
    results = web_tools.web_search(query, max_results=max_results)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def fetch_url(url: str, max_chars: int = 8_000) -> str:
    """Fetch a URL and return readable text (truncated)."""
    return web_tools.fetch_url(url, max_chars=max_chars)


if __name__ == "__main__":
    mcp.run()

