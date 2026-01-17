"""
Tool utilities for the Grok bot.

Note: Tools are now registered directly on the Pydantic AI agent in bot/ai.py.
This module provides standalone utility functions if needed elsewhere.
"""

from datetime import datetime
from logging import getLogger
from typing import Optional

import httpx

from .models import Config

logger = getLogger(__name__)


def current_datetime() -> str:
    """Gets current date and time."""
    return str(datetime.now())


def search_web(config: Config, query: str) -> Optional[str]:
    """
    Search the web using SearXNG.

    Args:
        config: Bot configuration containing SearXNG settings.
        query: Search query string.

    Returns:
        Search results as a string, or None if the search failed.
    """
    auth: Optional[httpx.BasicAuth] = None
    if config.searxng_user and config.searxng_password:
        auth = httpx.BasicAuth(config.searxng_user, config.searxng_password)
    transport = httpx.HTTPTransport(retries=config.max_retries)
    with httpx.Client(auth=auth, transport=transport) as client:
        try:
            response = client.post(
                f"{config.searxng_url}search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            data = response.json()
            return "\n---\n".join(
                [result.get("content") for result in data.get("results", [])[:5]]
            )
        except httpx.HTTPError:
            logger.exception("HTTP Error during web search")
            return None
