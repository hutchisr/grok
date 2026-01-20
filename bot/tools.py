"""Tool utilities for the Grok bot."""

from collections.abc import Callable
from datetime import datetime
from typing import Optional

import httpx
import logfire

from .models import Config


def current_datetime() -> str:
    """Gets current date and time."""
    return str(datetime.now())


def build_tools(config: Config) -> list[Callable[..., object]]:
    """Create tool functions for the given config.

    Tools are returned as plain functions and can be passed to Agent(..., tools=...).
    """
    tools: list[Callable[..., object]] = []

    def current_datetime_tool() -> str:
        """Gets current date and time."""
        return current_datetime()

    tools.append(current_datetime_tool)

    if config.searxng_url:

        def search_web(query: str) -> Optional[str]:
            """Search the web for information."""
            with logfire.span("search web", query=query):
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
                            [
                                result.get("content")
                                for result in data.get("results", [])[:5]
                            ]
                        )
                    except httpx.HTTPError:
                        logfire.exception("HTTP Error during web search")
                        return None

        tools.append(search_web)

    def create_note(
        text: str,
        visibility: str = "public",
        local_only: bool = False,
        mentions: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Create a new note/post on this Misskey instance.

        Args:
            text: The note content (plain text)
            visibility: public | home | followers | specified
            local_only: If True, only deliver to local instance
            mentions: Optional list of usernames/handles to mention (with or without @)
        """
        with logfire.span(
            "create note",
            visibility=visibility,
            local_only=local_only,
        ):
            if not text.strip():
                return "Note text is empty."

            if visibility not in {"public", "home", "followers", "specified"}:
                return "Invalid visibility. Use public, home, followers, or specified."

            # Normalize mentions and prefix to text if missing
            mention_prefix = ""
            if mentions:
                normalized: list[str] = []
                seen = set()
                for m in mentions:
                    if not m:
                        continue
                    handle = m.strip()
                    if not handle:
                        continue
                    if not handle.startswith("@"):
                        handle = f"@{handle}"
                    key = handle.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    normalized.append(handle)
                if normalized:
                    # Only prefix handles that are not already present in text
                    text_lower = text.lower()
                    to_prefix = [h for h in normalized if h.lower() not in text_lower]
                    if to_prefix:
                        mention_prefix = " ".join(to_prefix) + " "

            transport = httpx.HTTPTransport(retries=config.max_retries)
            with httpx.Client(transport=transport) as client:
                try:
                    payload = {
                        "text": f"{mention_prefix}{text}",
                        "visibility": visibility,
                        "localOnly": local_only,
                    }
                    response = client.post(
                        f"{config.url}api/notes/create",
                        json=payload,
                        headers={"Authorization": f"Bearer {config.token}"},
                    )
                    response.raise_for_status()
                    created = response.json().get("createdNote", {})
                    note_id = created.get("id")
                    return (
                        f"Created note {note_id}." if note_id else "Note created."
                    )
                except httpx.HTTPError:
                    logfire.exception("HTTP Error during note creation")
                    return None

    def search_users(query: str, limit: int = 10, offset: int = 0) -> Optional[str]:
        """Search for users on this Misskey instance by username or display name.

        Args:
            query: The search query.
            limit: Maximum number of results to return (1-50, default 10)
            offset: Number of results to skip for pagination (default 0)
        """
        with logfire.span("search users", query=query, limit=limit, offset=offset):
            limit = max(1, min(50, limit))  # Clamp to 1-50
            transport = httpx.HTTPTransport(retries=config.max_retries)
            with httpx.Client(
                transport=transport,
                timeout=httpx.Timeout(config.http_timeout_seconds),
            ) as client:
                try:
                    response = client.post(
                        f"{config.url}api/users/search",
                        json={"query": query, "limit": limit, "offset": offset},
                        headers={"Authorization": f"Bearer {config.token}"},
                    )
                    response.raise_for_status()
                    users = response.json()
                    if not users:
                        return "No users found."
                    results = []
                    for user in users:
                        username = user.get("username", "unknown")
                        host = user.get("host")
                        name = user.get("name") or username
                        bio = user.get("description") or ""
                        handle = f"@{username}" + (f"@{host}" if host else "")
                        results.append(f"{name} ({handle}): {bio[:100]}")
                    return "\n---\n".join(results)
                except httpx.HTTPError:
                    logfire.exception("HTTP Error during user search")
                    return None

    def search_notes(query: str, limit: int = 10, offset: int = 0) -> Optional[str]:
        """Search for notes/posts on this Misskey instance.

        Args:
            query: The search query. Simple text search on note content.
            limit: Maximum number of results to return (1-50, default 10)
            offset: Number of results to skip for pagination (default 0)
        """
        with logfire.span("search notes", query=query, limit=limit, offset=offset):
            limit = max(1, min(50, limit))  # Clamp to 1-50
            transport = httpx.HTTPTransport(retries=config.max_retries)
            with httpx.Client(
                transport=transport,
                timeout=httpx.Timeout(config.http_timeout_seconds),
            ) as client:
                try:
                    response = client.post(
                        f"{config.url}api/notes/search",
                        json={"query": query, "limit": limit, "offset": offset},
                        headers={"Authorization": f"Bearer {config.token}"},
                    )
                    response.raise_for_status()
                    notes = response.json()
                    if not notes:
                        return "No notes found."
                    results = []
                    for note in notes:
                        user = note.get("user", {})
                        username = user.get("username", "unknown")
                        host = user.get("host")
                        handle = f"@{username}" + (f"@{host}" if host else "")
                        text = note.get("text") or "(no text)"
                        results.append(f"{handle}: {text[:200]}")
                    return "\n---\n".join(results)
                except httpx.HTTPError:
                    logfire.exception("HTTP Error during note search")
                    return None

    tools.extend([search_users, search_notes])
    return tools
