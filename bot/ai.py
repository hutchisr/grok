import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl, RunContext
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
import logfire
from redis.asyncio import Redis

from .models import Config, Note, User
from .tools import build_tools


def _user_handle(user: User) -> str:
    """Get full handle: username for local, username@host for remote."""
    if user.host:
        return f"{user.username}@{user.host}"
    return user.username


def _image_urls_for(note: Note, vision: bool) -> list[ImageUrl]:
    """Extract ImageUrl objects for a note's image attachments."""
    if not vision or not note.files:
        return []
    return [ImageUrl(url=f.thumbnailUrl) for f in note.files if f.thumbnailUrl and f.type.startswith("image/")]


def _build_user_content(note: Note, vision: bool) -> str | list[str | ImageUrl]:
    """Build content for a user prompt part, with optional images."""
    text = f"{_user_handle(note.user)}: {note.text or ''}"
    images = _image_urls_for(note, vision)
    if images:
        return [text, *images]
    return text


class ReplyOutput(BaseModel):
    """Output schema for the chat agent."""

    reply: str
    """Reply to the message. Must NOT include mentions or usernames. Must not be None."""


@dataclass
class AgentDeps:
    """Runtime dependencies passed to the agent on each run."""

    username: str
    """The handle of the user who sent the message."""
    social_credit_score: Optional[int] = None
    """The user's current social credit score, or None if unavailable."""


class ChatAgent:
    def __init__(self, config: Config, redis_client: Optional[Redis] = None):
        self._config = config
        self._redis = redis_client

        fallback_on = (ModelAPIError, httpx.TimeoutException)

        # Create fallback model from all configured models
        if len(config.llm_models) == 1:
            model = config.llm_models[0]
        else:
            model = FallbackModel(*config.llm_models, fallback_on=fallback_on)

        tools = build_tools(config, redis_client=redis_client)

        async def _inject_social_credit(ctx: RunContext[AgentDeps]) -> str:
            parts: list[str] = []
            parts.append(f"Current user: @{ctx.deps.username}")
            if ctx.deps.social_credit_score is not None:
                parts.append(f"Current user's social credit score: {ctx.deps.social_credit_score}")
            else:
                parts.append("Current user's social credit score: 0 (no score recorded yet)")
            return "\n".join(parts)

        self._agent: Agent[AgentDeps, ReplyOutput] = Agent(
            model,
            output_type=ReplyOutput,
            deps_type=AgentDeps,
            instructions=[config.system_prompt, _inject_social_credit],
            tools=tools,
            retries=3,
        )

        self._auto_agent: Optional[Agent[Any, ReplyOutput]] = None
        if config.system_prompt_auto:
            self._auto_agent = Agent(
                model,
                output_type=ReplyOutput,
                instructions=[config.system_prompt_auto],
                tools=tools,
                retries=3,
            )

    async def run(self, note: Note, context: Optional[list[Note]] = None) -> ReplyOutput:
        """Process a note and generate a reply."""
        if not note.text:
            raise ValueError("Note text is empty")

        bot_user_id = self._config.bot_user_id
        vision = self._config.vision

        # Build message history from context notes (oldest first)
        message_history: list[ModelMessage] = []
        if context:
            for c in reversed(context):
                if c.userId == bot_user_id:
                    # Bot's own previous messages become assistant responses
                    message_history.append(ModelResponse(parts=[TextPart(content=c.text or "")]))
                else:
                    # Other users' messages become user prompts (with any attached images)
                    message_history.append(ModelRequest(parts=[UserPromptPart(content=_build_user_content(c, vision))]))

        # Build current user prompt
        current_parts: list[str | ImageUrl] = []
        if note.user.location:
            current_parts.append(f"User location: {note.user.location}")
        current_parts.append(f"{_user_handle(note.user)}: {note.text}")
        current_images = _image_urls_for(note, vision)
        if current_images:
            current_parts.extend(current_images)

        prompt: str | list[str | ImageUrl]
        if len(current_parts) == 1 and isinstance(current_parts[0], str):
            prompt = current_parts[0]
        else:
            prompt = current_parts

        # Pre-fetch social credit score for the current user
        handle = _user_handle(note.user)
        score = await self._get_social_credit_score(handle)
        deps = AgentDeps(username=handle, social_credit_score=score)

        result = await self._agent.run(
            prompt, deps=deps, message_history=message_history, model_settings={"timeout": 300.0}
        )
        logfire.info(f"Reply: {result.output}")
        return result.output

    async def run_auto(self) -> ReplyOutput:
        """Generate an autonomous post with no user input."""
        if not self._auto_agent:
            raise ValueError("No system_prompt_auto configured")
        result = await self._auto_agent.run(
            "Generate a post for the timeline.",
            model_settings={"timeout": 300.0},
        )
        logfire.info(f"Autonomous post: {result.output}")
        return result.output

    async def _get_social_credit_score(self, username: str) -> Optional[int]:
        """Fetch the user's social credit score from Redis."""
        if not self._redis:
            return None
        # Normalize: lowercase, strip @
        key = username.strip().lower()
        if key.startswith("@"):
            key = key[1:]
        try:
            raw = await self._redis.get(f"score:{key}")
            if raw is None:
                return None
            return int(raw)
        except Exception:
            logfire.exception("Error pre-fetching social credit score")
            return None

    def run_sync(self, *args, **kwargs) -> ReplyOutput:
        """Sync wrapper for run - for compatibility with notebooks."""

        def _run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.run(*args, **kwargs))
            finally:
                loop.close()

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_async)
            return future.result()
