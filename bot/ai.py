import asyncio
from logging import getLogger
from typing import Optional

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from .models import Config, Note, MiFile


logger = getLogger(__name__)


class ReplyOutput(BaseModel):
    """Output schema for the chat agent."""

    reply: str
    """Reply to the message. Must NOT include mentions or usernames. Must not be None."""
    mentions: Optional[list[str]] = None
    """List of usernames mentioned in the message (without @ prefix)."""


class ChatAgent:
    def __init__(self, config: Config):
        self._config = config
        self._agents: list[tuple[str, Agent[None, ReplyOutput]]] = []
        self._vision_models = config.vision_models

        # Create an agent for each model for fallback support
        for model in config.llm_models:
            agent: Agent[None, ReplyOutput] = Agent(
                model,
                output_type=ReplyOutput,
                instructions=config.system_prompt,
            )

            # Register tools
            @agent.tool_plain
            def current_datetime() -> str:
                """Gets current date and time"""
                from datetime import datetime

                return str(datetime.now())

            if config.searxng_url:
                self._register_web_search(agent, config)

            self._agents.append((model, agent))

    def _register_web_search(self, agent: Agent[None, ReplyOutput], config: Config):
        """Register web search tool on an agent."""

        @agent.tool_plain
        def search_web(query: str) -> Optional[str]:
            """Search the web for information."""
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
                    logger.exception("HTTP Error during web search")
                    return None

    async def run(
        self, note: Note, context: Optional[list[Note]] = None
    ) -> ReplyOutput:
        """Process a note and generate a reply."""
        if not note.text:
            raise ValueError("Note text is empty")

        # Collect files from context and current note
        files: list[MiFile] = []
        if context:
            for n in context:
                if n.files:
                    files.extend(n.files)
        if note.files:
            files.extend(note.files)

        logger.info(f"Checking {len(files)} file(s)")

        # Describe images if present
        descriptions = None
        if files:
            descriptions = await self.describe_images(files)

        # Build message history from context
        message_history: list[ModelRequest | ModelResponse] = []
        if context:
            # Context is in reverse order (newest first), so reverse it
            for c in reversed(context):
                if c.text:
                    # Determine if this is a bot message or user message
                    is_bot = c.userId == self._config.bot_user_id
                    if is_bot:
                        # Bot's previous response
                        message_history.append(
                            ModelResponse(parts=[TextPart(content=c.text)])
                        )
                    else:
                        # User message
                        message_history.append(
                            ModelRequest(parts=[UserPromptPart(content=f"{c.user.username}: {c.text}")])
                        )

        # Build the current prompt
        prompt_parts = []
        if descriptions:
            prompt_parts.append(f"Image descriptions: {', '.join(descriptions)}")
        if note.user.location:
            prompt_parts.append(f"User location: {note.user.location}")
        prompt_parts.append(f"{note.user.username}: {note.text}")

        user_prompt = "\n".join(prompt_parts)

        # Try each model until one succeeds
        last_error = None
        for model, agent in self._agents:
            try:
                result = await agent.run(user_prompt, message_history=message_history)
                logger.info(f"Reply: {result.output}")
                return result.output
            except Exception as e:
                logger.exception(
                    f"Error occurred while processing message with {model}. Will try next LLM"
                )
                last_error = e
                await asyncio.sleep(1)
                continue

        raise RuntimeError(f"All LLM models failed. Last error: {last_error}")

    async def describe_images(self, files: list[MiFile]) -> Optional[list[str]]:
        """Describe images using vision models."""
        image_urls: list[str] = []
        for f in files:
            logger.info(f"Looking at file: {f.id} ({f.type}): {f.thumbnailUrl}")
            if f.thumbnailUrl and f.type.startswith("image/"):
                image_urls.append(f.thumbnailUrl)

        if not image_urls:
            return None

        # Try each vision model
        for model in self._vision_models:
            try:
                vision_agent: Agent[None, str] = Agent(
                    model,
                    output_type=str,
                    instructions="Describe the images provided in a concise way.",
                )

                # Build prompt with ImageUrl objects for proper multimodal input
                prompt: list[str | ImageUrl] = ["Describe these images:"]
                for url in image_urls:
                    prompt.append(ImageUrl(url=url))

                result = await vision_agent.run(prompt)
                descriptions = [result.output]
                logger.info(f"Image descriptions: {descriptions}")
                return descriptions

            except Exception:
                logger.exception(
                    f"Error occurred while describing images with {model}. Will try next vision model"
                )
                await asyncio.sleep(1)
                continue

        logger.warning("All vision models failed, returning None")
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
