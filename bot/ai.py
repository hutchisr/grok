import asyncio
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from pydantic_ai.models.fallback import FallbackModel
import logfire

from .models import Config, Note, MiFile
from .tools import build_tools


class ReplyOutput(BaseModel):
    """Output schema for the chat agent."""

    reply: str
    """Reply to the message. Must NOT include mentions or usernames. Must not be None."""
    mentions: Optional[list[str]] = None
    """List of usernames mentioned in the message (without @ prefix)."""


class ChatAgent:
    def __init__(self, config: Config):
        self._config = config

        # Create fallback model from all configured models
        if len(config.llm_models) == 1:
            model = config.llm_models[0]
        else:
            model = FallbackModel(*config.llm_models)

        # Create vision fallback model
        if len(config.vision_models) == 1:
            self._vision_model = config.vision_models[0]
        else:
            self._vision_model = FallbackModel(*config.vision_models)

        self._agent: Agent[None, ReplyOutput] = Agent(
            model,
            output_type=ReplyOutput,
            instructions=config.system_prompt,
            tools=build_tools(config),
        )

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

        # Run with fallback model
        result = await self._agent.run(user_prompt, message_history=message_history)
        logfire.info(f"Reply: {result.output}")
        return result.output

    async def describe_images(self, files: list[MiFile]) -> Optional[list[str]]:
        """Describe images using vision models."""
        image_urls: list[str] = []
        for f in files:
            logfire.info(f"Looking at file: {f.id} ({f.type}): {f.thumbnailUrl}")
            if f.thumbnailUrl and f.type.startswith("image/"):
                image_urls.append(f.thumbnailUrl)

        if not image_urls:
            return None

        # Use vision model with fallback
        try:
            vision_agent: Agent[None, str] = Agent(
                self._vision_model,
                output_type=str,
                instructions="Describe the images provided in a concise way.",
            )

            # Build prompt with ImageUrl objects for proper multimodal input
            prompt: list[str | ImageUrl] = ["Describe these images:"]
            for url in image_urls:
                prompt.append(ImageUrl(url=url))

            result = await vision_agent.run(prompt)
            descriptions = [result.output]
            logfire.info(f"Image descriptions: {descriptions}")
            return descriptions

        except Exception:
            logfire.exception("Error occurred while describing images")
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
