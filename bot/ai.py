import asyncio
import inspect
import json
from typing import Callable, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl
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


class ReActStep(BaseModel):
    """Single ReAct step output."""

    action: str
    """Tool name to call, or 'final'."""
    action_input: Optional[str] = None
    """Tool input. Can be plain text or JSON string for args."""
    final: Optional[str] = None
    """Final reply when action == 'final'."""


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

        self._tools = build_tools(config)
        self._tool_map = {tool.__name__: tool for tool in self._tools}
        self._tool_descriptions = self._format_tool_descriptions(self._tools)
        self._max_react_steps = 6

        react_instructions = (
            "You MUST use an explicit ReAct loop. Available tools:\n"
            f"{self._tool_descriptions}\n\n"
            "At each step, output a ReActStep with:\n"
            "- action: a tool name from the list, OR 'final'\n"
            "- action_input: tool input as plain text or JSON string for args\n"
            "- final: only when action == 'final'\n\n"
            "Do not reveal chain-of-thought. Do not include tool deliberations. "
            "Only provide the final reply in the 'final' field when done. "
            "The final reply must NOT include mentions or usernames."
        )

        combined_instructions = f"{config.system_prompt}\n\n{react_instructions}"

        self._react_agent: Agent[None, ReActStep] = Agent(
            model,
            output_type=ReActStep,
            instructions=combined_instructions,
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

        # Build the current prompt
        prompt_parts = []
        if descriptions:
            prompt_parts.append(f"Image descriptions: {', '.join(descriptions)}")
        if note.user.location:
            prompt_parts.append(f"User location: {note.user.location}")
        prompt_parts.append(f"{note.user.username}: {note.text}")

        user_prompt = "\n".join(prompt_parts)

        # Build context text
        context_lines: list[str] = []
        if context:
            for c in reversed(context):
                if c.text:
                    context_lines.append(f"{c.user.username}: {c.text}")

        base_prompt_parts: list[str] = []
        if context_lines:
            base_prompt_parts.append("Conversation so far:\n" + "\n".join(context_lines))
        base_prompt_parts.append("Current message:\n" + user_prompt)
        base_prompt = "\n\n".join(base_prompt_parts)

        result = await self._run_react(base_prompt)
        logfire.info(f"Reply: {result}")
        return result

    async def describe_images(self, files: list[MiFile]) -> Optional[list[str]]:
        """Describe images using vision models."""
        with logfire.span("describe images", file_count=len(files)):
            image_urls: list[str] = []
            for f in files:
                logfire.info(
                    f"Looking at file: {f.id} ({f.type}): {f.thumbnailUrl}"
                )
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

    def _format_tool_descriptions(
        self, tools: list[Callable[..., object]]
    ) -> str:
        lines = []
        for tool in tools:
            name = tool.__name__
            doc = inspect.getdoc(tool) or ""
            if doc:
                doc = " ".join(doc.split())
            lines.append(f"- {name}: {doc}".strip())
        return "\n".join(lines)

    async def _run_react(self, base_prompt: str) -> ReplyOutput:
        steps: list[dict[str, str]] = []
        tool_names = ", ".join(self._tool_map.keys())

        for step_index in range(1, self._max_react_steps + 1):
            react_prompt_parts = [base_prompt]
            if steps:
                react_prompt_parts.append("\nPrevious actions and observations:")
                for idx, step in enumerate(steps, start=1):
                    react_prompt_parts.append(
                        "\n".join(
                            [
                                f"Step {idx}:",
                                f"Action: {step['action']}",
                                f"Input: {step['action_input']}",
                                f"Observation: {step['observation']}",
                            ]
                        )
                    )

            react_prompt_parts.append(
                "\nDecide your next action. If you have enough information, return action='final'."
            )

            react_prompt = "\n\n".join(react_prompt_parts)
            result = await self._react_agent.run(react_prompt)
            output = result.output

            action = (output.action or "").strip()
            action_input = (output.action_input or "").strip()

            if action == "final":
                reply = (output.final or "").strip()
                return ReplyOutput(reply=reply, mentions=None)

            if action not in self._tool_map:
                observation = (
                    f"Unknown tool '{action}'. Available tools: {tool_names}."
                )
            else:
                observation = await asyncio.to_thread(
                    self._call_tool_sync,
                    self._tool_map[action],
                    action_input,
                )

            steps.append(
                {
                    "action": action or "(none)",
                    "action_input": action_input or "(none)",
                    "observation": str(observation),
                }
            )

        return ReplyOutput(
            reply="Sorry, I couldn't complete that within the allowed steps.",
            mentions=None,
        )

    def _call_tool_sync(self, tool: Callable[..., object], action_input: str) -> object:
        if not action_input:
            return tool()

        try:
            parsed = json.loads(action_input)
        except json.JSONDecodeError:
            return tool(action_input)

        if isinstance(parsed, dict):
            return tool(**parsed)
        if isinstance(parsed, list):
            return tool(*parsed)
        return tool(str(parsed))
