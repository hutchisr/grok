import asyncio
from logging import getLogger
from pathlib import Path
from typing import Callable, List, Optional
from datetime import datetime, timezone
import json

import dspy

from .models import Config, Note
from .tools import configure_web_search


logger = getLogger(__name__)


class Reply(dspy.Signature):
    message: str = dspy.InputField()
    context: Optional[str] = dspy.InputField(desc="context referenced by message")
    location: Optional[str] = dspy.InputField(desc="location of the user, if known")
    # history: Optional[dspy.History] = dspy.InputField()
    reply: str = dspy.OutputField(
        desc="Reply to the message. Must NOT include mentions or usernames. Must not be None."
    )
    mentions: Optional[list[str]] = dspy.OutputField(
        desc="list of users mentioned in the message"
    )


class ChatBotModule(dspy.Module):
    def __init__(self, system_prompt: str, tools: Optional[list[Callable]] = None):
        super().__init__()
        self.system_prompt = system_prompt
        if tools is None:
            tools = []

        self.generate_reply = dspy.ReAct(
            Reply.with_instructions(system_prompt), tools=tools
        )

    async def aforward(self, message, context=None, location=None):
        return await self.generate_reply.acall(
            message=message,
            context=context,
            location=location,
        )

    def forward(self, message, context=None, location=None):
        return self.generate_reply(
            message=message,
            context=context,
            location=location,
        )


class ChatAgent:
    def __init__(self, config: Config):
        self._config = config

        search_tool = configure_web_search(config)
        self._agent = ChatBotModule(config.system_prompt, [search_tool])
        if config.model_file and Path(config.model_file).is_file():
            self._agent.load(config.model_file)
            logger.info(f"Loaded model {config.model_file}")

    async def reply(self, note: Note, context: Optional[str] = None) -> dspy.Prediction:
        if not note.text:
            raise ValueError("Note text is empty")
        for ep in self._config.llm_endpoints:
            while True:
                with dspy.context(
                    lm=dspy.LM(
                        model=f"{ep.provider if ep.provider else "openai"}/{ep.model}",
                        api_key=ep.key,
                        api_base=str(ep.url),
                        max_tokens=self._config.max_tokens,
                        temperature=0.7,
                        track_usage=False,
                    ),
                ):
                    try:
                        output = await self._agent.acall(
                            message=note.text,
                            location=note.user.location,
                            context=context,
                        )

                        logger.info(f"Reply: {output}")

                        return output
                    except TypeError as err:
                        logger.info(f"dspy probably ate shit ({err}), retrying...")
                        continue
                    except Exception:
                        logger.exception("Error occurred while processing message. Will try next LLM")
                        await asyncio.sleep(1)
                        break

        raise RuntimeError("All LLM endpoints failed")
