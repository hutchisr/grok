import asyncio
from logging import getLogger
from typing import Optional
from datetime import datetime, timezone

import dspy

from .models import Config, Note
from .tools import configure_web_search


logging = getLogger(__name__)


class Reply(dspy.Signature):
    message: str = dspy.InputField()
    context: Optional[str] = dspy.InputField(desc="context referenced by message")
    location: Optional[str] = dspy.InputField(desc="location of the user, if known")
    # history: Optional[dspy.History] = dspy.InputField()
    reply: str = dspy.OutputField(desc="the reply to the message, minus any mentions")
    mentions: Optional[list[str]] = dspy.OutputField(
        desc="list of users mentioned in the message"
    )


class ChatAgent:
    def __init__(self, config: Config):
        self._config = config

        search_tool = configure_web_search(config)
        self._agent = dspy.ReAct(
            Reply.with_instructions(
                config.system_prompt
                + f"\n\nThe current time is {datetime.now(timezone.utc).isoformat()}"
            ),
            [search_tool],
        )

    async def reply(self, note: Note, context: Optional[str] = None) -> dspy.Prediction:
        if not note.text:
            raise ValueError("Note text is empty")
        for ep in self._config.llm_endpoints:
            with dspy.context(
                lm=dspy.LM(
                    model=f"{ep.provider if ep.provider else "openai"}/{ep.model}",
                    api_key=ep.key,
                    api_base=str(ep.url),
                    max_tokens=self._config.max_tokens,
                ),
                track_usage=True,
            ):
                try:
                    output = await self._agent.acall(
                        message=note.text, location=note.user.location, context=context
                    )

                    logging.info(f"Reply: {output}")
                    logging.info(f"Usage: {output.get_lm_usage()}")

                    return output
                except Exception:
                    logging.exception("Error occurred while processing message")
                    await asyncio.sleep(1)

        raise RuntimeError("All LLM endpoints failed")
