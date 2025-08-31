import asyncio
from logging import getLogger
from typing import Optional

import dspy

from .models import Config, Note
from .tools import configure_web_search


logging = getLogger(__name__)



class ChatAgent:
    def __init__(self, config: Config):
        self._config = config

        class Reply(dspy.Signature):
            message: str = dspy.InputField()
            context: Optional[str] = dspy.InputField(desc="context referenced by message")
            # history: Optional[dspy.History] = dspy.InputField()
            reply: str = dspy.OutputField(desc="the reply to the message, minus any mentions")
            mentions: Optional[list[str]] = dspy.OutputField(
                desc="list of users mentioned in the message"
            )
        Reply.__doc__ = f"{config.system_prompt}"

        search_tool = configure_web_search(config)
        self._agent = dspy.ReAct(Reply, [search_tool])

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
                    output = await self._agent.acall(message=note.text, context=context)

                    logging.info(f"Reply: {output}")
                    logging.info(f"Usage: {output.get_lm_usage()}")

                    return output
                except Exception as e:
                    logging.error(f"Error occurred while processing message: {e}")
                    await asyncio.sleep(1)

        raise RuntimeError("All LLM endpoints failed")
