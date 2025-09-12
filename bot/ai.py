import asyncio
from logging import getLogger
from pathlib import Path
from typing import Optional

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


class ChatAgent(dspy.Module):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        search_tool = configure_web_search(config)
        tools = [search_tool]

        self.generate_reply = dspy.ReAct(
            Reply.with_instructions(config.system_prompt), tools=tools
        )

        if config.model_file and Path(config.model_file).is_file():
            self.load(config.model_file)
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
                        temperature=1,
                        track_usage=True,

                    ),
                ):
                    try:
                        output = await self.acall(
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

    async def aforward(self, message, context=None, location=None):
        return await self.generate_reply.acall(
            message=message,
            context=context,
            location=location,
        )
