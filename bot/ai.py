import asyncio
from logging import getLogger
from pathlib import Path
from typing import Optional

import dspy

from .models import Config, Note, MiFile
from .tools import configure_web_search


logger = getLogger(__name__)


class Reply(dspy.Signature):
    message: str = dspy.InputField()
    descriptions: Optional[list[str]] = dspy.InputField(desc="descriptions of images included in message")
    context: Optional[list[str]] = dspy.InputField(desc="context referenced by message")
    location: Optional[str] = dspy.InputField(desc="location of the user, if known")
    # history: Optional[dspy.History] = dspy.InputField()
    reply: str = dspy.OutputField(
        desc="Reply to the message. Must NOT include mentions or usernames. Must not be None."
    )
    mentions: Optional[list[str]] = dspy.OutputField(
        desc="list of users mentioned in the message"
    )


class ImagesSig(dspy.Signature):
    images: list[dspy.Image] = dspy.InputField()
    descriptions: list[str] = dspy.OutputField()


class ChatAgent(dspy.Module):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        search_tool = configure_web_search(config)
        tools = [search_tool]

        self.generate_reply = dspy.ReAct(
            Reply.with_instructions(config.system_prompt), tools=tools
        )
        self.image_describer = dspy.Predict(ImagesSig.with_instructions("Describe the images provided"))


        if config.model_file and Path(config.model_file).is_file():
            self.load(config.model_file)
            logger.info(f"Loaded model {config.model_file}")

    async def reply(self, note: Note, context: Optional[list[Note]] = None) -> dspy.Prediction:
        if not note.text:
            raise ValueError("Note text is empty")
        files: list[MiFile] = []
        if context:
            for n in context:
                if n.files:
                    files.extend(n.files)
        if note.files:
            files.extend(note.files)
        logger.info(f"Checking {len(files)} file(s)")
        if files:
            descriptions = await self.describe_images(files)
        else:
            descriptions = None
        for ep in self._config.llm_endpoints:
            while True:
                with dspy.context(
                    lm=dspy.LM(
                        model=f"{ep.provider if ep.provider else "openai"}/{ep.model}",
                        api_key=ep.key,
                        api_base=str(ep.url),
                        max_tokens=self._config.max_tokens,
                        temperature=1,
                    ),
                ):
                    try:
                        output = await self.acall(
                            message=note.text,
                            descriptions=descriptions,
                            location=note.user.location,
                            context=[f"{c.user.name}: {c.text}" for c in context if c.text] if context else None,
                        )

                        logger.info(f"Reply: {output}")

                        return output
                    except TypeError as err:
                        logger.info(f"dspy probably ate shit ({err}), retrying...")
                        await asyncio.sleep(0)
                        continue
                    except Exception:
                        logger.exception(
                            "Error occurred while processing message. Will try next LLM"
                        )
                        await asyncio.sleep(1)
                        break

        raise RuntimeError("All LLM endpoints failed")

    async def describe_images(self, files: list[MiFile]) -> Optional[list[str]]:
        images: list[dspy.Image] = []
        for f in files:
            logger.info(f"Looking at file: {f.id} ({f.type}): {f.thumbnailUrl}")
            if f.thumbnailUrl:
                try:
                    images.append(await asyncio.to_thread(dspy.Image.from_url, f.thumbnailUrl))
                except Exception:
                    logger.exception("Failed to load image")
        for ep in self._config.vision_endpoints:
            while True:
                with dspy.context(
                    lm=dspy.LM(
                        model=f"{ep.provider if ep.provider else "openai"}/{ep.model}",
                        api_key=ep.key,
                        api_base=str(ep.url),
                        temperature=1,
                    ),
                ):
                    try:
                        pred = await self.image_describer.acall(images=images)
                        logger.info(f"Image descriptions: {pred}")
                        return pred.descriptions
                    except TypeError as err:
                        logger.info(f"dspy probably ate shit ({err}), retrying...")
                        await asyncio.sleep(0)
                        continue
                    except Exception:
                        logger.exception(
                            "Error occurred while processing message. Will try next LLM"
                        )
                        await asyncio.sleep(1)
                        break

    async def aforward(self, *args, **kwargs):
        return await self.generate_reply.acall(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.generate_reply(*args, **kwargs)
