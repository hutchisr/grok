import asyncio
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Optional

import dspy

from .models import Config, Note, MiFile
from .tools import configure_web_search, current_datetime


logger = getLogger(__name__)


class Reply(dspy.Signature):
    message: str = dspy.InputField()
    user: str = dspy.InputField(desc="Author of the message")
    descriptions: Optional[list[str]] = dspy.InputField(
        desc="descriptions of images included in message"
    )
    history: Optional[dspy.History] = dspy.InputField(desc="Conversation history")
    location: Optional[str] = dspy.InputField(desc="location of the user, if known")
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
        tools = [search_tool, current_datetime]

        self.generate_reply = dspy.ReAct(
            Reply.with_instructions(config.system_prompt), tools=tools
        )
        self.image_describer = dspy.Predict(
            ImagesSig.with_instructions("Describe the images provided")
        )

        if config.model_file and Path(config.model_file).is_file():
            self.load(config.model_file)
            logger.info(f"Loaded model {config.model_file}")

    async def aforward(
        self, note: Note, context: Optional[list[Note]] = None
    ) -> dspy.Prediction:
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
                        output = await self.generate_reply.acall(
                            message=note.text,
                            user=note.user.name,
                            descriptions=descriptions,
                            location=note.user.location,
                            history=dspy.History(
                                messages=(
                                    [
                                        {"message": c.text, "user": c.user.name}
                                        for c in context
                                    ]
                                    if context
                                    else []
                                )
                            ),
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
                    images.append(
                        await asyncio.to_thread(dspy.Image.from_url, f.thumbnailUrl)
                    )
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

    def forward(self, *args, **kwargs):
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.aforward(*args, **kwargs))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result()
