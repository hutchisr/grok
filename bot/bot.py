import json
from logging import getLogger
import re

import asyncio
from typing import Optional
import dspy
from pydantic import ValidationError
from websockets import ClientConnection, ConnectionClosed
from websockets.asyncio.client import connect
import httpx

from .ai import ChatAgent
from .models import Config, MiWebsocketMessage, Note
from .api import api_client

logger = getLogger(__name__)


class Bot:
    def __init__(
        self,
        config: Config,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.url = config.url
        self.ws_url = config.ws_url
        self.api_key = config.token
        self.username = config.bot_username
        self.user_id = config.bot_user_id
        self.loop = asyncio.get_event_loop() if loop is None else loop
        self.ws: Optional[ClientConnection] = None

        self._config = config
        self._agent = ChatAgent(config)
        self._shutdown_event = asyncio.Event()

    async def on_mention(self, note: Note):
        if note.user.id == self.user_id:
            logger.debug("Ignoring own mention")
            return
        if not note.text:
            logger.debug(f"Empty note? {note}")
            return
        logger.info(
            f"Received note: {note.id} `{note.user.username}: {note.text.replace("\n", "⏎")[:100]}`"
        )
        context: Optional[list[Note]] = []
        if note.replyId:
            reply_id = note.replyId
            for _ in range(self._config.max_context):
                try:
                    reply = await self.get_note(reply_id)
                except httpx.HTTPError:
                    logger.exception("Error fetching context")
                    break
                # Add to context if it has text OR files
                if reply.text or reply.files:
                    context.append(reply)
                # fetch next note in thread
                if reply.replyId:
                    reply_id = reply.replyId
                else:
                    break
        if note.renote and (note.renote.text or note.renote.files):
            context.append(note.renote)
        predict = await self._agent.reply(note=note, context=context)
        await self.send_note(predict, in_reply_to=note)

    async def send_note(
        self,
        prediction: dspy.Prediction,
        in_reply_to: Optional[Note] = None,
    ):
        # Filter out bot's own username from mentions
        mentions = (
            {
                f"{mention if mention.startswith("@") else "@" + mention}"
                for mention in prediction.mentions
                if not re.match(
                    rf"^@?{self._config.bot_username}(@{self._config.domain})?$",
                    mention.strip(),
                )
            }
            if prediction.mentions
            else set()
        )

        if in_reply_to and in_reply_to.user:
            username = f"@{in_reply_to.user.username}"
            if in_reply_to.user.host:
                username += f"@{in_reply_to.user.host}"
            mentions.add(username)

        payload = {
            "text": f"{' '.join(mentions)}\n{re.sub(r"^@[\w\-]+(:?@[\w\-\.]+)?\s+", "", prediction.reply)}",
            "visibility": "public",
        }
        if in_reply_to and in_reply_to.id:
            payload["replyId"] = in_reply_to.id

        response = await api_client.post(f"{self.url}api/notes/create", json=payload)
        response.raise_for_status()
        logger.info(f"Sent note: {response.json().get("createdNote").get("id")}")

    async def get_note(self, note_id: str) -> Note:
        response = await api_client.post(
            f"{self.url}api/notes/show",
            json={"noteId": note_id},
        )
        response.raise_for_status()
        note = response.json()
        note = Note(**note)
        logger.info(f"Fetched note: {note.id} with {len(note.files) if note.files else "no"} file(s)")
        return note

    async def run(self):
        logger.info("Connecting to WebSocket...")
        async for websocket in connect(f"{self.ws_url}/streaming?i={self.api_key}"):
            try:
                await websocket.send(
                    json.dumps(
                        {"type": "connect", "body": {"channel": "main", "id": "11111"}}
                    )
                )
                logger.info("WebSocket connected")

                shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                message_task = asyncio.create_task(self._handle_messages(websocket))

                done, pending = await asyncio.wait(
                    [shutdown_task, message_task], return_when=asyncio.FIRST_COMPLETED
                )

                if shutdown_task in done:
                    logger.info("Shutdown requested, closing connection")
                    await websocket.close()
                    return

            except ConnectionClosed:
                if self._shutdown_event.is_set():
                    return
                logger.warning("WebSocket connection closed, reconnecting...")
                continue

    async def _handle_messages(self, websocket: ClientConnection):
        async for message in websocket:
            try:
                msg = MiWebsocketMessage(**json.loads(message))
                logger.debug(f"{msg}")
                if msg.type == "channel" and msg.body and msg.body.type in {"mention"}:
                    if msg.body and msg.body.body:
                        task = asyncio.create_task(self.on_mention(msg.body.body))
                        task.add_done_callback(self._task_done_callback)
            except ValidationError as e:
                logger.debug(
                    f"Validation error: {e}. Message doesn't match expected format, ignoring."
                )
                pass
            except asyncio.CancelledError:
                logger.info("Message handler cancelled")
                raise
            except Exception:
                logger.exception("Error processing message")

    def _task_done_callback(self, task: asyncio.Task):
        """Handle completed tasks - log exceptions and discard."""
        if task.cancelled():
            return

        try:
            task.result()  # This will raise any exception that occurred
        except Exception:
            logger.exception("Task failed with exception")

    def shutdown(self):
        self._shutdown_event.set()
