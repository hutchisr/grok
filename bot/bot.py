import json
import re

import asyncio
from typing import Optional
from pydantic import ValidationError
from websockets import ClientConnection, ConnectionClosed
from websockets.asyncio.client import connect
import httpx
import logfire

from .ai import ChatAgent, ReplyOutput
from .models import Config, MiWebsocketMessage, Note, User
from .api import api_client


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
        with logfire.span(
            "handle mention",
            note_id=note.id,
            user_id=note.user.id,
            username=note.user.username,
            has_reply=bool(note.replyId),
            has_renote=bool(note.renote),
        ):
            if note.user.id == self.user_id:
                logfire.debug("Ignoring own mention")
                return
            if not note.text:
                logfire.debug(f"Empty note? {note}")
                return
            logfire.info(
                f"Received note: {note.id} `{note.user.username}: {note.text.replace('\n', '⏎')[:100]}`"
            )
            context: Optional[list[Note]] = []
            if note.replyId:
                reply_id = note.replyId
                for _ in range(self._config.max_context):
                    try:
                        reply = await self.get_note(reply_id)
                    except httpx.HTTPError:
                        logfire.exception("Error fetching context")
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
            result = await self._agent.run(note=note, context=context)
            await self.send_note(result, in_reply_to=note)

    async def send_note(
        self,
        output: ReplyOutput,
        in_reply_to: Optional[Note] = None,
    ):
        mentions = await self._build_mentions_from_note(in_reply_to)

        payload = {
            "text": f"{' '.join(mentions)}\n{self._strip_leading_mentions(output.reply or "")}",
            "visibility": "public",
        }
        if in_reply_to and in_reply_to.id:
            payload["replyId"] = in_reply_to.id

        response = await api_client.post(f"{self.url}api/notes/create", json=payload)
        response.raise_for_status()
        logfire.info(f"Sent note: {response.json().get('createdNote').get('id')}")

    async def _build_mentions_from_note(self, note: Optional[Note]) -> list[str]:
        if not note or not note.user:
            return []

        mentions: list[str] = []
        if note.mentions:
            for mention in note.mentions:
                normalized = await self._normalize_note_mention(mention)
                if not normalized:
                    continue
                if re.match(
                    rf"^@?{self._config.bot_username}(@{self._config.domain})?$",
                    normalized.strip(),
                    re.IGNORECASE,
                ):
                    continue
                mentions.append(normalized)

        mentions.append(self._format_handle(note.user))
        return self._unique_ordered(mentions)

    async def _normalize_note_mention(self, mention: str) -> Optional[str]:
        raw = mention.strip()
        if not raw:
            return None

        raw = raw.lstrip("@")
        if not raw:
            return None

        if "@" in raw:
            username, host = raw.split("@", 1)
            if not username:
                return None
            return f"@{username}@{host}" if host else f"@{username}"

        resolved = await self._resolve_user_handle(raw)
        return resolved or f"@{raw}"

    async def _resolve_user_handle(self, user_id: str) -> Optional[str]:
        try:
            response = await api_client.post(
                f"{self.url}api/users/show",
                json={"userId": user_id},
            )
            response.raise_for_status()
            data = response.json()
            username = data.get("username")
            if not username:
                return None
            host = data.get("host")
            return f"@{username}@{host}" if host else f"@{username}"
        except httpx.HTTPError:
            return None

    async def get_note(self, note_id: str) -> Note:
        response = await api_client.post(
            f"{self.url}api/notes/show",
            json={"noteId": note_id},
        )
        response.raise_for_status()
        note = response.json()
        note = Note(**note)
        text_preview = note.text.replace("\n", "⏎")[:100] if note.text else ""
        logfire.info(
            "Fetched note",
            note_id=note.id,
            username=note.user.username if note.user else "unknown",
            file_count=len(note.files) if note.files else 0,
            text_preview=text_preview,
        )
        return note

    async def run(self):
        logfire.info("Connecting to WebSocket...")
        async for websocket in connect(f"{self.ws_url}/streaming?i={self.api_key}"):
            try:
                with logfire.span("connect websocket"):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "connect",
                                "body": {"channel": "main", "id": "11111"},
                            }
                        )
                    )
                logfire.info("WebSocket connected")

                shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                message_task = asyncio.create_task(self._handle_messages(websocket))

                done, pending = await asyncio.wait(
                    [shutdown_task, message_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if shutdown_task in done:
                    logfire.info("Shutdown requested, closing connection")
                    await websocket.close()
                    return

            except ConnectionClosed:
                if self._shutdown_event.is_set():
                    return
                logfire.warning("WebSocket connection closed, reconnecting...")
                continue

    async def _handle_messages(self, websocket: ClientConnection):
        async for message in websocket:
            try:
                msg = MiWebsocketMessage(**json.loads(message))
                logfire.debug(f"{msg}")
                if msg.type == "channel" and msg.body and msg.body.type in {"mention"}:
                    if msg.body and msg.body.body:
                        task = asyncio.create_task(self.on_mention(msg.body.body))
                        task.add_done_callback(self._task_done_callback)
            except ValidationError as e:
                logfire.debug(
                    f"Validation error: {e}. Message doesn't match expected format, ignoring."
                )
                pass
            except asyncio.CancelledError:
                logfire.info("Message handler cancelled")
                raise
            except Exception:
                logfire.exception("Error processing message")

    def _task_done_callback(self, task: asyncio.Task):
        """Handle completed tasks - log exceptions and discard."""
        if task.cancelled():
            return

        try:
            task.result()  # This will raise any exception that occurred
        except Exception:
            logfire.exception("Task failed with exception")

    def shutdown(self):
        self._shutdown_event.set()

    def _format_handle(self, user: User) -> str:
        handle = f"@{user.username}"
        if user.host:
            handle += f"@{user.host}"
        return handle


    def _unique_ordered(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _strip_leading_mentions(self, text: str) -> str:
        return re.sub(r"^(?:@[\w\-]+(?:@[\w\-\.]+)?\s+)+", "", text)
