import asyncio
import threading
from logging import getLogger
from typing import Optional, cast, TYPE_CHECKING

import httpx
from .models import Config

logger = getLogger(__name__)


class ApiClient:
    def __init__(self):
        self.__async_client: Optional[httpx.AsyncClient] = None
        self.__config: Optional[Config] = None
        self.__lock = threading.Lock()

    @property
    def __client(self) -> httpx.AsyncClient:
        with self.__lock:
            if self.__async_client is None:
                if self.__config:
                    self.__async_client = httpx.AsyncClient(
                        transport=httpx.AsyncHTTPTransport(
                            retries=self.__config.max_retries
                        ),
                        headers={"Authorization": f"Bearer {self.__config.token}"},
                    )
                else:
                    logger.warning("API client accessed before configuration")
                    self.__async_client = httpx.AsyncClient()
        return self.__async_client

    def configure(self, config: Config) -> None:
        with self.__lock:
            self.__config = config
            old_client = (
                self.__async_client
                if self.__async_client and not self.__async_client.is_closed
                else None
            )
            if old_client:
                self.__async_client = None

        if old_client:
            try:
                asyncio.get_running_loop().create_task(old_client.aclose())
            except RuntimeError:
                asyncio.run(old_client.aclose())

    def get_client(self) -> httpx.AsyncClient:
        """Gets the underlying AsyncClient"""
        return self.__client

    async def close(self) -> None:
        """Close current client."""
        if self.__async_client and not self.__async_client.is_closed:
            await self.__async_client.aclose()

    def __getattr__(self, name):
        return getattr(self.__client, name)

    async def __aenter__(self):
        return await self.__client.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.__client.__aexit__(exc_type, exc_val, exc_tb)


# Create the instance
api_client = ApiClient()

if TYPE_CHECKING:
    class ApiAsyncClient(ApiClient, httpx.AsyncClient): ...
    api_client = cast(ApiAsyncClient, api_client)
