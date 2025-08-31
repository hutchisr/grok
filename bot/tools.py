from typing import Optional
from colorlog import getLogger
import httpx

from .models import Config

logger = getLogger(__name__)


def configure_web_search(config: Config):
    def search_web(query: str) -> Optional[str]:
        auth: Optional[httpx.BasicAuth] = None
        if config.searxng_user and config.searxng_password:
            auth = httpx.BasicAuth(config.searxng_user, config.searxng_password)
        transport = httpx.HTTPTransport(retries=config.max_retries)
        with httpx.Client(auth=auth, transport=transport) as client:
            try:
                response = client.post(
                    f"{config.searxng_url}search", params={"q": query, "format": "json"}
                )
                response.raise_for_status()
                data = response.json()
                return "\n---\n".join([result.get("content") for result in data.get("results", [])[:5]])
            except httpx.HTTPError:
                logger.exception("HTTP Error")
                return None

    return search_web
