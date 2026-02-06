import asyncio
import signal
from typing import Optional

import click
import logfire
import yaml
from redis.asyncio import Redis

from .bot import Bot
from .models import Config
from .api import api_client


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")
def main(config):
    asyncio.run(main_async(config))

async def main_async(config):
    loop = asyncio.get_running_loop()

    with open(config, "r") as f:
        config = Config(**yaml.safe_load(f))

    debug_enabled = bool(config.debug)
    min_level = "debug" if debug_enabled else "info"
    logfire.configure(
        min_level=min_level,
        console=logfire.ConsoleOptions(min_log_level=min_level, verbose=debug_enabled),
    )
    logfire.instrument_pydantic_ai()


    # Configure global HTTP client with config settings
    api_client.configure(config)

    # Initialize Redis client if configured
    redis_client: Optional[Redis] = None
    if config.redis_url:
        redis_client = Redis.from_url(
            config.redis_url,
            password=config.redis_password,
            db=config.redis_db or 0,
            decode_responses=True,  # Get strings instead of bytes
        )
        logfire.info("Redis client initialized")

    bot = Bot(
        config=config,
        redis_client=redis_client,
    )

    def shutdown_handler():
        bot.shutdown()
        logfire.info("Shut down bot")
        loop.create_task(api_client.close())
        logfire.info("Closed API client")
        if redis_client:
            loop.create_task(redis_client.aclose())
            logfire.info("Closed Redis client")

    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    await bot.run()
