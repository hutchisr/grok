import asyncio
from logging import INFO, DEBUG, basicConfig, getLogger
import signal

import click
import yaml

from .bot import Bot
from .models import Config
from .api import api_client

logger = getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")
def main(config):
    asyncio.run(main_async(config))

async def main_async(config):
    loop = asyncio.get_running_loop()

    with open(config, "r") as f:
        config = Config(**yaml.safe_load(f))

    if config.debug:
        basicConfig(level=DEBUG)
    else:
        basicConfig(level=INFO)


    # Configure global HTTP client with config settings
    api_client.configure(config)

    bot = Bot(
        config=config
    )

    def shutdown_handler():
        bot.shutdown()
        logger.info("Shut down bot")
        loop.create_task(api_client.close())
        logger.info("Closed API client")

    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    await bot.run()
