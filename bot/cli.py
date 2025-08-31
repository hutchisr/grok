import asyncio
from logging import INFO, basicConfig
import signal

import click
import dspy
import yaml

from .bot import Bot
from .models import Config

basicConfig(level=INFO)


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")
def main(config):
    asyncio.run(main_async(config))

async def main_async(config):
    loop = asyncio.get_running_loop()

    with open(config, "r") as f:
        config = Config(**yaml.safe_load(f))

    dspy.settings.configure(track_usage=True)

    bot = Bot(
        config=config
    )

    loop.add_signal_handler(signal.SIGTERM, bot.shutdown)
    loop.add_signal_handler(signal.SIGINT, bot.shutdown)


    await bot.run()
