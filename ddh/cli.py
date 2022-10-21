import logging
import coloredlogs
import click

from .sources import Sources

logger = logging.getLogger(__name__)

@click.command()
@click.option('--log-level', default='info', help='Set log level')
@click.option('--sources', default='sources.toml', type=click.Path())
def ddh(log_level, sources):
    coloredlogs.install(level=log_level)
    logger.info("ddh")

    sources = Sources.from_toml(sources)
    logger.debug(f'sources: {sources}')

if __name__ == '__main__':
    ddh()
