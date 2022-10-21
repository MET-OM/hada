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

    # Load datasets
    # Figure out which variables are available in each dataset
    # Compute which to acquire from each
    # Compute target grid
    # Acquire variables around grid
    # Interpolate to target grid
    # Rotate vectors if necessary
    # Store variables

if __name__ == '__main__':
    ddh()
