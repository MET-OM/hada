import logging
import coloredlogs
import click

from .sources import Sources
from .target import Target

logger = logging.getLogger(__name__)

@click.command()
@click.option('--log-level', default='info', help='Set log level')
@click.option('--sources', default='sources.toml', type=click.Path())
@click.option('--bbox', default="5,10,60,65", help='Bounding box in degrees (xmin, xmax, ymin, ymax)')
@click.option('--nx', default=100, help='Resolution in x (longitude)')
@click.option('--ny', default=100, help='Resolution in y (latitude)')
@click.option('--output', type=click.Path(), help='Output file')
def ddh(log_level, sources, bbox, nx, ny, output):
    coloredlogs.install(level=log_level)
    logger.info("ddh")
    bbox = map(lambda x: float(x.strip()), bbox.split(","))

    # Load datasets
    sources = Sources.from_toml(sources)
    logger.debug(f'sources: {sources}')

    # Figure out which variables are available in each dataset
    # Compute which to acquire from each

    # Compute target grid
    target = Target(*bbox, nx, ny, output)

    # Acquire variables around grid
    # Interpolate to target grid
    # Rotate vectors if necessary
    # Store variables

if __name__ == '__main__':
    ddh()
