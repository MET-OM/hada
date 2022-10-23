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
    bbox = list(map(lambda x: float(x.strip()), bbox.split(",")))
    assert len(bbox) == 4, "Bounding box should consit of 4 comma-separated floats"

    # Load datasets
    sources = Sources.from_toml(sources)
    logger.debug(f'sources: {sources}')

    # Figure out which variables are available in each dataset
    # Compute which to acquire from each

    # Compute target grid
    target = Target(bbox[0], bbox[1], bbox[2], bbox[3], nx, ny, output)

    for var in sources.variables:
        logger.info(f'Searching for variable {var}')
        (d, v) = sources.find_dataset_for_var(var)

        if v is not None:
            logger.info(f'Extracting {var} from {d}')
            # Calculate target grid on source grid
            d.calculate_grid(target)

            # Acquire variables around grid

            # Interpolate to target grid

            # Rotate vectors if necessary

            # Store variables
        else:
            logger.error(f'No dataset found for variable {var}.')

    # Flush file

if __name__ == '__main__':
    ddh()
