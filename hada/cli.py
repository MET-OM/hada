import logging
import coloredlogs
import click
from datetime import datetime, timedelta
import xarray as xr

from .sources import Sources
from .target import Target
from . import vector

logger = logging.getLogger(__name__)

@click.command()
@click.option('--log-level', default='info', help='Set log level')
@click.option('--sources', default='sources.toml', type=click.Path())
@click.option('--bbox-deg', default="5,10,60,65", help='Bounding box in degrees (xmin, xmax, ymin, ymax)')
@click.option('--bbox-m', help='Bounding box in meters on EPSG:3575 (xmin, xmax, ymin, ymax)')
@click.option('--nx', default=100, help='Resolution in x (longitude)')
@click.option('--ny', default=150, help='Resolution in y (latitude)')
@click.option('--from', 't0', type=click.DateTime(), help='UTC date-time start (default: -1day)')
@click.option('--to', 't1', type=click.DateTime(), help='UTC date-time end (default: now)')
@click.option('-d', '--dataset-filter', multiple=True, help='Only include datasets containing string')
@click.option('-v', '--variable-filter', multiple=True, help='Only include variables containing string')
@click.option('--output', type=click.Path(), help='Output file')
def hada(log_level, sources, bbox_deg, bbox_m, nx, ny, t0, t1, dataset_filter, variable_filter, output):
    coloredlogs.install(level=log_level)

    if t0 is None:
        t0 = datetime.utcnow() - timedelta(days=1)

    if t1 is None:
        t1 = datetime.utcnow()

    logger.info(f"hada: {t0} -> {t1}")

    # Load datasets
    sources = Sources.from_toml(sources, dataset_filter, variable_filter)
    logger.debug(f'sources: {sources}')

    # Compute target grid
    if bbox_deg is not None and bbox_m is not None:
        logger.error('Only one of bbox_deg and bbox_m can be specified at the same time.')
        return 1

    if bbox_m:
        bbox_m = list(map(lambda x: float(x.strip()), bbox_m.split(",")))
        assert len(bbox_m) == 4, "Bounding box should consit of 4 comma-separated floats"
        target = Target(bbox_m[0], bbox_m[1], bbox_m[2], bbox_m[3], nx, ny, output)
    else:
        bbox_d = list(map(lambda x: float(x.strip()), bbox_deg.split(",")))
        assert len(bbox_d) == 4, "Bounding box should consit of 4 comma-separated floats"
        target = Target.from_lonlat(bbox_d[0], bbox_d[1], bbox_d[2], bbox_d[3], nx, ny, output)

    ds = xr.Dataset()

    for var in sources.scalar_variables:
        logger.info(f'Searching for variable {var}')
        (d, v) = sources.find_dataset_for_var(var)

        if v is not None:
            logger.info(f'Extracting {var} from {d}')

            # Acquire variables on target grid
            vo = d.regrid(v, target, t0, t1)
            ds[var] = vo
        else:
            logger.error(f'No dataset found for variable {var}.')

    for var, vvar in sources.vector_magnitude_variables.items():
        assert len(vvar) == 2, "Vector variables can only contain two components"
        varx = vvar[0]
        vary = vvar[1]
        logger.info(f'Searching for variable {varx},{vary}')
        (d, vx, vy) = sources.find_dataset_for_var_pair(varx, vary)

        if vx is not None:
            logger.info(f'Extracting {varx} and {vary} from {d}')

            # Acquire variables on target grid
            vox = d.regrid(vx, target, t0, t1)
            voy = d.regrid(vy, target, t0, t1)

            vox.values = vector.magnitude(vox.values, voy.values)
            ds[var] = vox
        else:
            logger.error(f'No dataset found for variable {varx},{vary}.')

    logger.info('Merging variables into new dataset..')
    ds[target.proj_name] = target.proj_var
    logger.info('Re-gridded dataset done')
    print(ds)

    # Save to file
    if output is not None:
        logger.info(f'Saving dataset to file: {output}..')
        ds.to_netcdf(output, format='NETCDF4')

if __name__ == '__main__':
    hada()
