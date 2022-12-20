import logging
import coloredlogs
import click
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd
import numpy as np

from .sources import Sources
from .target import Target
from . import vector
from . import derive

logger = logging.getLogger(__name__)

@click.command()
@click.option('--log-level', default='info', help='Set log level')
@click.option('--sources', default='sources.toml', type=click.Path())
@click.option('--bbox-deg', help='Bounding box in degrees (xmin, xmax, ymin, ymax)')
@click.option('--bbox-m', help='Bounding box in meters on EPSG:3575 (xmin, xmax, ymin, ymax)')
@click.option('--dx', help='Grid size in x direction')
@click.option('--dy', help='Grid size in y direction')
@click.option('--from', 't0', type=click.DateTime(), help='UTC date-time start (default: -1day)')
@click.option('--to', 't1', type=click.DateTime(), help='UTC date-time end (default: now)')
@click.option('--freq', default='1H', type=str, help='Time delta in time span.')
@click.option('-d', '--dataset-filter', multiple=True, help='Only include datasets containing string')
@click.option('-v', '--variable-filter', multiple=True, help='Only include variables containing string')
@click.option('--output', type=click.Path(), help='Output file')
def hada(log_level, sources, bbox_deg, bbox_m, dx, dy, t0, t1, freq, dataset_filter, variable_filter, output):
    coloredlogs.install(level=log_level)

    if t0 is None:
        t0 = datetime.utcnow() - timedelta(days=1)

    if t1 is None:
        t1 = datetime.utcnow()

    logger.info(f"hada: {t0} -> {t1}")

    time = pd.date_range(t0, t1, freq=freq)

    # Compute target grid
    if bbox_deg is not None and bbox_m is not None:
        logger.error('Only one of bbox_deg and bbox_m can be specified at the same time.')
        return 1

    if bbox_m:
        bbox_m = list(map(lambda x: float(x.strip()), bbox_m.split(",")))
        assert len(bbox_m) == 4, "Bounding box should consist of 4 comma-separated floats"

        if dx is None:
            dx = 800

        if dy is None:
            dy = 800

        assert bbox_m[1] > bbox_m[0], "xmax must be greater than xmin"
        assert bbox_m[3] > bbox_m[2], "ymax must be greater than ymin"

        nx = np.max((int((bbox_m[1] - bbox_m[0]) / dx), 1))
        ny = np.max((int((bbox_m[3] - bbox_m[2]) / dy), 1))

        logger.debug(f'{bbox_m}: {nx=}, {ny=}, {dx=}, {dy=}')

        target = Target(bbox_m[0], bbox_m[1], bbox_m[2], bbox_m[3], nx, ny, output)
    else:
        # default
        if bbox_deg is None:
            bbox_deg = "5,10,60,65"

        if dx is None:
            dx = 0.5

        if dy is None:
            dy = 0.5

        bbox_d = list(map(lambda x: float(x.strip()), bbox_deg.split(",")))
        assert len(bbox_d) == 4, "Bounding box should consist of 4 comma-separated floats"

        assert bbox_d[1] > bbox_d[0], "xmax must be greater than xmin"
        assert bbox_d[3] > bbox_d[2], "ymax must be greater than ymin"

        nx = np.max((int((bbox_d[1] - bbox_d[0]) / dx), 1))
        ny = np.max((int((bbox_d[3] - bbox_d[2]) / dy), 1))

        target = Target.from_lonlat(bbox_d[0], bbox_d[1], bbox_d[2], bbox_d[3], nx, ny, output)

    # Load datasets
    sources = Sources.from_toml(sources, dataset_filter, variable_filter)
    logger.debug(f'sources: {sources}')

    ds = xr.Dataset()

    for var in sources.scalar_variables:
        logger.info(f'Searching for variable {var}')

        vo = sources.regrid(var, target, time)
        if vo is not None:  # None if outside domain
            ds[var] = vo
        else:
            logger.error(f'No dataset found for variable {var}.')

    for var, vvar in sources.vector_magnitude_variables.items():
        assert len(vvar) == 2, "Vector variables can only contain two components"
        varx = vvar[0]
        vary = vvar[1]
        logger.info(f'Searching for variable {varx},{vary}')

        # Acquire variables on target grid
        vox = sources.regrid(varx, target, time)
        voy = sources.regrid(vary, target, time)

        if vox is not None and voy is not None:  # None if outside domain.
            vox.values = vector.magnitude(vox.values, voy.values)
            ds[var] = vox
        else:
            logger.error(f'No dataset found for variable {varx},{vary}.')

    for var, vvar in sources.derived_variables.items():
        logger.info(f'Searching for variables {vvar}')

        vos = []

        for vn in vvar:
            vo = sources.regrid(vn, target, time)
            if vo is not None:
                vos.append(vo)
                continue

            logger.error(f'Could not find dataset for {vn}.')
            break

        if len(vos) != len(vvar):
            logger.error(f'Could not find all necessary variables for {var}, skipping.')
            continue
        else:
            vo = derive.derive(var, *vos)
            if vo is not None:
                ds[var] = vo

    ds[target.proj_name] = target.proj_var

    logger.info('Re-gridded dataset done')
    print(ds)

    # Save to file
    if output is not None:
        logger.info(f'Saving dataset to file: {output}..')
        ds.to_netcdf(output, format='NETCDF4')

if __name__ == '__main__':
    hada()
