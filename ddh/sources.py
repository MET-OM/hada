import logging
from dataclasses import dataclass
from typing import List, Any
import toml
import numpy as np
import xarray as xr
import rioxarray as _
import cf_xarray as _
import pyproj
import xesmf as xe
from functools import cache
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Dataset:
    name: str
    url: str
    variables: List[str]
    ds: xr.Dataset

    x: np.ndarray
    y: np.ndarray
    kdtree: Any

    target_x: np.ndarray
    target_y: np.ndarray
    target_hash: Any = None

    def __init__(self, name, url, variables):
        self.name = name
        self.url = url
        self.variables = variables if variables else []

        logger.info(
            f'{self.name}: opening: {self.url} for variables: {self.variables}'
        )
        self.ds = xr.decode_cf(xr.open_dataset(url, decode_coords='all'))

        # TODO: likely to be specific to dataset
        self.x = self.ds['X'].values
        self.y = self.ds['Y'].values

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.xmin, self.xmax = self.x.min(), self.x.max()
        self.ymin, self.ymax = self.y.min(), self.y.max()

        assert all(np.diff(self.x) - self.dx == 0)
        assert all(np.diff(self.y) - self.dy == 0)

        logger.debug(
            f'x: {self.x.shape} / {self.dx}, y: {self.y.shape} / {self.dy}')
        logger.debug(
            f'x: {self.x.min()} -> {self.x.max()}, y: {self.y.min()} -> {self.y.max()}'
        )

        self.crs = pyproj.Proj(self.ds.rio.crs.to_proj4())
        # self.crs = pyproj.Proj(
        #     '+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
        # )
        logger.debug(f'CRS: {self.crs}')

    def __repr__(self):
        return f'<Dataset ({self.name} / {self.url})>'

    @cache
    def __make_regridder__(self, target):
        logger.debug(f'Making regridder for {self.name}..')
        return xe.Regridder(self.ds, target.ds, 'bilinear')

    def regrid_xesmf(self, var, target, t0, t1):
        re = self.__make_regridder__(target)
        var = var.sel(time=slice(t0, t1))
        logger.info(f'Regridding {var.name} on {self.name}..')
        vo = re(var)

        vo.attrs = var.attrs
        vo.name = var.name

        vo.lat.attrs['units'] = 'degrees_north'
        vo.lat.attrs['standard_name'] = 'latitude'
        vo.lat.attrs['long_name'] = 'latitude'

        vo.lon.attrs['units'] = 'degrees_east'
        vo.lon.attrs['standard_name'] = 'longitude'
        vo.lon.attrs['long_name'] = 'longitude'

        vo.attrs['grid_mapping'] = target.proj_name
        vo.attrs['source'] = self.url

        # plt.figure()
        # vo.isel(time=0).plot()
        # plt.show(block=True)

        return vo

@dataclass
class Sources:
    variables: List[str]
    datasets: List[Dataset]

    def find_dataset_for_var(self, var):
        """
        Find first dataset with variable.
        """
        for d in self.datasets:
            logger.debug(f'Looking for {var} in {d}')
            if var in d.variables:
                if d.ds.cf[var] is not None:
                    return (d, d.ds.cf[var])

        return (None, None)

    @staticmethod
    def from_toml(file):
        logger.info(f'Loading sources from {file}')
        d = toml.load(open(file))

        global_variables = d['variables']

        datasets = [
            Dataset(name=name,
                    url=d['url'],
                    variables=d.get('variables', global_variables))
            for name, d in d['datasets'].items()
        ]
        return Sources(d['variables'], datasets)
