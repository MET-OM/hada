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
from pyresample.geometry import AreaDefinition
from pyresample.bilinear import XArrayBilinearResampler
from pyresample import bilinear, kd_tree

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

    @cache
    def __area_definition__(self):
        logger.debug(f'Setting area definition for {self.name}..')
        import pyresample.utils
        return pyresample.utils.load_cf_area(self.url)
        # return AreaDefinition('source_grid', 'Source grid', proj_id='source_grid',
        #                       projection=self.crs.to_proj4(), width=len(self.x), height=len(self.y),
        #                       area_extent=(self.xmin, self.ymin, self.xmax, self.ymax))

    @cache
    def __regridder__(self, target):
        logger.debug(f'Creating regridder for {self.name}..')
        source_area = self.__area_definition__()
        logger.debug(f'{source_area=}')
        resampler = XArrayBilinearResampler(source_area, target.area, 30e3)
        return resampler


    def regrid_pyresample(self, var, target, t0, t1):
        # re = self.__regridder__(target)
        var = var.sel(time=slice(t0, t1))
        logger.info(f'Regridding {var.name} on {self.name}..')

        var = var.transpose('Y', 'X', ...)

        print(var)

        # vo = bilinear.resample_bilinear(var, self.__area_definition__(),
        vo = kd_tree.resample_nearest(self.__area_definition__(), var.data, target.area, radius_of_influence=30e3)

        # vo = re.resample(var)
        print(vo)

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
