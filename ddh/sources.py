import logging
from dataclasses import dataclass
from typing import List, Any
import toml
import numpy as np
import xarray as xr
import rioxarray as _
import cf_xarray as _
import pyproj
from pykdtree.kdtree import KDTree
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

        # logger.debug(f'Setting up KDTree for coordinates')
        # self.kdtree = KDTree(np.vstack((self.x, self.y)).T)

        # self.crs = pyproj.Proj(self.ds.rio.crs.to_proj4())
        self.crs = pyproj.Proj(
            '+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
        )
        logger.debug(f'CRS: {self.crs}')

    def __repr__(self):
        return f'<Dataset ({self.name} / {self.url})>'

    def calculate_grid(self, target):
        target_hash = hash(target)
        if self.target_hash != target_hash:
            logger.debug(f'Calculating grid for target..')

            # Calculating the location of the target grid cells
            # in this datasets coordinate system.
            # tf = pyproj.Transformer.from_proj(target.crs, self.crs)

            # self.target_x, self.target_y = tf.transform(
            #     target.xx.ravel(), target.yy.ravel())

            self.target_x, self.target_y = self.crs(target.xx.ravel(),
                                                    target.yy.ravel(),
                                                    inverse=False)
            self.target_x.shape = target.xx.shape
            self.target_y.shape = target.yy.shape

            # Target coordinates within source domain
            self.inbounds = (self.target_x>=self.xmin) & (self.target_x<self.xmax) & (self.target_y>=self.ymin) & (self.target_y<self.ymax)

            if not any(self.inbounds.ravel()):
                logger.warning('Target is outside the domain of this reader')

            self.target_hash = target_hash

        return self.target_x, self.target_y

    def regrid(self, var, target, t0, t1):
        """
        Return values for the target grid.
        """
        logger.info(f'Regridding {var} between {t0} and {t1}')

        var = var.sel(time=slice(t0, t1))

        # Extract block
        x0 = np.min(self.target_x[self.inbounds]) - self.dx
        x1 = np.max(self.target_x[self.inbounds]) + self.dx
        y0 = np.min(self.target_y[self.inbounds]) - self.dy
        y1 = np.max(self.target_y[self.inbounds]) + self.dy

        logger.debug(f'Load block between x: {x0}..{x1}, y: {y0}..{y1}')
        block = var.sel(X=slice(x0, x1), Y=slice(y0, y1)).load()
        block.isel(time=1).plot()

        logger.debug(f'Extracting values from block: {block.shape=}')

        tx = np.floor((self.target_x[self.inbounds] - x0) / self.dx).astype(int)
        ty = np.floor((self.target_y[self.inbounds] - y0) / self.dy).astype(int)

        shape = (var.time.size, *self.target_x.shape)

        vo = np.full(shape, np.nan, dtype=block.dtype)
        vo[:, self.inbounds] = block.values[:, ty.ravel(), tx.ravel()]

        vo = xr.DataArray(vo,
                          [
                              ("time", var.time.data),
                              ("latitude", target.y),
                              ("longitude", target.x),
                          ],
                          attrs=var.attrs,
                          name=var.name)

        vo.latitude.attrs['units'] = 'degrees_north'
        vo.latitude.attrs['standard_name'] = 'latitude'
        vo.latitude.attrs['long_name'] = 'latitude'

        vo.longitude.attrs['units'] = 'degrees_east'
        vo.longitude.attrs['standard_name'] = 'longitude'
        vo.longitude.attrs['long_name'] = 'longitude'
        vo.attrs['grid_mapping'] = target.proj_name

        plt.figure()
        vo.isel(time=1).plot()
        logger.debug(f'Block ({block.shape}) -> vo ({vo.shape})')

        plt.show()
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
