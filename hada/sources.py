import logging
from dataclasses import dataclass
from typing import List, Any, Dict
import toml
import numpy as np
import xarray as xr
import rioxarray as _
import cf_xarray as _
import pyproj
from functools import cache

from .vector import rotate_vectors

logger = logging.getLogger(__name__)


class Dataset:
    name: str
    url: str
    variables: List[str]
    ds: xr.Dataset
    variable_map: Dict

    # name of x and y vars
    x_v: str
    y_v: str

    x: np.ndarray
    y: np.ndarray
    kdtree: Any

    def __init__(self,
                 name,
                 url,
                 x,
                 y,
                 global_variables,
                 variable_map=None,
                 variables=None):
        self.name = name
        self.url = url
        self.variables = variables if variables is not None else global_variables
        self.variable_map = variable_map
        self.x_v = x
        self.y_v = y

        logger.info(
            f'{self.name}: opening: {self.url} for variables: {self.variables}'
        )
        self.ds = xr.decode_cf(xr.open_dataset(url, decode_coords='all'))
        if x != 'X':
            self.ds = self.ds.rename_vars({self.x_v: 'X'})
            # self.ds = self.ds.rename_dims({self.x_v: 'X'})

        if y != 'Y':
            self.ds = self.ds.rename_vars({self.y_v: 'Y'})
            # self.ds = self.ds.rename_dims({self.y_v: 'Y'})

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
    def __calculate_grid__(self, target):
        logger.debug(f'Calculating grid for target: {target.xx.shape}..')

        # Calculating the location of the target grid cells
        # in this datasets coordinate system.
        # tf = pyproj.Transformer.from_proj(target.crs, self.crs)

        # self.target_x, self.target_y = tf.transform(
        #     target.xx.ravel(), target.yy.ravel())

        target_x, target_y = self.crs(target.xx.ravel(),
                                      target.yy.ravel(),
                                      inverse=False)
        target_x.shape = target.xx.shape
        target_y.shape = target.yy.shape

        # Target coordinates within source domain
        inbounds = (target_x >= self.xmin) & (target_x < self.xmax) & (
            target_y >= self.ymin) & (target_y < self.ymax)

        if not any(inbounds.ravel()):
            logger.warning('Target is outside the domain of this reader')

        return target_x, target_y, inbounds

    def regrid(self, var, target, t0, t1):
        """
        Return values for the target grid.
        """
        logger.info(f'Regridding {var} between {t0} and {t1}')

        target_x, target_y, inbounds = self.__calculate_grid__(target)

        logger.info('Selecting time slice..')
        var = var.sel(time=slice(t0, t1))

        if 'depth' in var.dims:
            logger.info('Selecting depth0..')
            var = var.sel(depth=0)

        # Extract block
        x0 = np.min(target_x[inbounds]) - self.dx
        x1 = np.max(target_x[inbounds]) + self.dx
        y0 = np.min(target_y[inbounds]) - self.dy
        y1 = np.max(target_y[inbounds]) + self.dy

        logger.debug(f'Load block between x: {x0}..{x1}, y: {y0}..{y1}')
        block = var.sel(X=slice(x0, x1), Y=slice(y0, y1)).load()
        # block.isel(time=1).plot()

        logger.debug(f'Extracting values from block: {block.shape=}')

        tx = np.floor((target_x[inbounds] - x0) / self.dx).astype(int)
        ty = np.floor((target_y[inbounds] - y0) / self.dy).astype(int)

        shape = list(block.shape)[:-2] + list(target_x.shape)
        shape = tuple(shape)
        logger.debug(f'New shape: {shape}')

        vo = np.full(shape, np.nan, dtype=block.dtype)
        vo[..., inbounds] = block.values[..., ty.ravel(), tx.ravel()]

        coords = var.coords
        coords = dict(var.coords)
        print(coords)
        del coords['X']
        del coords['Y']
        print(coords)

        coords['latitude'] = ("latitude", target.y)
        coords['longitude'] = ("longitude", target.x)


        vo = xr.DataArray(vo, coords=coords,
                          attrs=var.attrs,
                          name=var.name)

        # Positions in source grid
        # vo.attrs['x'] = target_x
        # vo.attrs['y'] = target_y

        vo.latitude.attrs['units'] = 'degrees_north'
        vo.latitude.attrs['standard_name'] = 'latitude'
        vo.latitude.attrs['long_name'] = 'latitude'

        vo.longitude.attrs['units'] = 'degrees_east'
        vo.longitude.attrs['standard_name'] = 'longitude'
        vo.longitude.attrs['long_name'] = 'longitude'
        vo.attrs['grid_mapping'] = target.proj_name
        vo.attrs['source'] = self.url
        vo.attrs['source_name'] = self.name

        # plt.figure()
        # vo.isel(time=1).plot()
        logger.debug(f'Block ({block.shape}) -> vo ({vo.shape})')

        # plt.show()
        return vo

    def rotate_vectors(self, vx, vy, target):
        x, y, _ = self.__calculate_grid__(target)
        vox, voy = rotate_vectors(x, y, vx.values, vy.values, self.crs,
                                  target.crs)
        vx.values = vox
        vy.values = voy

        return vx, vy

    def get_var(self, var):
        """
        Return variable name for input variable.
        """
        logger.debug(f'Looking for {var} in {self}')
        if var in self.variables:
            if var in self.variable_map:
                logger.debug(f'Variable map: {var} -> {self.variable_map[var]}')
                return self.variable_map[var]
            else:
                if self.ds.cf[var] is not None:
                    return self.ds.cf[var].name


@dataclass
class Sources:
    scalar_variables: List[str]
    vector_variables: List[str]
    datasets: List[Dataset]

    def find_dataset_for_var(self, var):
        """
        Find first dataset with variable.
        """
        for d in self.datasets:
            v = d.get_var(var)
            if v is not None:
                return (d, d.ds[v])

        return (None, None)

    def find_dataset_for_var_pair(self, var1, var2):
        """
        Find first dataset with both variables.
        """
        for d in self.datasets:
            logger.debug(f'Looking for {var1} and {var2} in {d}')
            if var1 in d.variables and var2 in d.variables:
                if d.ds.cf[[var1]] is not None and d.ds.cf[[var2]] is not None:
                    print(d.ds.cf[[var1]])
                    return (d, d.ds.cf[[var1]], d.ds.cf[[var2]])

        return (None, None, None)

    @staticmethod
    def from_toml(file, dataset_filter=(), variable_filter=()):
        logger.info(f'Loading sources from {file}')
        d = toml.load(open(file))

        global_variables = d['scalar_variables'] + [
            v for l in d['vector_variables'] for v in l
        ]

        if len(variable_filter) > 0:
            logger.debug(f'Filtering variables: {variable_filter}')
            global_variables = list(
                filter(lambda v: any(map(lambda f: f in v, variable_filter)),
                       global_variables))

        datasets = [
            Dataset(name=name, **d, global_variables=global_variables)
            for name, d in d['datasets'].items() if len(dataset_filter) == 0
            or any(map(lambda f: f in name, dataset_filter))
        ]
        return Sources(scalar_variables=d['scalar_variables'],
                       vector_variables=d['vector_variables'],
                       datasets=datasets)
