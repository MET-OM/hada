import logging
from dataclasses import dataclass
from typing import List, Any, Dict
import toml
import numpy as np
import xarray as xr
import rioxarray as _
import cf_xarray as _
from pyproj.crs import CRS
from functools import cache

from .vector import rotate_vectors

logger = logging.getLogger(__name__)


class Dataset:
    name: str
    url: str
    variables: Dict
    ds: xr.Dataset

    # name of x and y vars
    x_v: str
    y_v: str

    x: np.ndarray
    y: np.ndarray
    kdtree: Any

    def __init__(self, name, url, x, y, variables, proj4=None):
        self.name = name
        self.url = url
        self.variables = variables
        self.x_v = x
        self.y_v = y

        logger.info(
            f'{self.name}: opening: {self.url} for variables: {self.variables}'
        )

        if '*' in url or isinstance(url, list):
            self.ds = xr.decode_cf(
                xr.open_mfdataset(
                    url,
                    decode_coords='all',
                    parallel=True,
                    # engine='h5netcdf',
                    chunks={
                        'time': 1,
                        self.x_v: 100,
                        self.y_v: 100
                    }))
        else:
            self.ds = xr.decode_cf(
                xr.open_dataset(url, decode_coords='all'))

        if x != 'X':
            self.ds = self.ds.rename_vars({self.x_v: 'X'})
            # self.ds = self.ds.rename_dims({self.x_v: 'X'})

        if y != 'Y':
            self.ds = self.ds.rename_vars({self.y_v: 'Y'})
            # self.ds = self.ds.rename_dims({self.y_v: 'Y'})

        self.x = self.ds['X'].values
        self.y = self.ds['Y'].values

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.xmin, self.xmax = self.x.min(), self.x.max()
        self.ymin, self.ymax = self.y.min(), self.y.max()

        if not all(np.diff(self.x) - self.dx == 0):
            logger.error(
                f'X coordinate not monotonic, max deviation from dx: {np.max(np.diff(self.x)-self.dx)}'
            )

        if not all(np.diff(self.y) - self.dy == 0):
            logger.error(
                f'Y coordinate not monotonic, max deviation from dy: {np.max(np.diff(self.y)-self.dy)}'
            )

        logger.debug(
            f'x: {self.x.shape} / {self.dx}, y: {self.y.shape} / {self.dy}')
        logger.debug(
            f'x: {self.x.min()} -> {self.x.max()}, y: {self.y.min()} -> {self.y.max()}'
        )

        dt = (self.ds.time.values[1] -
              self.ds.time.values[0]) / np.timedelta64(1, 'h')
        logger.info(
            f'time: {self.ds.time.values[0]} -> {self.ds.time.values[-1]} (dt: {dt} h)'
        )

        if proj4 is not None:
            self.crs = CRS.from_proj4(proj4)
        else:
            self.crs = self.ds.rio.crs

        logger.debug(f'CRS: {self.crs}')

    def __repr__(self):
        return f'<Dataset ({self.name} / {self.url})>'

    @cache
    def __calculate_grid__(self, target):
        logger.debug(f'Calculating grid for target: {target.xx.shape}..')

        # Calculating the location of the target grid cells
        # in this datasets coordinate system.
        target_x, target_y = target.itransform(self.crs, target.xx.ravel(),
                                               target.yy.ravel())

        assert len(target_x) > 0

        target_x.shape = target.xx.shape
        target_y.shape = target.yy.shape

        # Target coordinates within source domain
        inbounds = (target_x >= self.xmin) & (target_x < self.xmax) & (
            target_y >= self.ymin) & (target_y < self.ymax)

        return target_x, target_y, inbounds

    def regrid(self, var, target, time):
        """
        Return values for the target grid.
        """
        time = np.atleast_1d(time)
        logger.info(
            f'Regridding {var} between {np.min(time)} and {np.max(time)}')

        target_x, target_y, inbounds = self.__calculate_grid__(target)

        if not any(inbounds.ravel()):
            logger.warning('Target is outside the domain of this reader')
            return None

        if np.min(time) > var.time[-1] or np.max(time) < var.time[0]:
            logger.warning(
                'Target time is outside the time span of this reader')
            return None

        logger.info('Selecting time slice..')
        var = var.sel(time=time, method='nearest')

        ## Reduce and remove unwanted dimensions (selecting first item)
        if 'depth' in var.dims:
            logger.info('Selecting depth0..')
            var = var.sel(depth=0)

        if 'height0' in var.dims:
            var = var.isel(height0=0)

        if 'height1' in var.dims:
            var = var.isel(height1=0)

        if 'height2' in var.dims:
            var = var.isel(height2=0)

        if 'ensemble_member' in var.dims:
            var = var.isel(ensemble_member=0)

        # Extract block
        x0 = np.min(target_x[inbounds]) - np.abs(self.dx)
        x1 = np.max(target_x[inbounds]) + np.abs(self.dx)
        y0 = np.min(target_y[inbounds]) - np.abs(self.dy)
        y1 = np.max(target_y[inbounds]) + np.abs(self.dy)

        swap_y = self.dy < 0
        swap_x = self.dx < 0

        # Shifted indices (XXX: is this flipped somehow when swap_*?)
        tx = np.floor((target_x[inbounds] - x0) / np.abs(self.dx)).astype(int)
        ty = np.floor((target_y[inbounds] - y0) / np.abs(self.dy)).astype(int)

        if swap_y:
            logger.debug('y is decreasing, swapping direction.')
            y1, y0 = y0, y1

        if swap_x:
            logger.debug('x is decreasing, swapping direction.')
            x1, x0 = x0, x1

        logger.debug(
            f'Load block between x: {x0}..{x1}/{self.dx}, y: {y0}..{y1}/{self.dy}'
        )
        block = var.sel(X=slice(x0, x1), Y=slice(y0, y1)).load()

        if swap_y:
            block = block[..., ::-1, :]

        if swap_x:
            block = block[..., :, ::-1]

        logger.debug(f'Extracting values from block: {block.shape=}')

        shape = list(block.shape)[:-2] + list(target_x.shape)
        shape = tuple(shape)
        logger.debug(f'New shape: {shape}')

        vo = np.full(shape, np.nan, dtype=block.dtype)
        vo[..., inbounds] = block.values[..., ty.ravel(), tx.ravel()]

        # Fill invalid times with nans
        invalid = np.logical_or(time > var.time.values[-1],
                                time < var.time.values[0])
        vo[invalid, ...] = np.nan

        # Construct new coordinates
        coords = {'time': time}
        coords['Y'] = ("Y", target.y)
        coords['X'] = ("X", target.x)

        vo = xr.DataArray(vo,
                          dims=('time', 'Y', 'X'),
                          coords=coords,
                          attrs=var.attrs,
                          name=var.name)

        vo.attrs['grid_mapping'] = target.proj_name
        vo.attrs['source'] = self.url
        vo.attrs['source_name'] = self.name

        logger.debug(f'Block ({block.shape}) -> vo ({vo.shape})')

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
        return self.variables.get(var, None)


@dataclass
class Sources:
    scalar_variables: List[str]
    derived_variables: Dict
    vector_magnitude_variables: Dict
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
            var1 = d.get_var(var1)
            var2 = d.get_var(var2)

            if var1 is not None and var2 is not None:
                return (d, d.ds[var1], d.ds[var2])

        return (None, None, None)

    def regrid(self, var, target, time):
        """
        Search through datasets and try to cover the entire target grid with data.
        """
        vo = None

        for d in self.datasets:
            v = d.get_var(var)

            if v is not None:
                logger.info(f'Found {var} in {d}..')
                v = d.ds[v]

                vod = d.regrid(v, target, time)
                if vod is not None:
                    if vo is None:
                        vo = vod
                    else:
                        assert vo.shape == vod.shape
                        td = np.isnan(vo.values) & ~np.isnan(vod.values)
                        logger.info(
                            f'Merging {len(td[td])} values into output variable: {vo.shape}'
                        )
                        vo.values[td] = vod.values[td]
                else:
                    logger.debug(f'{var} completely out of domain of {d}.')

            if vo is not None and not np.isnan(vo).any():
                logger.debug(f'{var} completely covered.')
                break

        return vo

    @staticmethod
    def from_toml(file, dataset_filter=(), variable_filter=()):
        logger.info(f'Loading sources from {file}')
        d = toml.load(open(file))

        datasets = []

        for name, ds in d['datasets'].items():
            if len(dataset_filter) > 0:
                if not any(map(lambda f: f in name, dataset_filter)):
                    continue

            dataset = Dataset(name=name, **ds)

            datasets.append(dataset)

        scalar_vars = d['scalar_variables']
        derived_vars = d['derived_variables']
        vector_mag_vars = d['vector_magnitude_variables']

        if len(variable_filter) > 0:
            logger.debug(
                f'Filtering scalar variables: {scalar_vars} | {variable_filter}'
            )
            scalar_vars = list(
                filter(lambda v: any(map(lambda f: f in v, variable_filter)),
                       scalar_vars))
            logger.debug(f'New scalar variables: {scalar_vars}.')

            logger.debug(
                f'Filtering vector variables: {vector_mag_vars.keys()} | {variable_filter}'
            )

            fvector_mag_vars = list(
                filter(lambda v: any(map(lambda f: f in v, variable_filter)),
                       vector_mag_vars))
            new_v_m = dict()
            for k in fvector_mag_vars:
                new_v_m[k] = vector_mag_vars[k]
            vector_mag_vars = new_v_m

            logger.debug(f'New vector variables: {vector_mag_vars}.')

            logger.debug(
                f'Filtering derived variables: {derived_vars.keys()} | {variable_filter}'
            )

            fderived_vars = list(
                filter(lambda v: any(map(lambda f: f in v, variable_filter)),
                       derived_vars))
            new_d_v = dict()
            for k in fderived_vars:
                new_d_v[k] = derived_vars[k]
            derived_vars = new_d_v

            logger.debug(f'New derived variables: {derived_vars}.')

        return Sources(scalar_variables=scalar_vars,
                       vector_magnitude_variables=vector_mag_vars,
                       derived_variables=derived_vars,
                       datasets=datasets)
