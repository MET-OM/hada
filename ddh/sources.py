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
        self.ds = xr.open_dataset(url, decode_coords='all')

        # TODO: likely to be specific to dataset
        self.x = self.ds['X'].values
        self.y = self.ds['Y'].values
        logger.debug(f'Setting up KDTree for coordinates')
        self.kdtree = KDTree(np.hstack((self.x, self.y)))

        self.crs = pyproj.Proj(self.ds.rio.crs.to_proj4())
        logger.debug(f'CRS: {self.crs}')

    def __repr__(self):
        return f'<Dataset ({self.name} / {self.url})>'

    def calculate_grid(self, target):
        target_hash = hash(target)
        if self.target_hash != target_hash:
            logger.debug(f'Calculating grid for target..')

            # Calculating the location of the target grid cells
            # in this datasets coordinate system.
            tf = pyproj.Transformer.from_proj(target.crs, self.crs)

            self.target_x, self.target_y = tf.transform(
                target.xx.ravel(), target.yy.ravel())

            self.target_hash = target_hash

        return self.target_x, self.target_y

    def regrid(self, var):
        pass


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
