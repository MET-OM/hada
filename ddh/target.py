import numpy as np
import logging
import pyproj
import xarray as xr

logger = logging.getLogger(__name__)

class Target:
    crs = pyproj.Proj('epsg:4326')

    proj_name = 'latlon_proj'
    grid_mapping_name = 'latitude_longitude'

    x: np.ndarray  #  longitudes
    y: np.ndarray  #  latitudes

    xx: np.ndarray
    yy: np.ndarray

    def proj_var(self):
        """
        xarray projection definition (CF).
        """
        v = xr.DataArray(name=self.proj_name)
        v.attrs['grid_mapping_name'] = 'latitude_longitude'
        return v

    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, output):
        """
        Args:

            xmin, ...: bounding box in latitudes and longitudes.
            nx, ny: grid cells
        """
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny = nx, ny
        self.output = output

        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        logger.info(f'Target grid set up: {xmin, xmax, ymin, ymax}, resolution: {nx} x {ny}, output: {output}')

    @property
    def dx(self):
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self):
        return (self.ymax - self.ymin) / self.ny
