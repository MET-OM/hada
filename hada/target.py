import numpy as np
import logging
import pyproj
import xarray as xr

logger = logging.getLogger(__name__)


class Target:
    epsg = 3575
    crs = pyproj.CRS.from_epsg(epsg)

    proj_name = 'target_proj'
    grid_mapping_name = 'target_proj_grid'

    x: np.ndarray  #  projection horizontal coordinates
    y: np.ndarray  #  projection vertical coordinates

    xx: np.ndarray
    yy: np.ndarray

    @property
    def proj_var(self):
        """
        xarray projection definition (CF).
        """
        v = xr.DataArray(name=self.proj_name)
        v.attrs['grid_mapping_name'] = self.grid_mapping_name
        v.attrs['epsg'] = self.epsg
        return v

    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, output):
        """
        Args:

            xmin, ...: bounding box in target grid coordinates.
            nx, ny: grid cells
        """
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny = nx, ny
        self.output = output

        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        logger.info(
            f'Target grid set up: x={xmin, xmax}, y={ymin, ymax}, resolution: {nx} x {ny}, output: {output}'
        )

    @property
    def bbox(self):
        from shapely.geometry import box
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.epsg)

    @staticmethod
    def from_lonlat(lonmin, lonmax, latmin, latmax, *args, **kwargs):
        """
        Calculate coordinates in target projection and set up target grid. The corners
        [lonmin, latmin] -> [lonmax, latmax] will be converted to target projection
        and used as boudning box.
        """

        gcrs = Target.crs.geodetic_crs
        t = pyproj.Transformer.from_crs(gcrs, Target.crs, always_xy=True)

        x0, y0, x1, y1 = t.transform_bounds(lonmin, latmin, lonmax, latmax)
        logger.info(
            f'Transformed boundaries from {lonmin}E,{latmin}N - {lonmax}E,{latmax}N -> {x0,y0} - {x1,y1}'
        )

        return Target(x0, x1, y0, y1, *args, **kwargs)

    @staticmethod
    def transform(from_crs, x, y):
        t = pyproj.Transformer.from_crs(from_crs, Target.crs, always_xy=True)
        return t.transform(x, y)

    @staticmethod
    def itransform(to_crs, x, y):
        t = pyproj.Transformer.from_crs(Target.crs, to_crs, always_xy=True)
        return t.transform(x, y)

    @property
    def dx(self):
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self):
        return (self.ymax - self.ymin) / self.ny

    @staticmethod
    def modulate_longitude(lons, b180=True):
        """
        Modulate the input longitude to the domain supported by the reader.

        Args:
            lons: longitudes to be modulated.

            b180: True if extent is from -180 to 180, False if from 0 to 360

        Returns:

            lons: modulated longitudes.
        """

        if b180:
            # Domain is from -180 to 180 or somewhere in between.
            lons = np.mod(lons+180, 360) - 180
        else:
            # Domain is from 0 to 360 or somewhere in between.
            lons = np.mod(lons, 360)

        return lons
