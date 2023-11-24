import numpy as np
import logging
import pyproj
import xarray as xr

logger = logging.getLogger(__name__)


class Target:
    DEFAULT_EPSG = 3575
    epsg = DEFAULT_EPSG
    crs = pyproj.CRS.from_epsg(epsg)

    proj_name = 'target_proj'
    grid_mapping_name = 'target_proj_grid'

    x: np.ndarray  #  projection horizontal coordinates
    y: np.ndarray  #  projection vertical coordinates

    xx: np.ndarray
    yy: np.ndarray

    grid_id = None

    @property
    def proj_var(self):
        """
        xarray projection definition (CF).
        """
        v = xr.DataArray(name=self.proj_name)
        v.attrs['grid_mapping_name'] = self.grid_mapping_name
        v.attrs['epsg'] = self.epsg

        v = [v]

        if self.grid_id is not None:
            vlon = xr.DataArray(name='lons',
                                data=self.lon,
                                coords=[('grid_id', self.grid_id)])
            vlat = xr.DataArray(name='lats',
                                data=self.lat,
                                coords=[('grid_id', self.grid_id)])
            vx = xr.DataArray(name='X',
                              data=self.x,
                              coords=[('grid_id', self.grid_id)])
            vy = xr.DataArray(name='Y',
                              data=self.y,
                              coords=[('grid_id', self.grid_id)])

            v.extend([vlon, vlat, vx, vy])

        return v

    def __init__(self, output, epsg=None):
        self.output = output
        self.epsg = epsg if epsg is not None else Target.DEFAULT_EPSG
        self.crs = pyproj.CRS.from_epsg(self.epsg)

    @staticmethod
    def from_box(xmin, xmax, ymin, ymax, nx, ny, output, epsg=None):
        """
        Args:

            xmin, ...: bounding box in target grid coordinates.
            nx, ny: grid cells
        """
        t = Target(output, epsg)
        t.xmin, t.xmax, t.ymin, t.ymax = xmin, xmax, ymin, ymax
        t.nx, t.ny = nx, ny

        t.x = np.linspace(xmin, xmax, nx)
        t.y = np.linspace(ymin, ymax, ny)
        t.xx, t.yy = np.meshgrid(t.x, t.y)

        logger.info(
            f'Target grid set up: x={xmin, xmax}, y={ymin, ymax}, resolution: {nx} x {ny}, output: {output}'
        )

        return t

    @property
    def bbox(self):
        from shapely.geometry import box
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.epsg)

    @staticmethod
    def from_gridfile(fname, output, epsg=None):
        """
        Parse DNV CSV grid file.
        """
        import pandas as pd
        grid = pd.read_csv(fname, sep=',')

        print(grid)

        xmin = grid['X'].min()
        xmax = grid['X'].max()
        ymin = grid['Y'].min()
        ymax = grid['Y'].max()

        nx = len(grid['X'])
        ny = 1

        x = grid['X'].to_numpy()
        y = grid['Y'].to_numpy()

        t = Target(output, epsg)
        t.xmin, t.xmax, t.ymin, t.ymax = xmin, xmax, ymin, ymax
        t.nx, t.ny = nx, ny
        t.x = x
        t.y = y
        t.xx = x
        t.yy = y

        t.grid_id = grid['GRID_ID'].to_numpy()
        t.lon = grid['Long_WGS84'].to_numpy()
        t.lat = grid['Lat_WGS84'].to_numpy()

        return t

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

        return Target.from_box(x0, x1, y0, y1, *args, **kwargs)

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
            lons = np.mod(lons + 180, 360) - 180
        else:
            # Domain is from 0 to 360 or somewhere in between.
            lons = np.mod(lons, 360)

        return lons
