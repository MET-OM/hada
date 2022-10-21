import pyproj

class Target:
    proj = pyproj.Proj('+proj=latlong +ellps=WGS84')

    def __init__(self, xmin, xmax, ymin, ymax, nx, ny):
        """
        Args:

            xmin, ...: bounding box in latitudes and longitudes.
            nx, ny: grid cells
        """
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny = nx, ny

    @property
    def dx(self):
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self):
        return (self.ymax - self.ymin) / self.ny
