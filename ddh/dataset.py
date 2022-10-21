import xarray as xr
import rioxarray as rxr

class Dataset:
    ds: xr.Dataset

    def __init__(self, url, variables=None):
        self.ds = xr.open_dataset(url, decode_coords='all')
