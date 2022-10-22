import logging
from dataclasses import dataclass
from typing import List
import toml
import xarray as xr
import rioxarray as _
import cf_xarray as _

logger = logging.getLogger(__name__)

class Dataset:
    name: str
    url: str
    variables: List[str]
    ds: xr.Dataset

    def __init__(self, name, url, variables):
        self.name = name
        self.url = url
        self.variables = variables if variables else []

        logger.info(f'{self.name}: opening: {self.url} for variables: {self.variables}')
        self.ds = xr.open_dataset(url, decode_coords='all')

@dataclass
class Sources:
    variables: List[str]
    datasets: List[Dataset]

    @staticmethod
    def from_toml(file):
        logger.info(f'Loading sources from {file}')
        d = toml.load(open(file))

        global_variables = d['variables']

        datasets = [Dataset(name=name, url=d['url'], variables=d.get('variables', global_variables)) for name, d in d['datasets'].items()]
        return Sources(d['variables'], datasets)


