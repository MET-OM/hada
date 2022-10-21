import logging
from dataclasses import dataclass
from typing import List
import toml

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    name: str
    url: str

@dataclass
class Sources:
    variables: List[str]
    datasets: List[Dataset]

    @staticmethod
    def from_toml(file):
        logger.info(f'Loading sources from {file}')
        d = toml.load(open(file))
        datasets = [Dataset(name, **d) for name, d in d['datasets'].items()]
        return Sources(d['variables'], datasets)

