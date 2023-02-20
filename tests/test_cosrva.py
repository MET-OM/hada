import pytest
import os
from hada.sources import *
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from hada.cli import hada

if not os.path.exists('/lustre/storeB/project/fou/om/ERA/ERA5/atm'):
    pytest.skip("dataset not accessible, skipping dependent tests", allow_module_level=True)

def test_load_cosrva(cosrva_local_toml):
    s = Sources.from_toml(cosrva_local_toml)
    print(s)

def test_cosrva_sst(cosrva_local_toml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, ['--sources', str(cosrva_local_toml), '--output', td / 'era.nc', '--bbox-deg', '6,7,67,68', '--from', '2019-05-30', '--to', '2019-06-02', '--freq', '6H', '-v', 'sea_temp'])
        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'era.nc')
        print(ds)

        assert ds['sea_temperature'] is not None
        assert not np.isnan(ds['sea_temperature'].values).all()

