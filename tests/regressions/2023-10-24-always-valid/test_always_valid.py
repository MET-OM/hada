from pathlib import Path
from hada.cli import hada
import os.path, os, subprocess
import xarray as xr
import numpy as np
import sys
import pytest

@pytest.mark.skipif(sys.platform != 'linux', reason='need linux utils')
def test_always_valid_hs(test_dir, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        testd = test_dir / 'regressions' / '2023-10-24-always-valid'

        sourced = Path('/tmp/hada/2023-10-24')
        if not sourced.exists():
            os.makedirs(sourced)

        era1 = sourced / 'era1.nc'
        era2 = sourced / 'era2.nc'

        if not os.path.exists(era1):
            subprocess.check_call("""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_0HFwzoBFZtglnV52Lt-PcMX4AOk98jr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_0HFwzoBFZtglnV52Lt-PcMX4AOk98jr" -O era1.nc && rm /tmp/cookies.txt""", shell=True, cwd=sourced)

        if not os.path.exists(era2):
            subprocess.check_call("""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18-6FjHuOBuR1xpuQiIn_0Z59mz91-WTR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18-6FjHuOBuR1xpuQiIn_0Z59mz91-WTR" -O era2.nc && rm /tmp/cookies.txt""", shell=True, cwd=sourced)


        r = runner.invoke(hada, ['--sources', str(testd / 'scosrva23_local.toml'), '--grid', str(testd / 'GRID_SCOSRVA_Kyst_EPSG3575.csv'), '--output', td / 'test.nc', '--from', '2022-05-01', '--to', '2022-05-05', '--always-valid' ])
        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'test.nc')
        print(ds)

        assert 'sea_temperature' in ds
        assert 'air_temperature' in ds
        assert 'significant_wave_height' in ds
        assert 'ice_concentration' in ds

        assert np.all(~np.isnan(ds['air_temperature'].values))
        assert np.all(~np.isnan(ds['sea_temperature'].values))
        assert np.all(~np.isnan(ds['significant_wave_height'].values))

