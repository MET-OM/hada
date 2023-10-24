import pytest
import numpy as np
import xarray as xr

from hada.derive import *

def test_horizontal_vis_meps():
    ds = xr.open_dataset("https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc")

    ds = ds.isel(time=0)

    rh = ds.relative_humidity_2m.isel(height0=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))
    fog = ds.fog_area_fraction.isel(height1=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))

    assert (fog>0.9).any()

    vv = hor_vis(rh, fog)
    print(vv)

def test_horizontal_vis_meps_tdta():
    ds = xr.open_dataset("https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc")

    ds = ds.isel(time=0)

    rh = ds.relative_humidity_2m.isel(height0=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))
    fog = ds.fog_area_fraction.isel(height1=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))

    assert (fog>0.9).any()
    vv = hor_vis(rh, fog)
    print(vv)

    ta = ds.air_temperature_2m.isel(height0=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))
    rh = ds.relative_humidity_2m.isel(height0=0, ensemble_member=0).isel(x=slice(0, 10000), y=slice(0,10000))
    td = ta - ((100 - rh*100)/5.)

    vv = hor_vis_tdta_C(td, ta)
    print(vv)




