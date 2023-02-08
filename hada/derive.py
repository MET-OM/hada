import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)

def hor_vis(rh, fog):
    """
    Calculate horizontal visibilty based on relative humidity and fog fraction.
    """
    assert rh.shape == fog.shape, "rh and fog must be same dimensions"

    if not np.isfinite(rh).any() or not np.isfinite(fog).any():
        logger.warning('horizontal visibilty: rh or fog completely nan filled, setting to NaN')
        return xr.DataArray(np.full(rh.shape, np.nan), dims=rh.dims)

    assert np.nanmax(fog) <= 1.0 and np.nanmin(fog) >= 0.0, "fog fraction must be number between 0 and 1"

    # rh can be over-saturated, i.e.: more than 1.
    # assert rh.max() <= 1.0 and rh.min() >= 0.0, "relative humidity must be number between 0 and 1"

    """
    Calculates horizontal visibility based on a modified version of a formula proposed by Gultepe and Milbrandt (2010)

    rh : in percent
    """

    rh = rh * 100.
    vv = (-5.19e-10)*(rh**5.44) + 40.1

    ## Adjust for fog: set visibilty to zero when more than 90% fog.
    vv.values[np.logical_and(fog>0.9, vv>1)] = 0.

    return vv

def derive(var, *args):
    if var == 'horizontal_visibility':
        return hor_vis(*args)
    else:
        raise ValueError(f"Unknown derived variable: {var}")

