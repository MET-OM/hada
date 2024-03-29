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
        logger.warning(
            'horizontal visibilty: rh or fog completely nan filled, setting to NaN'
        )
        return xr.DataArray(np.full(rh.shape, np.nan), dims=rh.dims)

    if np.nanmax(fog) > 1.0 or np.nanmin(fog) < 0.0:
        logger.error('fog fraction is not between 0 and 1')

    # rh can be over-saturated, i.e.: more than 1.
    # assert rh.max() <= 1.0 and rh.min() >= 0.0, "relative humidity must be number between 0 and 1"
    """
    Calculates horizontal visibility based on a modified version of a formula proposed by Gultepe and Milbrandt (2010)

    rh : in percent
    """

    rh = rh * 100.
    vv = (-5.19e-10) * (rh**5.44) + 40.1

    ## Adjust for fog: set visibilty to zero when more than 90% fog.
    vv.values[np.logical_and(fog > 0.9, vv > 1)] = 0.

    return vv


def hor_vis_rh(rh):
    rh = rh * 100.
    vv = (-5.19e-10) * (rh**5.44) + 40.1
    return vv


def hor_vis_tdta_C(dew_point_temp, air_temp):
    """
    Calculate visibilty based on dew point temp and air temp.

    Temperatures in degrees Celsius.

    Returns visibility in meters.
    """
    td = dew_point_temp
    ta = air_temp

    e = 6.11 * 10.**(7.5 * td / (237.5 + td))
    es = 6.11 * 10.**(7.5 * ta / (237.5 + ta))
    rh = 100. * e / es
    vs = -0.000114 * rh**2.715 + 27.0
    vs.values[vs < 0.05] = 0.05
    vs = vs * 1.e3

    return vs

def hor_vis_tdta_K(dew_point_temp, air_temp):
    return hor_vis_tdta_C(dew_point_temp - 237.5, air_temp - 237.5)


def derive(var, *args):
    if var == 'horizontal_visibility':
        return hor_vis(*args)
    elif var == 'horizontal_visibility_rh':
        return hor_vis_rh(*args)
    elif var == 'horizontal_visibility_dew_C':
        return hor_vis_tdta_C(*args)
    elif var == 'horizontal_visibility_dew_K':
        return hor_vis_tdta_K(*args)
    else:
        raise ValueError(f"Unknown derived variable: {var}")

