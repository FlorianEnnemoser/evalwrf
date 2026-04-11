import xarray as xr
import numpy as np


def ms2kmh(ms: float) -> float:
    return ms * 3.6


def destagger(da: xr.DataArray, dim: str):
    """
    copied https://github.com/NCAR/wrf-python/blob/develop/src/wrf/destag.py
    and added dimension string handling for human readable dimensions
    """

    if "_stag" not in dim:
        raise ValueError("Not a staggered dimension.")

    if not isinstance(da, xr.DataArray):
        raise ValueError("Only DataArrays supported (Dataset has no shape attribute)!")

    da_shape = da.shape
    ndims = da.ndim
    stagger_dim = da.get_axis_num(dim)
    stagger_dim_size = da_shape[stagger_dim]

    full_slice = slice(None)
    slice1 = slice(0, stagger_dim_size - 1, 1)
    slice2 = slice(1, stagger_dim_size, 1)

    # default to full slices
    dim_ranges_1 = [full_slice] * ndims
    dim_ranges_2 = [full_slice] * ndims

    # for the stagger dim, insert the appropriate slice range
    dim_ranges_1[stagger_dim] = slice1
    dim_ranges_2[stagger_dim] = slice2

    result = 0.5 * (da[tuple(dim_ranges_1)].values + da[tuple(dim_ranges_2)].values)

    result = xr.DataArray(result, name=da.name, dims=da.dims, attrs=da.attrs)

    return result.rename({dim: dim.split("_stag")[0]})


def _meter2lat(meter: float) -> float:
    """Convert a north-south distance in metres to degrees latitude.

    Uses the approximation 1° latitude ≈ 110 574 m.

    Parameters
    ----------
    meter : float
        Distance in metres.

    Returns
    -------
    float
        Equivalent latitude in degrees.
    """
    return meter / (110.574e3)


def _meter2lon(meter: float, lat: float) -> float:
    """Convert an east-west distance in metres to degrees longitude.

    Accounts for the cosine factor at the given latitude.

    Parameters
    ----------
    meter : float
        Distance in metres.
    lat : float
        Reference latitude in degrees North.

    Returns
    -------
    float
        Equivalent longitude in degrees at *lat*.
    """
    return meter / (111.32e3 * np.cos(np.deg2rad(lat)))
