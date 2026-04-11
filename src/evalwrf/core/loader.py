import numpy as np
import xarray as xr
from pathlib import Path
from typing import Literal


def get_domain_files(
    foldername: str,
    domain: Literal["d01", "d02", "d03", "d04", "d05"],
    input: bool = False,
) -> list | Path:
    if not input:
        return sorted(Path(foldername).glob(f"wrfout*{domain}*"))
    else:
        return sorted(Path(foldername).glob(f"wrfin*{domain}*"))[0]


def _select_variables(ds: xr.Dataset, vars: list | None):
    DROP_COORDS = ["XLONG_U", "XLONG_V", "XLAT_U", "XLAT_V"]
    DROP_DIMS = ["west_east_stag", "south_north_stag"]

    if vars is not None:
        to_drop = np.setxor1d(list(ds.data_vars), vars)
        ds = ds.drop_vars(names=to_drop)

    # ds = ds.drop_dims(DROP_DIMS, errors="ignore")
    # ds = ds.drop_vars(DROP_COORDS, errors="ignore")
    return ds


def wrf_to_xr(files: list, variables: list | None = None):
    ds = xr.open_mfdataset(
        files,
        concat_dim="Time",
        combine="nested",
        preprocess=lambda ds: _select_variables(ds, variables),
    )
    ds["Time"] = ds["XTIME"]
    ds = ds.assign_coords(
        {
            "lat": (["south_north", "west_east"], ds.XLAT[0, ...].values),
            "lon": (["south_north", "west_east"], ds.XLONG[0, ...].values),
        }
    )
    ds = ds.drop_vars(["XLAT", "XLONG", "XTIME"], errors="ignore")
    return ds
