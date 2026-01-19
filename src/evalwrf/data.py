from typing import Literal
import pandas as pd
import xarray as xr
from .api import load_metadata


def load_dataset_from_csv(
    filename: str, configfile: str, na_action: Literal["drop", "raise", "ignore"] = "drop"
) -> xr.Dataset:
    df = pd.read_csv(filename)
    if na_action == "drop":
        df = df.dropna()
    elif (na_action == "raise") and df.isna().any():
        raise ValueError("NaNs are present in the dataset!")
    else:
        pass

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)

    station_meta = load_metadata(configfile, record="stations")
    df["station"] = df["station"].map(station_meta["name"])
    df = df.set_index(["station", "time"])
    df = df.to_xarray()

    para_meta = load_metadata(configfile, record="parameters")
    for var in df.data_vars:
        df[var].attrs = para_meta.loc[var, :].to_dict()
        df[var].attrs["label"] = df[var].attrs["long_name"] + " " + f"({df[var].attrs['unit']})"
    return df
