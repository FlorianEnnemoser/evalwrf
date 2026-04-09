from typing import Literal

from ..api import URL
from .helpers import load_json, save_data_stream
from pathlib import Path
import pandas as pd
import xarray as xr
import httpx
import json


def load_url_for_resource(filename: str, resource: str) -> URL | None:
    """Look up a dataset URL from a JSON config file by resource name.

    Searches the top-level keys of *filename* for a key containing *resource*
    and returns the corresponding ``"url"`` value wrapped in a :class:`URL`.

    Parameters
    ----------
    filename : str
        Path to the JSON file containing resource-to-URL mappings.
    resource : str
        Substring matched against the top-level keys of the JSON object.

    Returns
    -------
    URL or None
        A :class:`URL` for the first matching resource, or ``None`` if no key
        contains *resource*.

    Examples
    --------
    >>> url = load_url_from_resource("config/Datasets.json", "klima-v2-10min")
    >>> url
    URL('https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-10min')
    """
    data = load_json(filename)
    for k, v in data.items():
        if resource in k:
            return URL(v.get("url", ""))
    return None


def load_metadata(
    filename: str,
    record: Literal["parameters", "stations"],
) -> pd.DataFrame:
    """Load metadata from a locally saved JSON file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    filename : str
        Path to the JSON metadata file (typically saved with
        :func:`save_json_from_URL`).
    record : {"parameters", "stations"}
        Record type to extract.  ``"parameters"`` uses ``"name"`` as the
        index; ``"stations"`` uses ``"id"`` as the index.

    Returns
    -------
    pandas.DataFrame
        Normalised, indexed metadata table.

    Examples
    --------
    >>> df = load_metadata("Metadata_klima-v2-10min.json", "stations")
    """
    data = load_json(filename)
    idx = "name" if record == "parameters" else "id"
    df = pd.json_normalize(data, record_path=record).set_index(idx)
    return df


def save_json_from_URL(url: URL, filename: str, **kwargs) -> None:
    """Fetch a JSON endpoint and write the response body to *filename*.json.

    Parameters
    ----------
    url : URL
        Endpoint to request.  The HTTP response must return status 200.
    filename : str
        Destination filename **without** the ``.json`` extension.
    **kwargs
        Additional keyword arguments forwarded to :func:`httpx.get`.

    Raises
    ------
    ValueError
        If the HTTP response status code is not 200.

    Examples
    --------
    >>> url = load_url_from_resource("Datasets.json", "klima-v2-10min")
    >>> save_json_from_URL(url / "metadata", "Metadata_klima-v2-10min")
    """
    response = httpx.get(url, **kwargs)

    if response.status_code != 200:
        raise ValueError(
            f"Unexpected status code {response.status_code} for URL: {url}"
        )

    with open(f"{filename}.json", "w") as f:
        json.dump(response.json(), f, indent=4)
    return None


def save_data(url: URL, filename: str, **kwargs) -> None:
    """Stream a NetCDF file from *url* and write it to *filename*.

    Uses chunked streaming to avoid loading large files into memory.

    Parameters
    ----------
    url : URL
        The NetCDF download URL.
    filename : str
        Destination file path including extension (e.g. ``"output.nc"``).
    **kwargs
        Additional keyword arguments forwarded to :func:`httpx.stream`.
        Commonly used to pass ``params`` with query parameters.

    Examples
    --------
    >>> save_data(
    ...     url,
    ...     filename="murau_data.csv",
    ...     params=dict(
    ...         start="2025-12-30",
    ...         end="2026-01-01",
    ...         parameters=["TL", "RR"],
    ...         station_ids=15920,
    ...         output_format="csv",
    ...     ),
    ... )
    """
    if Path(filename).suffix not in [".nc", ".csv"]:
        raise ValueError("Only `.nc` and `.csv` files allowed!")

    with httpx.stream("GET", url, timeout=60, **kwargs) as response:
        response.raise_for_status()
        save_data_stream(response, filename)

    return None


def get_available_datasets() -> None:
    save_json_from_URL("https://dataset.api.hub.geosphere.at/v1/datasets", "Datasets")
    return None


def load_csv(filename: str) -> xr.Dataset:
    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    STATION_CONFIG = Path(__file__).parent / "config" / "Metadata_klima-v2-10min.json"
    config = load_json(STATION_CONFIG)
    stations = pd.json_normalize(config["stations"]).set_index("id")

    ds = df.set_index(["time", "station"]).to_xarray()

    # Attach per-station metadata as dataset attrs
    station_ids = ds["station"].values.tolist()
    ds.attrs["stations"] = {
        sid: stations.loc[sid].to_dict() for sid in station_ids if sid in stations.index
    }

    # Attach parameter metadata to each variable
    parameters = pd.json_normalize(config["parameters"]).set_index("name")
    for var in ds.data_vars:
        if var in parameters.index:
            ds[var].attrs.update(parameters.loc[var].dropna().to_dict())

    return ds
