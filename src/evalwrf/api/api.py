"""
evalwrf.api
===========

API interfaces and utilities for downloading meteorological data from multiple
sources used in WRF pre-processing workflows.

This module provides:

* Utility functions for loading and saving JSON, NetCDF, and CSV data via URLs.
* :class:`URL` — an :class:`httpx.URL` subclass that supports path-segment
  concatenation with the ``/`` operator.
* :class:`BaseAPI` — abstract base class for meteorological download APIs,
  shared by :class:`GFSAPI` and :class:`ERA5API`.
* :class:`GFSAPI` — downloads GFS (Global Forecast System) GRIB2 data from
  NOAA NOMADS or the NCAR/RDA archive.
* :class:`ERA5API` — downloads ERA5 reanalysis data via the Copernicus CDS API.
* :class:`ZAMGAPI` — queries and downloads data from the GeoSphere Austria
  dataset REST API.

Example
-------
Download station observations from GeoSphere Austria::

    api = ZAMGAPI("klima-v2-10min", type="station", mode="historical")
    api.download(
        filename="murau",
        params=dict(
            start="2025-01-03T00:00",
            end="2025-01-04T00:00",
            parameters=["TL", "RR"],
            station_ids=15920,
            output_format="csv",
        ),
        folder="data/",
    )
"""

from .api_classes import URL, GFSAPI, ERA5API, ZAMGAPI
from .geosphere_interface import load_url_for_resource, save_data, save_json_from_URL
