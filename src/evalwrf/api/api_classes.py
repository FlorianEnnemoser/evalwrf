import json
import time
from collections import namedtuple
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import ClassVar, List, Literal, Self

from .helpers import create_folder, load_json, save_data_stream

import httpx
import numpy as np
import pandas as pd


MetaData = namedtuple("MetaData", field_names=["stations", "parameters"])

try:
    import cdsapi
except ImportError as e:
    cdsapi = None  #: ``None`` when ``cdsapi`` is not installed.
    print(e, "\nInstall `cdsapi` in order to use ERA5 dataset!")


class URL(httpx.URL):
    """An :class:`httpx.URL` subclass that supports path-segment concatenation.

    Inherits all behaviour from :class:`httpx.URL` and adds the ``/`` operator
    for conveniently appending one or more path segments without needing to
    manually manage trailing slashes.

    Examples
    --------
    >>> base = URL("https://example.com/api/v1")
    >>> base / "datasets"
    URL('https://example.com/api/v1/datasets')
    >>> base / ["datasets", "klima", "metadata"]
    URL('https://example.com/api/v1/datasets/klima/metadata')
    """

    def __truediv__(self, other: str | List[str]) -> "URL":
        """Append one or more path segments to this URL.

        Parameters
        ----------
        other : str or list of str
            A single path segment or a list of path segments to append.

        Returns
        -------
        URL
            A new :class:`URL` with the segments joined by ``/`` and appended
            to the current URL string.

        Examples
        --------
        >>> URL("https://api.example.com") / "v1" / "data"
        URL('https://api.example.com/v1/data')
        >>> URL("https://api.example.com") / ["v1", "data", "metadata"]
        URL('https://api.example.com/v1/data/metadata')
        """
        if isinstance(other, str):
            other = [other]
        return __class__(self.__str__() + "/" + "/".join(other))


@dataclass
class BaseAPI:
    """Abstract base class for meteorological data download APIs.

    Provides shared utilities for HTTP file download, standardised GRIB2
    filename generation, and date-range parsing.  Concrete subclasses
    (:class:`GFSAPI`, :class:`ERA5API`) extend this class with
    dataset-specific logic.

    Parameters
    ----------
    daterange : str
        Date range in ``"YYYY-MM-DD|YYYY-MM-DD"`` format.  The ``|``
        character separates the start and end dates (both inclusive).
    grid_size : {"1p00", "0p50", "0p25"}
        Spatial grid resolution.  ``"1p00"`` → 1°×1°,
        ``"0p50"`` → 0.5°×0.5°, ``"0p25"`` → 0.25°×0.25°.
    savefolder : str, optional
        Directory where downloaded files are written.  Created automatically
        if it does not exist.  Defaults to ``"download_wrf"``.
    base_url : str, optional
        Base URL prepended to relative download paths.  Defaults to ``""``.
    """

    daterange: str
    """Date range string in ``"YYYY-MM-DD|YYYY-MM-DD"`` format."""

    grid_size: Literal["1p00", "0p50", "0p25"]
    """Spatial resolution identifier."""

    savefolder: str = field(default="download_wrf", repr=True)
    """Local directory for downloaded files."""

    base_url: str = field(default="", repr=True)
    """Base URL for constructing download requests."""

    def _download(self, url: str, file: str, max_sleep: int = 2) -> None:
        """Download a single file from *url* and save it under :attr:`savefolder`.

        After a successful download the method sleeps for a random duration between
        1 s and *max_sleep* s to avoid overloading the remote server.

        Parameters
        ----------
        url : str
            Relative URL path (appended to :attr:`base_url`).
        file : str
            Target filename, written inside :attr:`savefolder`.
        max_sleep : int, optional
            Upper bound (seconds) for the random post-download sleep interval.
            Defaults to ``2`` seconds.

        Raises
        ------
        ValueError
            If the HTTP response status code is not 200.
        """
        download_folder = create_folder(self.savefolder)

        response = httpx.get(self.base_url + url)

        if response.status_code != 200:
            raise ValueError(
                f"Status code {response.status_code}: cannot download file.\n"
                f"Requested URL: {self.base_url + url}"
            )

        with (download_folder / file).open("wb") as f:
            f.write(response.content)

        time_to_sleep = float(np.random.uniform(1, max_sleep))
        time.sleep(time_to_sleep)
        return None

    def _filename(self, prefix: str, *args) -> str:
        """Generate a GRIB2 filename from a prefix and positional components.

        Parameters
        ----------
        prefix : str
            Filename prefix (e.g. ``"GFS"``).
        *args
            Additional components stringified and joined with ``_``.

        Returns
        -------
        str
            Filename of the form ``"<prefix>_<arg1>_<arg2>_....grib2"``.

        Examples
        --------
        >>> api._filename("GFS", "20240101", "00", 70, 10, 0, 360)
        'GFS_20240101_00_70_10_0_360.grib2'
        """
        return f"{prefix}_" + "_".join([str(a) for a in args]) + ".grib2"

    @property
    def today(self) -> pd.Timestamp:
        """Current date and time.

        Returns
        -------
        pandas.Timestamp
            The current local timestamp via :func:`pandas.Timestamp.now`.
        """
        return pd.Timestamp.now()

    @property
    def date_range(self) -> pd.DatetimeIndex:
        """Parse :attr:`daterange` into a daily :class:`pandas.DatetimeIndex`.

        The :attr:`daterange` attribute must use ``"|"`` as the separator
        between start and end dates.

        Returns
        -------
        pandas.DatetimeIndex
            Daily date index spanning the requested period (inclusive).

        Examples
        --------
        >>> api = GFSAPI(daterange="2020-01-01|2020-01-03", grid_size="1p00")
        >>> api.date_range
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]', freq='D')
        """
        splitted_range = self.daterange.split("|", maxsplit=1)
        return pd.date_range(*splitted_range, freq="D")


@dataclass
class GFSAPI(BaseAPI):
    """API client for downloading GFS (Global Forecast System) GRIB2 data.

    Automatically selects the download source depending on the requested dates:

    * **Recent / forecast data** (within the last 9 days or in the future):
      NOAA NOMADS server at ``https://nomads.ncep.noaa.gov``.
    * **Historical data** (older than 9 days):
      NCAR/RDA archive at ``https://data.rda.ucar.edu/d083002/grib2/``.

    Currently only 1°×1° grid resolution (``"1p00"``) is supported.

    Parameters
    ----------
    daterange : str
        Date range in ``"YYYY-MM-DD|YYYY-MM-DD"`` format.
    grid_size : {"1p00", "0p50", "0p25"}
        Grid resolution.  Only ``"1p00"`` is implemented.
    savefolder : str, optional
        Download directory.  Defaults to ``"download_wrf"``.
    base_url : str, optional
        Override the automatically selected base URL.  Defaults to ``""``.

    Attributes
    ----------
    noaa_start_date : pandas.Timestamp
        Earliest date available on NOAA NOMADS (today - 9 days).
    is_forecast : bool
        ``True`` when any requested date lies in the future.
    valid_current : bool
        ``True`` when all requested dates fall inside the NOAA availability window.
    date_format : str
        strftime format used in GRIB2 filenames: ``"%Y%m%d"``.
    forecast_times : list of str
        Six-hourly synoptic times: ``["00", "06", "12", "18"]``.
    file_date : list or tuple
        Processed date strings for URL construction.  A tuple when both past
        and forecast dates coexist.
    file_urls : list of str
        Accumulated relative download URLs (populated by factory methods).
    filenames : list of str
        Local filenames corresponding to :attr:`file_urls`.
    set_noaa : bool
        ``True`` when NOAA NOMADS is the selected download source.

    Raises
    ------
    NotImplementedError
        If *grid_size* is not ``"1p00"``.
    ValueError
        If more than 20 dates are requested.

    Examples
    --------
    >>> gfs = GFSAPI(daterange="2020-01-01|2020-01-03", grid_size="1p00")
    >>> gfs.download(bottom=40, top=55, left=5, right=25, as_test=True)
    """

    def __post_init__(self) -> None:
        """Initialise GFSAPI attributes and select the appropriate download source."""
        self.noaa_start_date: pd.Timestamp = self.today - pd.Timedelta(days=9)
        self.is_forecast: bool = bool((self.today < self.date_range).any())
        self.valid_current: bool = bool((self.date_range > self.noaa_start_date).all())
        self.date_format: str = "%Y%m%d"
        self.forecast_times: list[str] = ["00", "06", "12", "18"]
        self.file_date = self._get_date_range()
        self.file_urls: list[str] = []
        self.filenames: list[str] = []

        if self.valid_current or self.is_forecast:
            self.base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl"
            self.set_noaa: bool = True
        else:
            self.base_url = "https://data.rda.ucar.edu/d083002/grib2/"
            self.set_noaa = False

        print(f"Using data from: {self.base_url}")

        if self.grid_size != "1p00":
            raise NotImplementedError(
                "Only 1°×1° grid size ('1p00') is currently implemented."
            )

        if len(self.file_date) > 20:
            raise ValueError("Maximum number of requested dates is 20.")

    def _get_date_range(self) -> list | tuple:
        """Split the date range into past and forecast sub-ranges.

        Returns
        -------
        list or tuple
            A list of past date strings if no forecast dates exist; otherwise
            a 2-tuple ``(past_dates, forecast_dates)`` where *forecast_dates*
            is a :class:`pandas.DatetimeIndex`.
        """
        condition_past = self.date_range < (self.today - pd.Timedelta(days=1))
        past_times = self.date_range[condition_past].strftime(self.date_format)
        forecast_times = self.date_range[~condition_past]

        if forecast_times.empty:
            return past_times
        return past_times, forecast_times

    def _factory_noaa(
        self,
        bottom: int = 10,
        top: int = 70,
        left: int = 0,
        right: int = 360,
    ) -> None:
        """Populate URL lists for NOAA NOMADS downloads.

        Dispatches to :meth:`_factory_forecast` when forecast data is requested,
        or :meth:`_factory_non_forecast` for purely historical requests.

        Parameters
        ----------
        bottom : int, optional
            Southern latitude boundary in degrees North.  Defaults to ``10``.
        top : int, optional
            Northern latitude boundary in degrees North.  Defaults to ``70``.
        left : int, optional
            Western longitude boundary in degrees East (0-360).  Defaults to ``0``.
        right : int, optional
            Eastern longitude boundary in degrees East (0-360).  Defaults to ``360``.

        Raises
        ------
        ValueError
            If *left* or *right* are negative (only 0-360 range is supported).
        """
        if any(np.sign([left, right]) == -1):
            raise ValueError(
                "Only positive longitude values (0°-360°E) are supported. "
                "For example, use 280 instead of -80 for 80°W."
            )

        if self.is_forecast:
            self._factory_forecast(bottom, top, left, right)
        else:
            self._factory_non_forecast(self.file_date, bottom, top, left, right)

    def _factory_forecast(
        self,
        bottom: int = 10,
        top: int = 70,
        left: int = 0,
        right: int = 360,
    ) -> None:
        """Build URLs for a mixed past-and-forecast download.

        Handles historical dates via :meth:`_factory_non_forecast` then appends
        6-hourly forecast steps for each future day starting at 00 UTC.

        Parameters
        ----------
        bottom : int, optional
            Southern latitude boundary.  Defaults to ``10``.
        top : int, optional
            Northern latitude boundary.  Defaults to ``70``.
        left : int, optional
            Western longitude boundary (0-360).  Defaults to ``0``.
        right : int, optional
            Eastern longitude boundary (0-360).  Defaults to ``360``.
        """
        past_dates, forecast_dates = self.file_date

        n_forecast_days = len(forecast_dates)
        six_hour_interval = np.arange(0, n_forecast_days * 24, 6)

        self._factory_non_forecast(past_dates, bottom, top, left, right)

        for future_hour in six_hour_interval:
            single_url = (
                f"?dir=%2Fgfs.{forecast_dates[0].strftime(self.date_format)}%2F00%2Fatmos"
                f"&file=gfs.t00z.pgrb2.1p00.f{future_hour:03}&all_var=on&all_lev=on"
                f"&subregion=&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
            )
            date = forecast_dates[0] + pd.Timedelta(hours=int(future_hour))
            hour = date.hour

            self.file_urls.append(single_url)
            self.filenames.append(
                self._filename(
                    "GFS",
                    date.strftime(self.date_format),
                    f"{hour:02}",
                    top,
                    bottom,
                    left,
                    right,
                )
            )

    def _factory_non_forecast(
        self,
        dates: list,
        bottom: int = 10,
        top: int = 70,
        left: int = 0,
        right: int = 360,
    ) -> None:
        """Build URLs for historical GFS analysis data (non-forecast).

        Iterates over every combination of *dates* and the four synoptic
        times in :attr:`forecast_times` (00, 06, 12, 18 UTC).

        Parameters
        ----------
        dates : list of str
            Date strings formatted as ``"%Y%m%d"``.
        bottom : int, optional
            Southern latitude boundary.  Defaults to ``10``.
        top : int, optional
            Northern latitude boundary.  Defaults to ``70``.
        left : int, optional
            Western longitude boundary (0-360).  Defaults to ``0``.
        right : int, optional
            Eastern longitude boundary (0-360).  Defaults to ``360``.
        """
        for day, hour in product(dates, self.forecast_times):
            single_url = (
                f"?dir=%2Fgfs.{day}%2F{hour}%2Fatmos"
                f"&file=gfs.t{hour}z.pgrb2.1p00.f000&all_var=on&all_lev=on"
                f"&subregion=&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
            )
            self.file_urls.append(single_url)
            self.filenames.append(
                self._filename("GFS", day, hour, top, bottom, left, right)
            )

    def _factory_ncar(self) -> None:
        """Build URLs for NCAR/RDA archive downloads.

        Populates :attr:`file_urls` and :attr:`filenames` with paths for
        historical GFS data from the NCAR Research Data Archive at
        ``https://data.rda.ucar.edu/d083002/grib2/``.
        """
        file_year = self.date_range.year
        file_yearmonth = self.date_range.strftime("%Y.%m")

        for (day, year, ym), hour in product(
            zip(self.file_date, file_year, file_yearmonth),
            self.forecast_times,
        ):
            single_url = f"{year}/{ym}/fnl_{day}_{hour}_00.grib2"
            self.file_urls.append(single_url)
            self.filenames.append(self._filename("GFS", day, hour))

    def download(
        self,
        bottom: int = 10,
        top: int = 70,
        left: int = 0,
        right: int = 360,
        as_test: bool = True,
    ) -> Self:
        """Build the download queue and optionally execute all downloads.

        Parameters
        ----------
        bottom : int, optional
            Southern latitude boundary in degrees North.  Defaults to ``10``.
        top : int, optional
            Northern latitude boundary in degrees North.  Defaults to ``70``.
        left : int, optional
            Western longitude boundary in degrees East (0-360).  Defaults to ``0``.
        right : int, optional
            Eastern longitude boundary in degrees East (0-360).  Defaults to ``360``.
        as_test : bool, optional
            When ``True`` (default), only print filenames without performing
            actual downloads.  Set to ``False`` to execute the downloads.

        Returns
        -------
        GFSAPI
            Returns ``self`` to allow method chaining.

        Examples
        --------
        >>> gfs = GFSAPI(daterange="2020-01-01|2020-01-02", grid_size="1p00")
        >>> gfs.download(bottom=40, top=55, left=5, right=25, as_test=True)
        """
        if self.set_noaa:
            self._factory_noaa(bottom, top, left, right)
        else:
            self._factory_ncar()

        print(f"Downloading to: {self.savefolder}")
        for url, file in zip(self.file_urls, self.filenames):
            print(file)
            if not as_test:
                self._download(url, file)
        return self


@dataclass
class ERA5API(BaseAPI):
    """API client for downloading ERA5 reanalysis data via the Copernicus CDS API.

    Requires the optional ``cdsapi`` library and a valid CDS API key stored in
    ``~/.cdsapirc``.  See the `CDS How-to API guide
    <https://cds.climate.copernicus.eu/how-to-api>`_ for setup instructions.

    Parameters
    ----------
    daterange : str
        Date range in ``"YYYY-MM-DD|YYYY-MM-DD"`` format.
    grid_size : {"1p00", "0p50", "0p25"}
        Spatial grid resolution.
    savefolder : str, optional
        Download directory.  Defaults to ``"download_wrf"``.
    base_url : str, optional
        Unused for ERA5; present for :class:`BaseAPI` compatibility.

    Attributes
    ----------
    pressure_dataset : str
        CDS dataset identifier for pressure-level fields:
        ``"reanalysis-era5-pressure-levels"``.
    surface_dataset : str
        CDS dataset identifier for single-level surface fields:
        ``"reanalysis-era5-single-levels"``.

    Notes
    -----
    ``cdsapi`` is a default dependency of *evalwrf*. If problems occure install it
    separately::

        pip install cdsapi

    The CDS API key must be placed in ``~/.cdsapirc``.  For an overview of
    available variables see https://www.youtube.com/watch?v=M91ec7EdCic.

    Examples
    --------
    >>> era5 = ERA5API(daterange="2022-03-13|2022-03-15", grid_size="0p25")
    >>> era5.download(bottom=20, top=45, left=35, right=70, as_test=True)
    """

    def __post_init__(self) -> None:
        """Initialise default pressure-level and surface CDS request templates."""
        if cdsapi is None:
            raise ImportError(
                "cdsapi is required for ERA5API. Install it with: pip install cdsapi"
            )

        self.pressure_dataset: str = "reanalysis-era5-pressure-levels"
        self.surface_dataset: str = "reanalysis-era5-single-levels"

        self._pressure: dict = {
            "product_type": ["reanalysis"],
            "variable": [
                "geopotential",
                "relative_humidity",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "time": [
                "00:00",
                "03:00",
                "06:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ],
            "pressure_level": [
                "1",
                "2",
                "3",
                "5",
                "7",
                "10",
                "20",
                "30",
                "50",
                "70",
                "100",
                "125",
                "150",
                "175",
                "200",
                "225",
                "250",
                "300",
                "350",
                "400",
                "450",
                "500",
                "550",
                "600",
                "650",
                "700",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
                "1000",
            ],
            "data_format": "grib",
            "download_format": "unarchived",
        }

        self._surface: dict = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "sea_surface_temperature",
                "surface_pressure",
                "total_precipitation",
                "skin_temperature",
                "surface_latent_heat_flux",
                "top_net_solar_radiation_clear_sky",
                "snow_depth",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "soil_type",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "leaf_area_index_high_vegetation",
                "geopotential",
                "land_sea_mask",
                "sea_ice_cover",
            ],
            "time": [
                "00:00",
                "03:00",
                "06:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ],
            "data_format": "grib",
            "download_format": "unarchived",
        }

    @property
    def pressure(self) -> dict:
        """CDS request template for ERA5 pressure-level data.

        Returns
        -------
        dict
            Current request dictionary for ``reanalysis-era5-pressure-levels``.
            Keys ``"year"``, ``"month"``, ``"day"``, ``"area"``, and ``"grid"``
            are populated by the ``_update_*`` helper methods before download.
        """
        return self._pressure

    @property
    def surface(self) -> dict:
        """CDS request template for ERA5 single-level surface data.

        Returns
        -------
        dict
            Current request dictionary for ``reanalysis-era5-single-levels``.
            Keys ``"year"``, ``"month"``, ``"day"``, ``"area"``, and ``"grid"``
            are populated by the ``_update_*`` helper methods before download.
        """
        return self._surface

    def _update_area(
        self,
        bottom: int,
        top: int,
        left: int,
        right: int,
    ) -> Self:
        """Set the bounding-box area in both CDS request dictionaries.

        Parameters
        ----------
        bottom : int
            Southern latitude boundary in degrees North.
        top : int
            Northern latitude boundary in degrees North.
        left : int
            Western longitude boundary.
        right : int
            Eastern longitude boundary.

        Returns
        -------
        ERA5API
            Returns ``self`` for method chaining.

        Notes
        -----
        The CDS API ``"area"`` key expects the format ``[North, West, South, East]``.
        """
        area = [top, left, bottom, right]
        self._surface.update({"area": area})
        self._pressure.update({"area": area})
        return self

    def _update_time(
        self,
        year: np.ndarray,
        month: np.ndarray,
        day: np.ndarray,
    ) -> Self:
        """Set year, month, and day fields in both CDS request dictionaries.

        Parameters
        ----------
        year : numpy.ndarray
            Array of unique years to request.
        month : numpy.ndarray
            Array of unique months to request.
        day : numpy.ndarray
            Array of unique days to request.

        Returns
        -------
        ERA5API
            Returns ``self`` for method chaining.
        """
        self._surface.update({"year": year.astype(str).tolist()})
        self._surface.update({"month": month.astype(str).tolist()})
        self._surface.update({"day": day.astype(str).tolist()})

        self._pressure.update({"year": year.astype(str).tolist()})
        self._pressure.update({"month": month.astype(str).tolist()})
        self._pressure.update({"day": day.astype(str).tolist()})
        return self

    def _update_grid(self) -> Self:
        """Translate :attr:`~BaseAPI.grid_size` to the CDS ``"grid"`` format.

        Maps the :attr:`~BaseAPI.grid_size` attribute (e.g. ``"0p25"``) to
        the CDS slash notation (e.g. ``"0.25/0.25"``) and writes it into both
        request dictionaries.

        Returns
        -------
        ERA5API
            Returns ``self`` for method chaining.
        """
        grid_translator = {
            "1p00": "1.0/1.0",
            "0p50": "0.5/0.5",
            "0p25": "0.25/0.25",
        }
        used_grid = grid_translator.get(self.grid_size)
        self._surface.update({"grid": used_grid})
        self._pressure.update({"grid": used_grid})
        return self

    def download(
        self,
        bottom: int = 10,
        top: int = 70,
        left: int = 0,
        right: int = 360,
        as_test: bool = True,
        as_netcdf: bool = False,
    ) -> None:
        """Download ERA5 surface and pressure-level data via the CDS API.

        Prepares area, time, and grid settings, then retrieves both the
        single-level and pressure-level datasets sequentially.

        Parameters
        ----------
        bottom : int, optional
            Southern latitude boundary in degrees North.  Defaults to ``10``.
        top : int, optional
            Northern latitude boundary in degrees North.  Defaults to ``70``.
        left : int, optional
            Western longitude boundary.  Defaults to ``0``.
        right : int, optional
            Eastern longitude boundary.  Defaults to ``360``.
        as_test : bool, optional
            If ``True`` (default), only print target filenames without
            executing downloads.  Set to ``False`` to download.
        as_netcdf : bool, optional
            Request NetCDF output instead of GRIB when ``True``.
            Defaults to ``False``.

        Raises
        ------
        ImportError
            If ``cdsapi`` is not installed.

        Notes
        -----
        A valid ``~/.cdsapirc`` API key file is required for actual downloads.

        Examples
        --------
        >>> era5 = ERA5API(daterange="2022-03-13|2022-03-15", grid_size="0p25")
        >>> era5.download(bottom=20, top=45, left=35, right=70, as_test=True)
        """

        self._update_area(bottom, top, left, right)
        self._update_time(
            np.unique(self.date_range.year),
            np.unique(self.date_range.month),
            np.unique(self.date_range.day),
        )
        self._update_grid()

        client = cdsapi.Client()

        if as_netcdf:
            file_ending = "nc"
            self._surface["data_format"] = "netcdf"
            self._pressure["data_format"] = "netcdf"
        else:
            file_ending = self._surface["data_format"]

        for dataset, request in zip(
            [self.surface_dataset, self.pressure_dataset],
            [self.surface, self.pressure],
        ):
            filename = f"{dataset}.{file_ending}"
            print(filename)
            if not as_test:
                client.retrieve(dataset, request, filename)


@dataclass
class ZAMGAPI:
    """Query and download data from the GeoSphere Austria dataset REST API.

    Looks up the dataset endpoint from a local ``Datasets.json`` catalogue
    (bundled with *evalwrf* by default) and provides methods to download data,
    inspect metadata, and refresh the catalogue from the live API.

    The name ``ZAMGAPI`` is chosen intentionally - a ``GeosphereAPI`` object
    could be confused with the **actual** API class from GeoSphere Austria
    (see https://dataset.api.hub.geosphere.at/v1/docs/).

    Parameters
    ----------
    resource : str
        Dataset resource identifier, e.g. ``"klima-v2-10min"``.
    type : {"grid", "timeseries", "station"}
        Dataset type.
    mode : {"historical", "current", "forecast"}
        Temporal mode of the dataset.
    datasets_file : str, optional
        Path to a ``Datasets.json`` catalogue file.  Defaults to the file
        bundled with the package at ``config/Datasets.json``.  Pass a custom
        path after calling :meth:`update_api` to use a freshly downloaded
        catalogue.

    Attributes
    ----------
    dataset_url : str
        Full REST endpoint URL for the selected dataset (resolved from the
        catalogue during initialisation).
    response_formats : list of str
        Output formats supported by this dataset (e.g. ``["csv", "geojson"]``).

    Raises
    ------
    ValueError
        If no matching entry is found in the catalogue, or if multiple entries
        match the ``(resource, type, mode)`` combination.

    References
    ----------
    Getting-started guide:
        https://dataset.api.hub.geosphere.at/v1/docs/getting-started.html
    Resource reference:
        https://dataset.api.hub.geosphere.at/v1/docs/user-guide/resource.html#resources

    Examples
    --------
    Download station observations as CSV::

        api = ZAMGAPI("klima-v2-10min", type="station", mode="historical")
        api.download(
            filename="murau",
            params=dict(
                start="2025-01-03T00:00",
                end="2026-01-04T12:00",
                parameters=["TL", "RR"],
                station_ids=15920,
                output_format="csv",
            ),
            folder="data/",
        )
        # saves → data/murau.csv

    Download a grid forecast as NetCDF::

        api = ZAMGAPI("nwp-v1-1h-2500m", type="grid", mode="forecast")
        api.download(
            filename="forecast_austria",
            params=dict(
                start="2025-01-03T00:00",
                end="2025-01-04T12:00",
                parameters=["t2m", "tcc", "rr_acc"],
                bbox="46.0,9.0,49.0,18.0",
                output_format="netcdf",
            ),
        )
        # saves → ./forecast_austria.nc
    """

    resource: str
    """Dataset resource identifier (e.g. ``"klima-v2-10min"``)."""

    type: Literal["grid", "timeseries", "station"]
    """Dataset type: ``"grid"``, ``"timeseries"``, or ``"station"``."""

    mode: Literal["historical", "current", "forecast"]
    """Temporal mode: ``"historical"``, ``"current"``, or ``"forecast"``."""

    datasets_file: str = field(default="")
    """Path to the ``Datasets.json`` catalogue.  Empty string → bundled default."""

    #: Map from ``output_format`` values to file extensions.
    _FORMAT_EXT: ClassVar[dict[str, str]] = {
        "netcdf": "nc",
        "csv": "csv",
        "geojson": "geojson",
    }

    #: GeoSphere Austria datasets catalogue endpoint.
    _DATASETS_ENDPOINT: ClassVar[str] = (
        "https://dataset.api.hub.geosphere.at/v1/datasets"
    )

    def __post_init__(self) -> None:
        """Resolve the dataset URL from the catalogue on construction.

        If :attr:`datasets_file` is empty, the bundled catalogue at
        ``src/evalwrf/config/Datasets.json`` is used.  The catalogue is then
        searched for an entry whose key contains
        ``"/<type>/<mode>/<resource>"``.

        Raises
        ------
        ValueError
            If the ``(resource, type, mode)`` combination is not found in the
            catalogue, or matches more than one entry.
        """
        if not self.datasets_file:
            self.datasets_file = str(Path(__file__).parent / "config" / "Datasets.json")

        data: dict = load_json(self.datasets_file)

        key_pattern = f"/{self.type}/{self.mode}/{self.resource}"
        matches = {k: v for k, v in data.items() if key_pattern in k}

        if not matches:
            raise ValueError(
                f"No dataset found for resource='{self.resource}', "
                f"type='{self.type}', mode='{self.mode}' "
                f"in '{self.datasets_file}'.\n"
                "Call update_api() to refresh the catalogue and try again."
            )
        if len(matches) > 1:
            keys = "\n  ".join(matches.keys())
            raise ValueError(
                f"Multiple datasets matched '{key_pattern}':\n  {keys}\n"
                "Provide a more specific resource name."
            )

        entry = next(iter(matches.values()))
        self.dataset_url: str = entry["url"]
        self.response_formats: list[str] = entry.get("response_formats", [])

    def download(
        self,
        filename: str,
        params: dict,
        folder: str = ".",
    ) -> Path:
        """Download data from the dataset endpoint to a local file.

        The output file format (and therefore the file extension) is derived
        automatically from ``params["output_format"]``. Any extension already
        present in *filename* is stripped and replaced.

        Parameters
        ----------
        filename : str
            Base name for the output file, without extension
            (e.g. ``"murau"`` or ``"murau.csv"`` - the extension is
            always overridden).
        params : dict
            Query parameters forwarded directly to the API.  Common keys:

            * ``"start"`` / ``"end"`` - ISO 8601 datetime strings.
            * ``"parameters"`` - list of parameter codes (e.g. ``["TL", "RR"]``).
            * ``"station_ids"`` - single integer or list of station IDs.
            * ``"bbox"`` - bounding box string ``"lat_min,lon_min,lat_max,lon_max"``.
            * ``"output_format"`` - ``"csv"`` (default), ``"netcdf"``, or
              ``"geojson"``.
        folder : str, optional
            Directory where the file is written.  Created automatically if it
            does not exist.  Defaults to the current working directory.

        Returns
        -------
        pathlib.Path
            Absolute path of the saved file.

        Raises
        ------
        httpx.HTTPStatusError
            If the server returns a non-2xx HTTP status code.

        Examples
        --------
        Download station data as CSV:

        >>> api = ZAMGAPI("klima-v2-10min", type="station", mode="historical")
        >>> path = api.download(
        ...     filename="murau",
        ...     params=dict(
        ...         start="2025-01-03T00:00",
        ...         end="2025-01-04T00:00",
        ...         parameters=["TL", "RR"],
        ...         station_ids=15920,
        ...         output_format="csv",
        ...     ),
        ...     folder="data/",
        ... )
        >>> print(path)
        data/murau.csv

        Download grid forecast as NetCDF:

        >>> api = ZAMGAPI("nwp-v1-1h-2500m", type="grid", mode="forecast")
        >>> api.download(
        ...     filename="forecast",
        ...     params=dict(
        ...         start="2025-01-03T00:00",
        ...         end="2025-01-04T12:00",
        ...         parameters=["t2m"],
        ...         bbox="46.0,9.0,49.0,18.0",
        ...         output_format="netcdf",
        ...     ),
        ... )
        """
        output_format: str = params.get("output_format", "csv")
        ext: str = self._FORMAT_EXT.get(output_format, output_format)

        stem = Path(filename).stem
        dest: Path = create_folder(folder) / f"{stem}.{ext}"

        with httpx.stream(
            "GET", self.dataset_url, timeout=60, params=params
        ) as response:
            response.raise_for_status()
            save_data_stream(response, str(dest))

        return dest

    def update_api(self, folder: str = ".") -> None:
        """Download a fresh ``Datasets.json`` catalogue from the GeoSphere API.

        Saves the catalogue to *folder* and updates :attr:`datasets_file` so
        that subsequent calls on this instance use the refreshed data.

        Parameters
        ----------
        folder : str, optional
            Directory where ``Datasets.json`` is written.  Created
            automatically if it does not exist.  Defaults to the current
            working directory.

        Raises
        ------
        ValueError
            If the API returns a non-200 status code.

        Examples
        --------
        >>> api = ZAMGAPI("klima-v2-10min", type="station", mode="historical")
        >>> api.update_api(folder="config/")
        # downloads → config/Datasets.json
        # api.datasets_file is updated automatically
        """
        out_dir = create_folder(folder)
        out_path = out_dir / "Datasets.json"

        response = httpx.get(self._DATASETS_ENDPOINT)
        if response.status_code != 200:
            raise ValueError(
                f"Could not fetch datasets catalogue: "
                f"HTTP {response.status_code} from {self._DATASETS_ENDPOINT}"
            )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=4)

        self.datasets_file = str(out_path)
        print(f"Saved catalogue → {out_path}")

    @property
    def metadata(self) -> MetaData:
        """Fetch dataset metadata from the API and return as a :class:`MetaData`.

        Requests the ``<dataset_url>/metadata`` endpoint and parses the JSON
        response into two :class:`pandas.DataFrame` objects - one for stations
        and one for parameters.

        Returns
        -------
        MetaData
            Named tuple with fields:

            * ``stations`` - :class:`pandas.DataFrame` of available stations.
            * ``parameters`` - :class:`pandas.DataFrame` of available
              parameters (all-NaN rows and columns are dropped).

        Raises
        ------
        httpx.HTTPStatusError
            If the metadata endpoint returns a non-2xx status code.

        Examples
        --------
        >>> api = ZAMGAPI("klima-v2-10min", type="station", mode="historical")
        >>> meta = api.metadata
        >>> meta.stations.head()
        >>> meta.parameters.head()
        """
        response = httpx.get(f"{self.dataset_url}/metadata")
        response.raise_for_status()
        meta: dict = response.json()

        df_stations = pd.json_normalize(meta.get("stations", []))
        df_parameters = (
            pd.json_normalize(meta.get("parameters", []))
            .dropna(how="all", axis="index")
            .dropna(how="all", axis="columns")
        )
        return MetaData(stations=df_stations, parameters=df_parameters)
