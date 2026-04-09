"""
tests_api.py
============

Unit tests for :mod:`evalwrf.api`.

Run with::

    pytest tests/tests_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalwrf.api import (
    URL,
    BaseAPI,
    ERA5API,
    GFSAPI,
    ZAMGAPI,
    load_json,
    load_metadata,
    load_url_from_resource,
    save_json_from_URL,
    save_netcdf,
    save_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAST_DATERANGE = "2020-01-01|2020-01-03"
"""A historical date range that will always fall outside the NOAA 9-day window."""


@pytest.fixture()
def past_gfs() -> GFSAPI:
    """Return a :class:`GFSAPI` instance for a historical date range."""
    return GFSAPI(daterange=PAST_DATERANGE, grid_size="1p00")


@pytest.fixture()
def era5() -> ERA5API:
    """Return an :class:`ERA5API` instance."""
    return ERA5API(daterange="2022-03-13|2022-03-15", grid_size="0p25")


@pytest.fixture()
def zamg() -> ZAMGAPI:
    """Return a :class:`ZAMGAPI` instance."""
    return ZAMGAPI(type="station", mode="historical", resource="klima-v2-10min")


@pytest.fixture()
def sample_json(tmp_path: Path) -> Path:
    """Write a small resource-map JSON and return its path."""
    data = {
        "klima-v2-10min": {
            "url": "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-10min"
        },
        "other-resource": {"url": "https://example.com/other"},
    }
    p = tmp_path / "Datasets.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def metadata_json(tmp_path: Path) -> Path:
    """Write a minimal metadata JSON file and return its path."""
    data = {
        "parameters": [
            {"name": "TL", "description": "Air temperature"},
            {"name": "RR", "description": "Precipitation"},
        ],
        "stations": [
            {"id": 15920, "name": "Murau"},
            {"id": 11035, "name": "Graz"},
        ],
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------


class TestURL:
    def test_truediv_single_str(self) -> None:
        base = URL("https://example.com/api")
        result = base / "v1"
        assert str(result) == "https://example.com/api/v1"

    def test_truediv_list_of_str(self) -> None:
        base = URL("https://example.com/api")
        result = base / ["v1", "data", "metadata"]
        assert str(result) == "https://example.com/api/v1/data/metadata"

    def test_truediv_chained(self) -> None:
        base = URL("https://example.com")
        result = base / "a" / "b" / "c"
        assert str(result) == "https://example.com/a/b/c"

    def test_truediv_returns_url_type(self) -> None:
        base = URL("https://example.com")
        result = base / "segment"
        assert isinstance(result, URL)

    def test_empty_segment(self) -> None:
        base = URL("https://example.com/api")
        result = base / ""
        assert str(result) == "https://example.com/api/"


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = load_json(str(p))
        assert result == data

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_json(str(p))


# ---------------------------------------------------------------------------
# load_url_from_resource
# ---------------------------------------------------------------------------


class TestLoadUrlFromResource:
    def test_finds_matching_resource(self, sample_json: Path) -> None:
        url = load_url_from_resource(str(sample_json), "klima-v2-10min")
        assert url is not None
        assert "klima-v2-10min" in str(url)

    def test_returns_none_for_missing_resource(self, sample_json: Path) -> None:
        result = load_url_from_resource(str(sample_json), "nonexistent-resource")
        assert result is None

    def test_returns_url_instance(self, sample_json: Path) -> None:
        url = load_url_from_resource(str(sample_json), "klima-v2-10min")
        assert isinstance(url, URL)

    def test_partial_match(self, sample_json: Path) -> None:
        """Resource key substring should be sufficient for a match."""
        url = load_url_from_resource(str(sample_json), "klima")
        assert url is not None


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_load_parameters(self, metadata_json: Path) -> None:
        df = load_metadata(str(metadata_json), "parameters")
        assert df.index.name == "name"
        assert "TL" in df.index
        assert "RR" in df.index

    def test_load_stations(self, metadata_json: Path) -> None:
        df = load_metadata(str(metadata_json), "stations")
        assert df.index.name == "id"
        assert 15920 in df.index

    def test_returns_dataframe(self, metadata_json: Path) -> None:
        df = load_metadata(str(metadata_json), "parameters")
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# save_json_from_URL
# ---------------------------------------------------------------------------


class TestSaveJsonFromURL:
    def test_saves_json_on_success(self, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}

        with patch("evalwrf.api.httpx.get", return_value=mock_response):
            save_json_from_URL(URL("https://example.com"), str(tmp_path / "out"))

        output = tmp_path / "out.json"
        assert output.exists()
        assert json.loads(output.read_text()) == {"result": "ok"}

    def test_raises_on_non_200(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("evalwrf.api.httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="404"):
                save_json_from_URL(URL("https://example.com"), "out")


# ---------------------------------------------------------------------------
# save_netcdf
# ---------------------------------------------------------------------------


class TestSaveNetcdf:
    def test_writes_binary_content(self, tmp_path: Path) -> None:
        content = b"NETCDF_BINARY_DATA"

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([content])

        output = tmp_path / "output.nc"
        with patch("evalwrf.api.httpx.stream", return_value=mock_response):
            save_netcdf(URL("https://example.com/data.nc"), str(output))

        assert output.read_bytes() == content


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------


class TestSaveCsv:
    def test_writes_csv_content(self, tmp_path: Path) -> None:
        content = b"col1,col2\n1,2\n"

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([content])

        output = tmp_path / "output.csv"
        with patch("evalwrf.api.httpx.stream", return_value=mock_response):
            save_csv(URL("https://example.com/data.csv"), str(output))

        assert output.read_bytes() == content


# ---------------------------------------------------------------------------
# BaseAPI
# ---------------------------------------------------------------------------


class TestBaseAPI:
    """Tests for the shared BaseAPI behaviour via the GFSAPI subclass."""

    def test_today_returns_timestamp(self, past_gfs: GFSAPI) -> None:
        assert isinstance(past_gfs.today, pd.Timestamp)

    def test_date_range_parsed_correctly(self, past_gfs: GFSAPI) -> None:
        dr = past_gfs.date_range
        assert isinstance(dr, pd.DatetimeIndex)
        assert len(dr) == 3
        assert str(dr[0].date()) == "2020-01-01"
        assert str(dr[-1].date()) == "2020-01-03"

    def test_filename_construction(self, past_gfs: GFSAPI) -> None:
        name = past_gfs._filename("GFS", "20200101", "00", 70, 10, 0, 360)
        assert name == "GFS_20200101_00_70_10_0_360.grib2"

    def test_filename_no_extra_args(self, past_gfs: GFSAPI) -> None:
        name = past_gfs._filename("PREFIX")
        assert name == "PREFIX_.grib2"

    def test_download_creates_directory(self, past_gfs: GFSAPI, tmp_path: Path) -> None:
        past_gfs.savefolder = str(tmp_path / "new_subdir")
        past_gfs.base_url = ""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"data"

        with patch("evalwrf.api.httpx.get", return_value=mock_response), \
             patch("evalwrf.api.t.sleep"):
            past_gfs._download("/some/path", "file.grib2")

        assert (tmp_path / "new_subdir").is_dir()
        assert (tmp_path / "new_subdir" / "file.grib2").exists()

    def test_download_raises_on_non_200(self, past_gfs: GFSAPI, tmp_path: Path) -> None:
        past_gfs.savefolder = str(tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("evalwrf.api.httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="503"):
                past_gfs._download("/bad/url", "file.grib2")


# ---------------------------------------------------------------------------
# GFSAPI
# ---------------------------------------------------------------------------


class TestGFSAPI:
    def test_historical_uses_ncar(self) -> None:
        gfs = GFSAPI(daterange=PAST_DATERANGE, grid_size="1p00")
        assert not gfs.set_noaa
        assert "rda.ucar.edu" in gfs.base_url

    def test_unsupported_grid_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            GFSAPI(daterange=PAST_DATERANGE, grid_size="0p25")

    def test_too_many_dates_raises(self) -> None:
        with pytest.raises(ValueError, match="20"):
            GFSAPI(daterange="2000-01-01|2000-02-28", grid_size="1p00")

    def test_factory_non_forecast_url_count(self, past_gfs: GFSAPI) -> None:
        """3 days × 4 synoptic times = 12 files."""
        past_gfs._factory_non_forecast(
            past_gfs.date_range.strftime(past_gfs.date_format),
            bottom=40, top=55, left=5, right=25,
        )
        assert len(past_gfs.file_urls) == 12
        assert len(past_gfs.filenames) == 12

    def test_factory_non_forecast_filename_format(self, past_gfs: GFSAPI) -> None:
        past_gfs._factory_non_forecast(
            past_gfs.date_range.strftime(past_gfs.date_format),
        )
        assert all(f.endswith(".grib2") for f in past_gfs.filenames)
        assert all(f.startswith("GFS_") for f in past_gfs.filenames)

    def test_factory_noaa_negative_longitude_raises(self, past_gfs: GFSAPI) -> None:
        past_gfs.is_forecast = False
        # past_gfs.file_date needs to be a list for _factory_non_forecast
        past_gfs.file_date = past_gfs.date_range.strftime(past_gfs.date_format)
        with pytest.raises(ValueError, match="positive"):
            past_gfs._factory_noaa(left=-10, right=30)

    def test_download_dry_run_returns_self(self, past_gfs: GFSAPI) -> None:
        result = past_gfs.download(as_test=True)
        assert result is past_gfs

    def test_download_dry_run_populates_filenames(self, past_gfs: GFSAPI) -> None:
        past_gfs.download(as_test=True)
        assert len(past_gfs.filenames) > 0

    def test_factory_ncar_url_structure(self, past_gfs: GFSAPI) -> None:
        past_gfs._factory_ncar()
        assert all("fnl_" in u for u in past_gfs.file_urls)


# ---------------------------------------------------------------------------
# ERA5API
# ---------------------------------------------------------------------------


class TestERA5API:
    def test_dataset_names(self, era5: ERA5API) -> None:
        assert era5.pressure_dataset == "reanalysis-era5-pressure-levels"
        assert era5.surface_dataset == "reanalysis-era5-single-levels"

    def test_pressure_property_returns_dict(self, era5: ERA5API) -> None:
        assert isinstance(era5.pressure, dict)
        assert "variable" in era5.pressure

    def test_surface_property_returns_dict(self, era5: ERA5API) -> None:
        assert isinstance(era5.surface, dict)
        assert "variable" in era5.surface

    def test_update_area(self, era5: ERA5API) -> None:
        era5._update_area(bottom=20, top=45, left=35, right=70)
        assert era5.surface["area"] == [45, 35, 20, 70]
        assert era5.pressure["area"] == [45, 35, 20, 70]

    def test_update_time(self, era5: ERA5API) -> None:
        era5._update_time(
            np.array([2022]),
            np.array([3]),
            np.array([13, 14, 15]),
        )
        assert era5.surface["year"] == ["2022"]
        assert era5.surface["month"] == ["3"]
        assert era5.surface["day"] == ["13", "14", "15"]
        assert era5.pressure["year"] == ["2022"]

    def test_update_grid_025(self, era5: ERA5API) -> None:
        era5._update_grid()
        assert era5.surface["grid"] == "0.25/0.25"
        assert era5.pressure["grid"] == "0.25/0.25"

    def test_update_grid_100(self) -> None:
        era5 = ERA5API(daterange="2022-03-13|2022-03-13", grid_size="1p00")
        era5._update_grid()
        assert era5.surface["grid"] == "1.0/1.0"

    def test_download_raises_without_cdsapi(self, era5: ERA5API) -> None:
        import evalwrf.api as api_module

        original = api_module._cdsapi
        api_module._cdsapi = None
        try:
            with pytest.raises(ImportError, match="cdsapi"):
                era5.download(as_test=False)
        finally:
            api_module._cdsapi = original

    def test_download_test_mode_prints_filenames(
        self, era5: ERA5API, capsys: pytest.CaptureFixture
    ) -> None:
        mock_client = MagicMock()
        import evalwrf.api as api_module

        original = api_module._cdsapi
        api_module._cdsapi = MagicMock()
        api_module._cdsapi.Client.return_value = mock_client
        try:
            era5.download(as_test=True)
        finally:
            api_module._cdsapi = original

        captured = capsys.readouterr()
        assert "reanalysis-era5" in captured.out

    def test_download_netcdf_sets_format(self, era5: ERA5API) -> None:
        import evalwrf.api as api_module

        original = api_module._cdsapi
        api_module._cdsapi = MagicMock()
        api_module._cdsapi.Client.return_value = MagicMock()
        try:
            era5.download(as_test=True, as_netcdf=True)
        finally:
            api_module._cdsapi = original

        assert era5.surface["data_format"] == "netcdf"
        assert era5.pressure["data_format"] == "netcdf"


# ---------------------------------------------------------------------------
# ZAMGAPI
# ---------------------------------------------------------------------------


class TestZAMGAPI:
    def test_dataset_url_construction(self, zamg: ZAMGAPI) -> None:
        expected = (
            "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-10min"
        )
        assert zamg.dataset_url == expected

    def test_metadata_url(self, zamg: ZAMGAPI) -> None:
        url = zamg.metadata()
        assert url.endswith("/metadata")
        assert zamg.output_format == "json"

    def test_parameter_accumulates(self, zamg: ZAMGAPI) -> None:
        zamg.parameter("TL", "RR")
        zamg.parameter("FF")
        assert zamg.parameters == ["TL", "RR", "FF"]

    def test_stations_accumulates(self, zamg: ZAMGAPI) -> None:
        zamg.stations(15920, 11035)
        zamg.stations(99)
        assert zamg.stationslist == [15920, 11035, 99]

    def test_timeframe_iso_format(self, zamg: ZAMGAPI) -> None:
        zamg.timeframe("2024-01-01", "2024-01-07")
        assert "2024-01-01T00:00:00.000Z" in zamg.time
        assert "2024-01-07T00:00:00.000Z" in zamg.time

    def test_compile_builds_url(self, zamg: ZAMGAPI) -> None:
        zamg.parameter("TL")
        zamg.timeframe("2024-01-01", "2024-01-02")
        zamg.stations(15920)
        url = zamg.compile()

        assert "parameters=TL" in url
        assert "station_ids=15920" in url
        assert "output_format=csv" in url
        assert url.startswith(zamg.dataset_url)

    def test_compile_sets_output_format_csv(self, zamg: ZAMGAPI) -> None:
        zamg.metadata()  # sets output_format to "json"
        zamg.compile()   # should reset to "csv"
        assert zamg.output_format == "csv"

    def test_compile_multiple_parameters(self, zamg: ZAMGAPI) -> None:
        zamg.parameter("TL", "RR")
        url = zamg.compile()
        assert "parameters=TL" in url
        assert "parameters=RR" in url

    def test_compile_multiple_stations(self, zamg: ZAMGAPI) -> None:
        zamg.stations(15920, 11035)
        url = zamg.compile()
        assert "station_ids=15920" in url
        assert "station_ids=11035" in url

    def test_download_aborted_by_user(self, zamg: ZAMGAPI) -> None:
        zamg.compile()
        with patch("builtins.input", return_value="n"):
            with pytest.raises(ValueError, match="aborted"):
                zamg.download("output.csv", schwimmflügerl=True)

    def test_download_writes_file(self, zamg: ZAMGAPI, tmp_path: Path) -> None:
        zamg.compile()
        output = tmp_path / "output.csv"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b"col1,col2\n1,2\n"

        with patch("evalwrf.api.httpx.get", return_value=mock_response):
            zamg.download(str(output), schwimmflügerl=False)

        assert output.exists()
        assert output.read_bytes() == b"col1,col2\n1,2\n"

    def test_download_replaces_extension(self, zamg: ZAMGAPI, tmp_path: Path) -> None:
        zamg.compile()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b""

        with patch("evalwrf.api.httpx.get", return_value=mock_response):
            zamg.download(str(tmp_path / "myfile.txt"), schwimmflügerl=False)

        # output_format is "csv" after compile(), so extension should be .csv
        assert zamg.full_filename.endswith(".csv")

    def test_plot_raises_not_implemented(self, zamg: ZAMGAPI) -> None:
        with pytest.raises(NotImplementedError):
            zamg.plot()

    def test_initial_state(self, zamg: ZAMGAPI) -> None:
        assert zamg.parameters == []
        assert zamg.stationslist == []
        assert zamg.time == ""
        assert zamg.request_url == ""
