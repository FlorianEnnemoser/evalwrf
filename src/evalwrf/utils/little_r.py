from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import fortranformat as ff
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sentinel / missing-value constants (LITTLE_R convention)
# ---------------------------------------------------------------------------

MISSING: float = -888888.0
"""Value used to indicate a missing observation."""

SENTINEL: float = -777777.0
"""Value used to mark the end of a LITTLE_R observation block."""

MISSING_QC: int = -88
"""Quality-control flag for a missing observation."""

GOOD_QC: int = 0
"""Quality-control flag for a good (valid) observation."""


# ---------------------------------------------------------------------------
# Helper functions (ported from _old/geosphere2litter.py)
# ---------------------------------------------------------------------------


def _saturation_water_vapor_pressure(T: float) -> float:
    """Saturation vapour pressure via the August-Roche-Magnus approximation.

    Parameters
    ----------
    T : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Saturation vapour pressure in Pascals.
    """
    return 0.61094 * np.exp(17.625 * T / (T + 243.04)) * 1e3


def _dewpoint_from_rh(T: float, rh: float) -> float:
    """Derive dewpoint temperature from temperature and relative humidity.

    Uses the inverse August-Roche-Magnus formula.

    Parameters
    ----------
    T : float
        Air temperature in degrees Celsius.
    rh : float
        Relative humidity as a percentage (0–100).

    Returns
    -------
    float
        Dewpoint temperature in degrees Celsius.
    """
    rh_frac = max(rh, 1e-6) / 100.0
    gamma = math.log(rh_frac) + 17.625 * T / (243.04 + T)
    return 243.04 * gamma / (17.625 - gamma)


def _u_component(speed: float, direction: float) -> float:
    """Compute the U (west-east) wind component.

    Uses the meteorological convention where *direction* is the angle
    **from** which the wind blows, measured clockwise from North.

    Parameters
    ----------
    speed : float
        Wind speed in m/s.
    direction : float
        Wind direction in meteorological degrees (0° = wind from North,
        90° = wind from East).

    Returns
    -------
    float
        U component in m/s (positive = eastward).
    """
    return -speed * math.sin(math.radians(direction))


def _v_component(speed: float, direction: float) -> float:
    """Compute the V (south-north) wind component.

    Uses the meteorological convention where *direction* is the angle
    **from** which the wind blows, measured clockwise from North.

    Parameters
    ----------
    speed : float
        Wind speed in m/s.
    direction : float
        Wind direction in meteorological degrees.

    Returns
    -------
    float
        V component in m/s (positive = northward).
    """
    return -speed * math.cos(math.radians(direction))


# ---------------------------------------------------------------------------
# Converter class
# ---------------------------------------------------------------------------


@dataclass
class LittleRConverter:
    """Convert a CSV of surface observations to WRFDA LITTLE_R format.

    Each row of the CSV becomes one LITTLE_R observation block consisting of:

    1. A **header record** — station metadata and observation counts.
    2. One **data record** — the actual surface observation (10 variable/QC
       pairs).
    3. Two **end-of-record sentinel lines** — required by the LITTLE_R spec.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file.
    station_lat : float
        Station latitude in degrees North.
    station_lon : float
        Station longitude in degrees East.
    station_elev : float
        Station elevation above sea level in metres.
    station_id : str
        Short station identifier (max 40 characters, e.g. ``"AT015"``).
    station_name : str
        Human-readable station name (max 40 characters).
    time_column : str, optional
        Name of the CSV column holding observation timestamps.  Must be
        parsable by :func:`pandas.to_datetime`.  Defaults to ``"time"``.
    temp_column : str or None, optional
        Column for air temperature **in degrees Celsius**.  The converter
        adds 273.15 to produce Kelvin for the LITTLE_R record.
        Defaults to ``None``.
    dewpoint_column : str or None, optional
        Column for dewpoint temperature **in degrees Celsius**.
        Defaults to ``None``.
    pressure_column : str or None, optional
        Column for surface pressure **in Pascals**.  Defaults to ``None``.
    wind_speed_column : str or None, optional
        Column for wind speed **in m/s**.  Defaults to ``None``.
    wind_dir_column : str or None, optional
        Column for wind direction **in meteorological degrees** (direction
        the wind is coming *from*, 0° = North, 90° = East).
        Defaults to ``None``.
    rh_column : str or None, optional
        Column for relative humidity **as a percentage** (0–100).  When
        ``dewpoint_column`` is ``None`` but both ``rh_column`` and
        ``temp_column`` are provided, the dewpoint is derived from RH.
        Defaults to ``None``.

    Raises
    ------
    ValueError
        If ``station_id`` or ``station_name`` exceeds 40 characters.

    Notes
    -----
    * Pressure is expected in **Pascals**.  Many surface station datasets
      report pressure in hPa — multiply by 100 before or use an appropriate
      column.
    * Timestamps are interpreted as UTC.  Timezone-aware timestamps in
      other zones are converted automatically; naive timestamps are assumed
      to be UTC.
    * The output file is written in ASCII (required by WRFDA ``obsproc``).

    References
    ----------
    LITTLE_R format specification:
    https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html

    Examples
    --------
    >>> conv = LittleRConverter(
    ...     csv_path="murau_data.csv",
    ...     station_lat=47.11,
    ...     station_lon=14.17,
    ...     station_elev=817.0,
    ...     station_id="AT15920",
    ...     station_name="Murau",
    ...     time_column="time",
    ...     temp_column="tl",
    ... )
    >>> conv.convert("OBS_NUDGING101")
    """

    csv_path: Union[str, Path]
    station_lat: float
    station_lon: float
    station_elev: float
    station_id: str
    station_name: str

    time_column: str = field(default="time")
    temp_column: Union[str, None] = field(default=None)
    dewpoint_column: Union[str, None] = field(default=None)
    pressure_column: Union[str, None] = field(default=None)
    wind_speed_column: Union[str, None] = field(default=None)
    wind_dir_column: Union[str, None] = field(default=None)
    rh_column: Union[str, None] = field(default=None)

    # Fortran record writers — created once in __post_init__ and reused
    _header_writer: object = field(init=False, repr=False)
    _data_writer: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)

        if len(self.station_id) > 40:
            raise ValueError(
                f"station_id must be at most 40 characters, got {len(self.station_id)}."
            )
        if len(self.station_name) > 40:
            raise ValueError(
                f"station_name must be at most 40 characters, got {len(self.station_name)}."
            )

        # Header format: lat, lon, id(A40), name(A40), platform(A40), source(A40),
        #   elev, n_levels, num_vld_fld, num_error, num_warning, seq_num, num_dups,
        #   is_sound(L), bogus(L), discard(L), sut, julian, date_char(A20),
        #   slp, ref_pres, ground_temp, sfc_pres, precip, max_temp, min_temp,
        #   night_temp, launch_time, rh, station_pres
        self._header_writer = ff.FortranRecordWriter(
            "(2F20.5,4A40,2F20.5,3I10,2I10,3L10,2I10,A20,11(F13.5,I7))"
        )
        # Data record: 10 (value, QC-flag) pairs
        self._data_writer = ff.FortranRecordWriter("(10(F13.5,I7))")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_value(self, row: pd.Series, column: Union[str, None]) -> float:
        """Extract a float from *row[column]*, returning ``MISSING`` on absence or NaN.

        Parameters
        ----------
        row : pandas.Series
            A single CSV row.
        column : str or None
            Column name, or ``None`` if the variable is not available.

        Returns
        -------
        float
            The observed value, or :data:`MISSING`.
        """
        if column is None:
            return MISSING
        val = row.get(column)
        if val is None:
            return MISSING
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return MISSING
        if math.isnan(fval):
            return MISSING
        return fval

    def _qc(self, value: float) -> int:
        """Return the appropriate QC flag for *value*."""
        return MISSING_QC if value == MISSING else GOOD_QC

    def _build_header(self, timestamp: pd.Timestamp, n_levels: int = 1) -> str:
        """Render the LITTLE_R header record for one observation.

        Parameters
        ----------
        timestamp : pandas.Timestamp
            Observation time (UTC).
        n_levels : int, optional
            Number of data records that follow.  Defaults to ``1``.

        Returns
        -------
        str
            Header line ending with ``"\\n"``.
        """
        # Ensure UTC
        if timestamp.tzinfo is not None:
            ts_utc = timestamp.tz_convert("UTC")
        else:
            ts_utc = timestamp

        date_char = ts_utc.strftime("%Y%m%d%H%M%S      ")  # A20 — pad to 20 chars

        # Seconds since Unix epoch (used as sut field)
        sut = int(ts_utc.timestamp())

        # Julian day (day-of-year)
        julian = ts_utc.day_of_year

        # All optional surface header variables are missing
        missing_pair = [MISSING, GOOD_QC]
        missing_fields = missing_pair * 11  # slp through station_pres

        record = self._header_writer.write(
            [
                self.station_lat,  # lat
                self.station_lon,  # lon
                self.station_id.ljust(40),  # id (A40)
                self.station_name.ljust(40),  # name (A40)
                "FM-12 SYNOP".ljust(40),  # platform (A40)
                "SURFACE".ljust(40),  # source (A40)
                self.station_elev,  # elevation
                float(n_levels),  # num_vld_fld (reused as n_levels here)
                0,  # num_error
                0,  # num_warning
                0,  # seq_num
                0,  # num_dups
                False,  # is_sound
                False,  # bogus
                False,  # discard
                sut,  # seconds since epoch
                julian,  # julian day
                date_char,  # date string (A20)
                *missing_fields,  # 11 surface met variables as (val, qc) pairs
            ]
        )
        return record + "\n"

    def _build_data_record(self, row: pd.Series) -> str:
        """Render one LITTLE_R surface data record from a CSV row.

        The 10 (value, QC-flag) pairs in order are: pressure, height,
        temperature (K), dewpoint (K), wind speed, wind direction, U
        component, V component, relative humidity, thickness.

        Parameters
        ----------
        row : pandas.Series
            A single CSV row.

        Returns
        -------
        str
            Data record line ending with ``"\\n"``.
        """
        # --- Pressure ---
        pressure = self._get_value(row, self.pressure_column)

        # --- Height (station elevation used as surface height) ---
        height = self.station_elev

        # --- Temperature: CSV in °C → LITTLE_R in Kelvin ---
        temp_c = self._get_value(row, self.temp_column)
        temp_k = (temp_c + 273.15) if temp_c != MISSING else MISSING

        # --- Dewpoint: prefer explicit column, derive from RH otherwise ---
        dewpoint_k = MISSING
        if self.dewpoint_column is not None:
            dew_c = self._get_value(row, self.dewpoint_column)
            if dew_c != MISSING:
                dewpoint_k = dew_c + 273.15
        elif self.rh_column is not None and temp_c != MISSING:
            rh = self._get_value(row, self.rh_column)
            if rh != MISSING:
                dew_c = _dewpoint_from_rh(temp_c, rh)
                dewpoint_k = dew_c + 273.15

        # --- Wind speed and direction ---
        speed = self._get_value(row, self.wind_speed_column)
        direction = self._get_value(row, self.wind_dir_column)

        u = v = MISSING
        if speed != MISSING and direction != MISSING:
            u = _u_component(speed, direction)
            v = _v_component(speed, direction)

        # --- Relative humidity ---
        rh = self._get_value(row, self.rh_column)

        # --- Thickness: surface obs have no thickness ---
        thickness = MISSING

        values = [
            pressure,
            self._qc(pressure),
            height,
            self._qc(height),
            temp_k,
            self._qc(temp_k),
            dewpoint_k,
            self._qc(dewpoint_k),
            speed,
            self._qc(speed),
            direction,
            self._qc(direction),
            u,
            self._qc(u),
            v,
            self._qc(v),
            rh,
            self._qc(rh),
            thickness,
            self._qc(thickness),
        ]

        record = self._data_writer.write(values)
        return record + "\n"

    def _build_end_record(self) -> str:
        """Render the two LITTLE_R end-of-record sentinel lines.

        Returns
        -------
        str
            Two sentinel lines, each ending with ``"\\n"``.
        """
        # Line 1: all SENTINEL pairs
        sentinel_values = [SENTINEL, GOOD_QC] * 10
        line1 = self._data_writer.write(sentinel_values)

        # Line 2: final terminator — first pair SENTINEL, second MISSING, rest MISSING
        term_values = [SENTINEL, GOOD_QC, MISSING, GOOD_QC] + [MISSING, GOOD_QC] * 8
        line2 = self._data_writer.write(term_values)

        return line1 + "\n" + line2 + "\n"

    def _build_observation_block(self, row: pd.Series, timestamp: pd.Timestamp) -> str:
        """Assemble a complete LITTLE_R block for one timestep.

        Parameters
        ----------
        row : pandas.Series
            CSV row for this observation.
        timestamp : pandas.Timestamp
            Parsed observation timestamp.

        Returns
        -------
        str
            Complete LITTLE_R observation block (header + data + end records).
        """
        header = self._build_header(timestamp, n_levels=1)
        data = self._build_data_record(row)
        end = self._build_end_record()
        return header + data + end

    def convert(self, output_path: Union[str, Path]) -> None:
        """Convert the CSV file to LITTLE_R format and write to *output_path*.

        Each CSV row becomes one LITTLE_R observation block.  Rows with an
        unparseable timestamp are skipped with a warning.

        Parameters
        ----------
        output_path : str or Path
            Destination file path.  The file is created (or overwritten).
            Written in ASCII encoding as required by WRFDA ``obsproc``.

        Raises
        ------
        FileNotFoundError
            If :attr:`csv_path` does not exist.
        ValueError
            If :attr:`time_column` is not present in the CSV.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        if self.time_column not in df.columns:
            raise ValueError(
                f"time_column '{self.time_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True)

        with Path(output_path).open("w", encoding="ascii") as f:
            for _, row in df.iterrows():
                ts = row[self.time_column]
                block = self._build_observation_block(row, ts)
                f.write(block)
