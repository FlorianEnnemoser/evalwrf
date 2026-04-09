#!/usr/bin/env python3
"""
Convert one-station CSV data (P, T, U, V, RH) to a LITTLE_R / OBS_DOMAIN file
for WRF OBSGRID / obs-nudging.

Input CSV expected columns:
    time, P, T, U, V, RH

Optional extra columns are ignored.

Assumptions:
- time is UTC
- P is either in hPa or Pa (auto-detected)
- T is either in °C or K (auto-detected)
- U/V are earth-relative wind components in m/s
- RH is in percent
- station metadata is constant for the whole file

Output:
- One LITTLE_R surface report per CSV row
- Report type: FM-12 SYNOP (fixed land station)
"""

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

MISSING_F = -888888.0
END_F = -777777.0
MISSING_I = -888888


@dataclass(frozen=True)
class StationMeta:
    station_id: str = "STA_A"
    station_name: str = "Station A"
    latitude: float = 48.20000
    longitude: float = 16.37000
    elevation_m: float = 200.0
    platform: str = "FM-12 SYNOP"
    source: str = "CSV"


def _fw(text: str, width: int, align: str = ">") -> str:
    """Fixed-width text field."""
    if len(text) > width:
        text = text[:width]
    if align == "<":
        return f"{text:<{width}}"
    return f"{text:>{width}}"


def _f20(x: float) -> str:
    return f"{x:20.5f}"


def _i10(x: int) -> str:
    return f"{x:10d}"


def _i7(x: int) -> str:
    return f"{x:7d}"


def _b1(x: bool) -> str:
    return " T" if x else " F"


def _detect_units(arr: np.ndarray, kind: str) -> np.ndarray:
    """
    Auto-detect likely units and convert to WRF/LITTLE_R conventions.
    """
    vals = arr.astype(float)

    if kind == "pressure":
        # If typical station pressure values look like hPa, convert to Pa.
        # If they already look like Pa, leave as-is.
        return np.where(np.nanmedian(vals) < 2000.0, vals * 100.0, vals)

    if kind == "temperature":
        # If values look like Celsius, convert to Kelvin.
        return np.where(np.nanmedian(vals) < 200.0, vals + 273.15, vals)

    raise ValueError(f"Unknown kind: {kind}")


def _wind_speed_dir_from_uv(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert earth-relative U/V wind components to meteorological wind speed
    and direction (direction FROM which the wind blows, degrees clockwise from north).
    """
    spd = np.hypot(u, v)
    dr = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return spd, dr


def _dewpoint_from_t_rh(temp_k: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Magnus formula. Returns dew point in Kelvin.
    """
    t_c = temp_k - 273.15
    rh = np.clip(rh, 0.1, 100.0)

    a = 17.625
    b = 243.04  # °C
    gamma = np.log(rh / 100.0) + (a * t_c) / (b + t_c)
    td_c = (b * gamma) / (a - gamma)
    return td_c + 273.15


def _parse_time(value: str) -> datetime:
    """
    Parse common CSV time formats and return a UTC datetime.
    Accepts:
      - YYYY-MM-DD HH:MM:SS
      - YYYY-MM-DDTHH:MM:SS
      - YYYYMMDDHHMMSS
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DDTHH:MM
    """
    s = value.strip()
    if not s:
        raise ValueError("Empty time value")

    if s.isdigit() and len(s) == 14:
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    # Handle ISO-ish formats
    s = s.replace("Z", "")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Unsupported time format: {value!r}") from e

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty")

    required = {"time", "P", "T", "U", "V", "RH"}
    missing_cols = required - set(rows[0].keys())
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    times = np.array([_parse_time(r["time"]).timestamp() for r in rows], dtype=np.int64)
    p = np.array([float(r["P"]) for r in rows], dtype=float)
    t = np.array([float(r["T"]) for r in rows], dtype=float)
    u = np.array([float(r["U"]) for r in rows], dtype=float)
    v = np.array([float(r["V"]) for r in rows], dtype=float)
    rh = np.array([float(r["RH"]) for r in rows], dtype=float)

    return times, p, t, u, v, rh


def _header_line(meta: StationMeta, dt: datetime, num_valid_fields: int) -> str:
    """
    Build a LITTLE_R header line using the WRFDA field order shown in the docs.
    We use a fixed land-station SYNOP-style surface report.
    """
    julian_day = dt.timetuple().tm_yday
    date_str = dt.strftime("%Y%m%d%H%M%S")

    # Header fields as used in the WRFDA LITTLE_R help page:
    # lat lon id name platform source elevation num_vld_fld num_errors num_warnings
    # seq_num num_duplicates is_sound is_bogus discard unix_time julian_day date
    # then many mostly-unused quality-control slots
    parts = [
        _f20(meta.latitude),
        _f20(meta.longitude),
        _fw(meta.station_id, 40, "<"),
        _fw(meta.station_name, 40, "<"),
        _fw(meta.platform, 40, "<"),
        _fw(meta.source, 40, "<"),
        _f20(meta.elevation_m),
        _i10(num_valid_fields),
        _i10(0),  # num_errors
        _i10(0),  # num_warnings
        _i10(0),  # seq_number
        _i10(0),  # num_duplicates
        _b1(False),  # is_sound
        _b1(False),  # is_bogus
        _b1(False),  # discard
        _i10(0),  # unix_time (kept as 0 in many examples)
        _i10(julian_day),  # julian day
        _fw(date_str, 20, ">"),
        _f20(MISSING_F),
        _i7(0),  # SLP
        _f20(MISSING_F),
        _i7(0),  # ref pressure
        _f20(MISSING_F),
        _i7(0),  # ground temp
        _f20(MISSING_F),
        _i7(0),  # SST
        _f20(MISSING_F),
        _i7(0),  # SFC pressure
        _f20(MISSING_F),
        _i7(0),  # precip
        _f20(MISSING_F),
        _i7(0),  # max temp
        _f20(MISSING_F),
        _i7(0),  # min temp
        _f20(MISSING_F),
        _i7(0),  # night min temp
        _f20(MISSING_F),
        _i7(0),  # 3h pres change
        _f20(MISSING_F),
        _i7(0),  # 24h pres change
        _f20(MISSING_F),
        _i7(0),  # cloud cover
        _f20(MISSING_F),
        _i7(0),  # ceiling
        _f20(MISSING_F),
        _i7(0),  # precipitable water
    ]
    return "".join(parts) + "\n"


def _data_line(
    p_pa: float, t_k: float, td_k: float, spd: float, dr: float, rh: float
) -> str:
    """
    One data record for a single-level surface report.

    Fields:
      Pressure, QC, Height, QC, Temperature, QC, Dew point, QC,
      Wind speed, QC, Wind direction, QC, Wind U, QC, Wind V, QC,
      Relative humidity, QC, Thickness, QC
    """
    parts = [
        _f20(p_pa),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),  # height unused here
        _f20(t_k),
        _i7(0),
        _f20(td_k),
        _i7(0),
        _f20(spd),
        _i7(0),
        _f20(dr),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),  # U unused in SYNOP surface report
        _f20(MISSING_F),
        _i7(0),  # V unused in SYNOP surface report
        _f20(rh),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),  # thickness unused
    ]
    return "".join(parts) + "\n"


def _end_record() -> str:
    parts = [
        _f20(END_F),
        _i7(0),
        _f20(END_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
        _f20(MISSING_F),
        _i7(0),
    ]
    return "".join(parts) + "\n"


def _tail_line(num_valid_fields: int) -> str:
    return f"{num_valid_fields:7d}{0:7d}{0:7d}\n"


def convert_csv_to_littler(csv_path: Path, out_path: Path, meta: StationMeta) -> None:
    times, p, t, u, v, rh = _read_csv(csv_path)

    # Sort chronologically
    order = np.argsort(times)
    times = times[order]
    p = p[order]
    t = t[order]
    u = u[order]
    v = v[order]
    rh = rh[order]

    p_pa = _detect_units(p, "pressure")
    t_k = _detect_units(t, "temperature")
    rh = np.clip(rh, 0.0, 100.0)
    spd, dr = _wind_speed_dir_from_uv(u, v)
    td_k = _dewpoint_from_t_rh(t_k, rh)

    # For this surface report, use 6 reported meteorological values:
    # pressure, temperature, dew point, wind speed, wind direction, RH
    num_valid_fields = 6

    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for ts, pp, tt, tdp, s, d, r in zip(times, p_pa, t_k, td_k, spd, dr, rh):
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            f.write(_header_line(meta, dt, num_valid_fields))
            f.write(_data_line(pp, tt, tdp, s, d, r))
            f.write(_end_record())
            f.write(_tail_line(num_valid_fields))


if __name__ == "__main__":
    # Example usage:
    #   python csv_to_littler.py input.csv OBS_DOMAIN101
    if len(sys.argv) < 3:
        print("Usage: python csv_to_littler.py input.csv OBS_DOMAIN101")
        sys.exit(1)

    csv_in = Path(sys.argv[1])
    out_file = Path(sys.argv[2])

    meta = StationMeta(
        station_id="STA_A",
        station_name="Station A",
        latitude=48.20000,
        longitude=16.37000,
        elevation_m=200.0,
        platform="FM-12 SYNOP",
        source="CSV",
    )

    convert_csv_to_littler(csv_in, out_file, meta)
    print(f"Wrote {out_file}")
