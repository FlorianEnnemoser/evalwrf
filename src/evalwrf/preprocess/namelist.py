from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

from ..calc import _meter2lat, _meter2lon


@dataclass
class Namelist:
    """WRF namelist generator for ``namelist.wps`` and ``namelist.input``.

    Generates both WPS pre-processing and WRF model input namelists from a
    minimal set of required parameters.  All domain geometry parameters accept
    either a scalar (single domain) or a list of length ``max_dom`` (nested
    domains).

    Parameters
    ----------
    start_date : str
        Simulation start date in ``"YYYY-MM-DD"`` format.
    end_date : str
        Simulation end date in ``"YYYY-MM-DD"`` format.
    ref_lat : float
        Reference latitude of the outermost domain centre (degrees North).
    ref_lon : float
        Reference longitude of the outermost domain centre (degrees East).
    dx : int or list of int, optional
        West-east grid spacing in metres for each domain.  Defaults to
        ``27000``.
    dy : int or list of int, optional
        South-north grid spacing in metres for each domain.  Defaults to
        ``27000``.
    e_we : int or list of int, optional
        West-east grid dimension (number of mass points + 1) per domain.
        Defaults to ``91``.
    e_sn : int or list of int, optional
        South-north grid dimension per domain.  Defaults to ``100``.
    max_dom : int, optional
        Total number of domains (1–3).  Defaults to ``1``.
    i_parent_start : int or list of int, optional
        West-east parent-grid start index for each domain.  Defaults to ``1``.
    j_parent_start : int or list of int, optional
        South-north parent-grid start index for each domain.  Defaults to
        ``1``.
    parent_id : int or list of int, optional
        Parent domain ID for each domain.  Scalars are auto-expanded to
        ``[value] * max_dom``.  Defaults to ``1``.
    parent_grid_ratio : int or list of int, optional
        Nesting ratio relative to the parent domain.  Scalars are
        auto-expanded.  Defaults to ``1``.
    map_proj : str, optional
        Map projection.  One of ``"lambert"``, ``"mercator"``, ``"polar"``,
        or ``"lat-lon"``.  Defaults to ``"lambert"``.
    truelat1 : float, optional
        First true latitude for Lambert/polar projections.  Defaults to
        ``30.0``.
    truelat2 : float, optional
        Second true latitude for Lambert projection.  Defaults to ``60.0``.
    stand_lon : float, optional
        Standard (centre) meridian longitude.  Defaults to ``ref_lon``.
    interval_seconds : int, optional
        Meteorological input interval in seconds.  Defaults to ``21600``
        (6 hours).
    geog_data_path : str, optional
        Path to the WPS static geography data directory.  Defaults to
        ``"/home/wrfuser/terrestrial_data/WPS_GEOG"``.
    geog_data_res : str, optional
        Geographic data resolution identifier.  Defaults to ``"default"``.
    out_format : str, optional
        Ungrib output format.  Defaults to ``"WPS"``.
    prefix : str, optional
        Ungrib file prefix.  Defaults to ``"FILE"``.
    fg_name : str, optional
        Metgrid first-guess file prefix.  Defaults to ``"FILE"``.

    Raises
    ------
    NotImplementedError
        If ``max_dom > 3``.
    ValueError
        If ``start_date >= end_date``, or if a multi-domain list field has
        the wrong length.

    Examples
    --------
    Single-domain simulation::

        >>> nl = Namelist(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-03",
        ...     ref_lat=48.0,
        ...     ref_lon=14.0,
        ... )
        >>> nl.write_wps("namelist.wps")
        >>> nl.write_input("namelist.input")

    Two-domain nested simulation::

        >>> nl = Namelist(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-03",
        ...     ref_lat=48.0,
        ...     ref_lon=14.0,
        ...     max_dom=2,
        ...     e_we=[91, 100],
        ...     e_sn=[100, 112],
        ...     dx=[27000, 9000],
        ...     dy=[27000, 9000],
        ...     i_parent_start=[1, 31],
        ...     j_parent_start=[1, 17],
        ...     parent_id=[1, 1],
        ...     parent_grid_ratio=[1, 3],
        ... )
        >>> fig = nl.plot_grids()
    """

    start_date: str
    end_date: str
    ref_lat: float
    ref_lon: float

    dx: int | list = field(default=27000)
    dy: int | list = field(default=27000)
    e_we: int | list = field(default=91)
    e_sn: int | list = field(default=100)
    max_dom: int = field(default=1)

    i_parent_start: int | list = field(default=1)
    j_parent_start: int | list = field(default=1)
    parent_id: int | list = field(default=1)
    parent_grid_ratio: int | list = field(default=1)

    map_proj: str = field(default="lambert")
    truelat1: float = field(default=30.0)
    truelat2: float = field(default=60.0)
    stand_lon: float = field(default=None)  # type: ignore[assignment]

    interval_seconds: int = field(default=21600)
    geog_data_path: str = field(default="/home/wrfuser/terrestrial_data/WPS_GEOG")
    geog_data_res: str = field(default="default")
    out_format: str = field(default="WPS")
    prefix: str = field(default="FILE")
    fg_name: str = field(default="FILE")

    # Internal state — set by __post_init__, excluded from __init__
    _start_dt: datetime = field(init=False, repr=False)
    _end_dt: datetime = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stand_lon is None:
            self.stand_lon = self.ref_lon

        self._start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        self._end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        self._validate()

    def _validate(self) -> None:
        if self.max_dom > 3:
            raise NotImplementedError("Only max_dom <= 3 is supported.")

        if self._start_dt >= self._end_dt:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})."
            )

        if self.max_dom > 1:
            # Auto-expand scalars for parent_id and parent_grid_ratio
            if not isinstance(self.parent_id, list):
                self.parent_id = [self.parent_id] * self.max_dom
            if not isinstance(self.parent_grid_ratio, list):
                self.parent_grid_ratio = [self.parent_grid_ratio] * self.max_dom

            # Require full lists for domain geometry
            for name, value in [
                ("e_we", self.e_we),
                ("e_sn", self.e_sn),
                ("dx", self.dx),
                ("dy", self.dy),
                ("i_parent_start", self.i_parent_start),
                ("j_parent_start", self.j_parent_start),
            ]:
                if not isinstance(value, list):
                    raise ValueError(
                        f"'{name}' must be a list of length {self.max_dom} "
                        f"when max_dom={self.max_dom}. Got scalar: {value}."
                    )
                if len(value) != self.max_dom:
                    raise ValueError(
                        f"'{name}' has length {len(value)}, expected {self.max_dom}."
                    )

            for name in ["parent_id", "parent_grid_ratio"]:
                value = getattr(self, name)
                if len(value) != self.max_dom:
                    raise ValueError(
                        f"'{name}' has length {len(value)}, expected {self.max_dom}."
                    )

    def _fmt_field(self, value) -> str:
        """Render a scalar or list as a Fortran namelist value string."""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    def _fmt_str_list(self, value: str) -> str:
        """Repeat a quoted string ``max_dom`` times: ``'val', 'val', ...``"""
        return ", ".join(f"'{value}'" for _ in range(self.max_dom))

    def _fmt_val_list(self, value) -> str:
        """Repeat a scalar value ``max_dom`` times: ``v, v, ...``"""
        return ", ".join(str(value) for _ in range(self.max_dom))

    def _as_wps_dict(self) -> dict:
        """Return a dict mirroring the output of ``parse_namelist_wps()``.

        Allows :meth:`plot_grids` to call :func:`evalwrf.preprocess.compute_grid`
        directly without writing a file first.

        Returns
        -------
        dict of str -> list of str
            Keys match WPS namelist parameter names; values are lists of
            strings, matching the shape produced by ``parse_namelist_wps``.
        """

        def _to_list(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            return [str(v)] * self.max_dom

        return {
            "max_dom": [str(self.max_dom)],
            "dx": [str(self.dx[0] if isinstance(self.dx, list) else self.dx)],
            "dy": [str(self.dy[0] if isinstance(self.dy, list) else self.dy)],
            "ref_lat": [str(self.ref_lat)],
            "ref_lon": [str(self.ref_lon)],
            "e_we": _to_list(self.e_we),
            "e_sn": _to_list(self.e_sn),
            "parent_grid_ratio": _to_list(self.parent_grid_ratio),
            "parent_id": _to_list(self.parent_id),
            "i_parent_start": _to_list(self.i_parent_start),
            "j_parent_start": _to_list(self.j_parent_start),
        }

    def _generate_wps(self) -> str:
        """Render the full ``namelist.wps`` content."""
        start = self._start_dt.strftime("%Y-%m-%d_%H:%M:%S")
        end = self._end_dt.strftime("%Y-%m-%d_%H:%M:%S")

        return f"""&share
    wrf_core          = 'ARW',
    max_dom           = {self.max_dom},
    start_date        = {self._fmt_str_list(start)},
    end_date          = {self._fmt_str_list(end)},
    interval_seconds  = {self.interval_seconds},
/

&geogrid
    parent_id         = {self._fmt_field(self.parent_id)},
    parent_grid_ratio = {self._fmt_field(self.parent_grid_ratio)},
    i_parent_start    = {self._fmt_field(self.i_parent_start)},
    j_parent_start    = {self._fmt_field(self.j_parent_start)},
    e_we              = {self._fmt_field(self.e_we)},
    e_sn              = {self._fmt_field(self.e_sn)},
    geog_data_res     = {self._fmt_str_list(self.geog_data_res)},
    dx                = {self._fmt_field(self.dx)},
    dy                = {self._fmt_field(self.dy)},
    map_proj          = '{self.map_proj}',
    ref_lat           = {self.ref_lat},
    ref_lon           = {self.ref_lon},
    truelat1          = {self.truelat1},
    truelat2          = {self.truelat2},
    stand_lon         = {self.stand_lon},
    geog_data_path    = '{self.geog_data_path}',
/

&ungrib
    out_format        = '{self.out_format}',
    prefix            = '{self.prefix}',
/

&metgrid
    fg_name           = '{self.fg_name}',
/
"""

    def _generate_input(self) -> str:
        """Render the full ``namelist.input`` content."""
        start = self._start_dt
        end = self._end_dt

        start_year = self._fmt_val_list(start.year)
        start_month = self._fmt_val_list(f"{start.month:02d}")
        start_day = self._fmt_val_list(f"{start.day:02d}")
        start_hour = self._fmt_val_list(f"{start.hour:02d}")
        end_year = self._fmt_val_list(end.year)
        end_month = self._fmt_val_list(f"{end.month:02d}")
        end_day = self._fmt_val_list(f"{end.day:02d}")
        end_hour = self._fmt_val_list(f"{end.hour:02d}")

        input_from_file = ", ".join([".true."] * self.max_dom)
        history_interval = ", ".join(["60"] * self.max_dom)
        frames_per_outfile = ", ".join(["1"] * self.max_dom)

        return f"""&time_control
    run_days                            = 0,
    run_hours                           = 0,
    run_minutes                         = 0,
    run_seconds                         = 0,
    start_year                          = {start_year},
    start_month                         = {start_month},
    start_day                           = {start_day},
    start_hour                          = {start_hour},
    end_year                            = {end_year},
    end_month                           = {end_month},
    end_day                             = {end_day},
    end_hour                            = {end_hour},
    interval_seconds                    = {self.interval_seconds},
    input_from_file                     = {input_from_file},
    history_interval                    = {history_interval},
    frames_per_outfile                  = {frames_per_outfile},
    restart                             = .false.,
    restart_interval                    = 1440,
    io_form_history                     = 2,
    io_form_restart                     = 2,
    io_form_input                       = 2,
    io_form_boundary                    = 2,
/

&domains
    time_step                           = 150,
    time_step_fract_num                 = 0,
    time_step_fract_den                 = 1,
    max_dom                             = {self.max_dom},
    e_we                                = {self._fmt_field(self.e_we)},
    e_sn                                = {self._fmt_field(self.e_sn)},
    e_vert                              = {self._fmt_val_list(45)},
    dx                                  = {self._fmt_field(self.dx)},
    dy                                  = {self._fmt_field(self.dy)},
    grid_id                             = {self._fmt_field(self.parent_id)},
    parent_id                           = {self._fmt_field(self.parent_id)},
    i_parent_start                      = {self._fmt_field(self.i_parent_start)},
    j_parent_start                      = {self._fmt_field(self.j_parent_start)},
    parent_grid_ratio                   = {self._fmt_field(self.parent_grid_ratio)},
    parent_time_step_ratio              = {self._fmt_val_list(1)},
    feedback                            = 1,
    smooth_option                       = 0,
    num_metgrid_levels                  = 34,
    num_metgrid_soil_levels             = 4,
    dzstretch_s                         = 1.1,
    p_top_requested                     = 5000,
/

&physics
    physics_suite                       = 'CONUS',
    mp_physics                          = {self._fmt_val_list(-1)},
    cu_physics                          = {self._fmt_val_list(-1)},
    ra_lw_physics                       = {self._fmt_val_list(-1)},
    ra_sw_physics                       = {self._fmt_val_list(-1)},
    bl_pbl_physics                      = {self._fmt_val_list(-1)},
    sf_sfclay_physics                   = {self._fmt_val_list(-1)},
    sf_surface_physics                  = {self._fmt_val_list(-1)},
    radt                                = {self._fmt_val_list(15)},
    bldt                                = {self._fmt_val_list(0)},
    cudt                                = {self._fmt_val_list(0)},
    icloud                              = 1,
    num_land_cat                        = 21,
    sf_urban_physics                    = {self._fmt_val_list(0)},
    fractional_seaice                   = 1,
/

&fdda
/

&dynamics
    hybrid_opt                          = 2,
    w_damping                           = 0,
    diff_opt                            = {self._fmt_val_list(2)},
    km_opt                              = {self._fmt_val_list(4)},
    diff_6th_opt                        = {self._fmt_val_list(0)},
    diff_6th_factor                     = {self._fmt_val_list(0.12)},
    base_temp                           = 290.0,
    damp_opt                            = 3,
    zdamp                               = {self._fmt_val_list(5000.0)},
    dampcoef                            = {self._fmt_val_list(0.2)},
    khdif                               = {self._fmt_val_list(0)},
    kvdif                               = {self._fmt_val_list(0)},
    non_hydrostatic                     = {self._fmt_val_list(".true.")},
    moist_adv_opt                       = {self._fmt_val_list(1)},
    scalar_adv_opt                      = {self._fmt_val_list(1)},
    gwd_opt                             = {self._fmt_val_list(1)},
/

&bdy_control
    spec_bdy_width                      = 5,
    specified                           = .true.,
/

&grib2
/

&namelist_quilt
    nio_tasks_per_group                 = 0,
    nio_groups                          = 1,
/
"""

    def write_wps(self, path: str = "namelist.wps") -> None:
        """Write the WPS namelist to *path*.

        Parameters
        ----------
        path : str or Path, optional
            Destination file path.  Defaults to ``"namelist.wps"``.
        """
        Path(path).write_text(self._generate_wps(), encoding="utf-8")
        return None

    def write_input(self, path: str = "namelist.input") -> None:
        """Write the WRF input namelist to *path*.

        Parameters
        ----------
        path : str or Path, optional
            Destination file path.  Defaults to ``"namelist.input"``.
        """
        Path(path).write_text(self._generate_input(), encoding="utf-8")
        return None

    def plot_grids(self, plot_grid: bool = True):
        """Visualise the WRF domain boundaries on a map.

        Calls :func:`evalwrf.preprocess.compute_grid` and
        :func:`evalwrf.preprocess.plot_grids` using this namelist's
        parameters, without requiring a file to be written first.

        Parameters
        ----------
        plot_grid : bool, optional
            Draw individual grid-cell lines inside each domain.
            Defaults to ``True``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated domain plot figure.
        """
        from .. import preprocess

        domain = self._as_wps_dict()
        grids = preprocess.compute_grid(domain)
        return preprocess.plot_grids(domain, grids, plot_grid=plot_grid)


def parse_namelist_wps(path: str) -> dict:
    """Parse a ``namelist.wps`` file into a flat key-to-values dictionary.

    Section headers (``&name``, ``/``) and inline comments (``!``) are
    stripped.  Each parameter key maps to a list of stripped string values,
    matching the multi-value convention used by WRF namelists.

    Parameters
    ----------
    path : str or Path
        Path to the ``namelist.wps`` file.

    Returns
    -------
    dict of str -> list of str
        Each WPS namelist key maps to a list of stripped string values.

    Examples
    --------
    >>> domain = parse_namelist_wps("namelist.wps")
    >>> domain["max_dom"]
    ['1']
    >>> domain["e_we"]
    ['91']
    """
    domain = {}
    with Path(path).open("r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if line.startswith("/") or line.startswith("&") or not line:
                continue
            if "=" in line:
                name, _, raw_values = line.partition("=")
                name = name.strip()
                values = [
                    v.strip().strip("'").strip('"')
                    for v in raw_values.split(",")
                    if v.strip()
                ]
                domain[name] = values
    return domain


def compute_grid(domain: dict) -> list:
    """Compute lat/lon arrays for every WRF domain.

    Parameters
    ----------
    domain : dict of str -> list of str
        Parsed WPS namelist dictionary as returned by
        :func:`parse_namelist_wps` (or :meth:`Namelist._as_wps_dict`).

    Returns
    -------
    list of dict
        One dict per domain with keys:

        ``"lons"`` : numpy.ndarray
            1-D array of longitude values (degrees East) along the west-east
            axis.
        ``"lats"`` : numpy.ndarray
            1-D array of latitude values (degrees North) along the south-north
            axis.
        ``"center_lat"`` : float
            Domain centre latitude.
        ``"center_lon"`` : float
            Domain centre longitude.
        ``"dx"`` : float
            Effective west-east grid spacing in metres.
        ``"dy"`` : float
            Effective south-north grid spacing in metres.

    Raises
    ------
    ValueError
        If any domain's ``e_we`` or ``e_sn`` does not satisfy the WRF nesting
        criterion ``(N - 1) % parent_grid_ratio == 0``.
    """
    grids = []

    for i in range(int(domain["max_dom"][0])):
        parent_grid_ratio = int(domain["parent_grid_ratio"][i])
        if i <= 1:
            dx = int(domain["dx"][0]) / parent_grid_ratio
            dy = int(domain["dy"][0]) / parent_grid_ratio
        else:
            dx = int(domain["dx"][0]) / (parent_grid_ratio * (i + 1))
            dy = int(domain["dy"][0]) / (parent_grid_ratio * (i + 1))

        ref_lat = float(domain["ref_lat"][0])
        ref_lon = float(domain["ref_lon"][0])
        e_we = int(domain["e_we"][i])
        e_sn = int(domain["e_sn"][i])

        if (e_we - 1) % parent_grid_ratio != 0:
            min_n = (e_we - 1) // parent_grid_ratio
            suggested_e_we = [
                (n * parent_grid_ratio + 1) for n in range(min_n, min_n + 5)
            ]
            raise ValueError(
                f"Domain {i + 1}: e_we={e_we} does not satisfy the nesting criterion. Try: {suggested_e_we}"
            )
        if (e_sn - 1) % parent_grid_ratio != 0:
            min_n = (e_sn - 1) // parent_grid_ratio
            suggested_e_sn = [
                (n * parent_grid_ratio + 1) for n in range(min_n, min_n + 5)
            ]
            raise ValueError(
                f"Domain {i + 1}: e_sn={e_sn} does not satisfy the nesting criterion. Try: {suggested_e_sn}"
            )

        if i == 0:
            center_lat = ref_lat
            center_lon = ref_lon

            i_start = j_start = 0
        else:
            parent_index = int(domain["parent_id"][i]) - 1

            i_start = int(domain["i_parent_start"][i])
            j_start = int(domain["j_parent_start"][i])

            start_lat = grids[parent_index]["lats"][j_start]
            start_lon = grids[parent_index]["lons"][i_start]

            width = _meter2lon((e_we - 1) * dx, start_lat)
            height = _meter2lat((e_sn - 1) * dy)

            center_lat = start_lat + height / 2.0  # / 111e3
            center_lon = start_lon + width / 2.0  # / 111e3

        grid_spacing_lon = _meter2lon(dx, center_lat)
        grid_spacing_lat = _meter2lat(dy)

        lons = center_lon + (np.arange(e_we) - e_we / 2) * grid_spacing_lon
        lats = center_lat + (np.arange(e_sn) - e_sn / 2) * grid_spacing_lat
        grids.append(
            {
                "lons": lons,
                "lats": lats,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "dx": dx,
                "dy": dy,
            }
        )
    return grids
