"""
windfarm.py

Mathematical functions for WRF wind farm analysis, following:

    Cuevas-Figueroa, Stansby & Stallard (2022).
    "Accuracy of WRF for prediction of operational wind farm data and
    assessment of influence of upwind farms on power production."
    Energy 254, 124362.  https://doi.org/10.1016/j.energy.2022.124362

All equation numbers referenced in docstrings correspond to the paper above.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from .stats import weibull, fit_weibull_mle

# ---------------------------------------------------------------------------
# 2.  Wind speed / direction occurrence  (Section 2.4)
# ---------------------------------------------------------------------------


def wind_speed_occurrence(
    wind_speeds,
    directions=None,
    n_speed_bins=25,
    du=1.0,
    n_dir_bins=36,
    speed_max=None,
):
    """
    Compute the normalised occurrence frequency of wind speed (and direction).

    With *directions* = None a 1-D wind speed histogram is returned; otherwise
    a 2-D occurrence matrix f(u, theta) is returned, matching the formulation
    used in Eq. (2) (n = 25 speed bins with du = 1 m/s, m = 36 direction bins
    with dtheta = 10 degrees).

    Parameters
    ----------
    wind_speeds : array_like
        Wind speed samples (m/s).
    directions : array_like or None
        Wind direction samples (degrees, meteorological convention). When
        supplied a 2-D occurrence matrix of shape (n_speed_bins, n_dir_bins)
        is returned.
    n_speed_bins : int
        Number of wind speed bins. Paper default: n = 25.
    du : float
        Wind speed bin width (m/s). Paper default: 1.0 m/s.
    n_dir_bins : int
        Number of direction bins. Paper default: m = 36 (dtheta = 10 deg).
        Ignored when directions is None.
    speed_max : float or None
        Upper edge of the highest speed bin. Defaults to n_speed_bins * du.

    Returns
    -------
    freq : ndarray
        Shape (n_speed_bins,) when directions is None, otherwise
        (n_speed_bins, n_dir_bins). Normalised so that the sum equals 1.
    speed_centers : ndarray
        Central wind speed of each bin (m/s).
    dir_centers : ndarray or None
        Central direction of each bin (degrees). None when directions is None.
    """
    u = np.asarray(wind_speeds, dtype=float)
    if speed_max is None:
        speed_max = n_speed_bins * du

    speed_edges = np.linspace(0.0, speed_max, n_speed_bins + 1)
    speed_centers = 0.5 * (speed_edges[:-1] + speed_edges[1:])

    if directions is None:
        counts, _ = np.histogram(u, bins=speed_edges)
        total = counts.sum()
        freq = counts / total if total > 0 else counts.astype(float)
        return freq, speed_centers, None

    theta = np.asarray(directions, dtype=float) % 360.0
    dir_edges = np.linspace(0.0, 360.0, n_dir_bins + 1)
    dir_centers = 0.5 * (dir_edges[:-1] + dir_edges[1:])

    counts, _, _ = np.histogram2d(u, theta, bins=[speed_edges, dir_edges])
    total = counts.sum()
    freq = counts / total if total > 0 else counts.astype(float)
    return freq, speed_centers, dir_centers


# ---------------------------------------------------------------------------
# 3.  Error metrics: RMSE_f, RMSE_W, RMSE_theta, cv  (Section 2.4, Eqs. 2-3)
# ---------------------------------------------------------------------------


def rmse_occurrence(f_ref, f_sim):
    """
    Root Mean Square Error of wind speed (and direction) occurrence frequency
    (Eq. 2).

    .. math::
        RMSE_f = \\sqrt{\\frac{\\sum_i \\sum_j
                 [f_0(u_i, theta_j) - f_s(u_i, theta_j)]^2}{n m}}

    Works for both 1-D (wind speed only, m = 1) and 2-D (wind speed x
    direction) occurrence arrays.

    Parameters
    ----------
    f_ref : array_like
        Reference (measured) occurrence frequency. Shape (n,) or (n, m).
    f_sim : array_like
        Simulated occurrence frequency. Same shape as f_ref.

    Returns
    -------
    float
        RMSE_f of occurrence frequency (dimensionless).
    """
    f_ref = np.asarray(f_ref, dtype=float)
    f_sim = np.asarray(f_sim, dtype=float)
    return float(np.sqrt(np.mean((f_ref - f_sim) ** 2)))


def rmse_weibull(k_ref, c_ref, k_sim, c_sim, u_bins):
    """
    RMSE between two Weibull probability density functions evaluated at
    discrete wind speed bin centres (RMSE_W, Section 4.1).

    Parameters
    ----------
    k_ref, c_ref : float
        Weibull shape and scale of the reference (measured) distribution.
    k_sim, c_sim : float
        Weibull shape and scale of the simulated distribution.
    u_bins : array_like
        Wind speed bin centres at which to evaluate the PDFs (m/s).

    Returns
    -------
    float
        RMSE_W between the two Weibull PDFs.
    """
    u = np.asarray(u_bins, dtype=float)
    pdf_ref = weibull(u, k_ref, c_ref)
    pdf_sim = weibull(u, k_sim, c_sim)
    return float(np.sqrt(np.mean((pdf_ref - pdf_sim) ** 2)))


def rmse_wind_rose(f_ref, f_sim):
    """
    RMSE of the wind direction occurrence distribution (RMSE_theta, Section 4.1).

    Applies the same formula as :func:`rmse_occurrence` to directional
    marginal distributions (shape (m,)) or full 2-D wind-rose arrays.

    Parameters
    ----------
    f_ref : array_like
        Reference direction occurrence frequency. Shape (m,) or (n, m).
    f_sim : array_like
        Simulated direction occurrence frequency. Same shape as f_ref.

    Returns
    -------
    float
        RMSE_theta of wind direction occurrence (dimensionless).
    """
    return rmse_occurrence(f_ref, f_sim)


# ---------------------------------------------------------------------------
# 4.  Power curve & turbine energy  (Section 3.3, Eqs. 4-5)
# ---------------------------------------------------------------------------


def interpolate_power_curve(wind_speed, pc_speeds, pc_powers):
    """
    Interpolate power output from a wind turbine manufacturer's power curve.

    Wind speeds outside the range of pc_speeds return 0 kW (below cut-in or
    above cut-out). Linear interpolation is used within the operational range.

    Parameters
    ----------
    wind_speed : array_like
        Hub-height wind speed(s) (m/s).
    pc_speeds : array_like
        Wind speed points of the power curve (m/s), in ascending order.
    pc_powers : array_like
        Electrical power output at each point of the power curve (kW).

    Returns
    -------
    ndarray
        Interpolated power output in kW for each input wind speed.
    """
    u = np.asarray(wind_speed, dtype=float)
    pc_u = np.asarray(pc_speeds, dtype=float)
    pc_p = np.asarray(pc_powers, dtype=float)
    f = interp1d(pc_u, pc_p, kind="linear", bounds_error=False, fill_value=0.0)
    return f(u)


def wind_farm_energy(wind_speeds_matrix, pc_speeds, pc_powers, dt_minutes=10.0):
    """
    Compute wind farm energy yield E_F over a time series (Eq. 4).

    .. math::
        E_F = \\sum_{t=1}^{N_t} \\sum_{i=1}^{N} P_i(U(t)) dt

    where dt = 1/6 h for 10-min sampled data, i runs over all N turbines, and
    t runs over all N_t time steps.

    Parameters
    ----------
    wind_speeds_matrix : array_like, shape (N_t, N)
        Hub-height wind speed (m/s) at each time step t and turbine i.
        A 1-D array of length N_t is also accepted (single turbine).
    pc_speeds : array_like
        Wind speed points of the power curve (m/s).
    pc_powers : array_like
        Power output values of the power curve (kW).
    dt_minutes : float
        Time step duration in minutes. Default is 10.

    Returns
    -------
    float
        Total wind farm energy yield in kWh.
    """
    U = np.atleast_2d(np.asarray(wind_speeds_matrix, dtype=float))
    dt_h = dt_minutes / 60.0
    P = interpolate_power_curve(U.ravel(), pc_speeds, pc_powers).reshape(U.shape)
    return float(np.sum(P) * dt_h)


def capacity_factor(
    energy_kwh, n_turbines, rated_power_kw, n_timesteps, dt_minutes=10.0
):
    """
    Wind farm capacity factor CF_F (Eq. 5).

    .. math::
        CF_F = \\frac{E_F}{N P_R N_t dt}

    Parameters
    ----------
    energy_kwh : float
        Total energy yield of the wind farm (kWh).
    n_turbines : int
        Number of wind turbines N.
    rated_power_kw : float
        Rated (nameplate) power of a single turbine P_R (kW).
    n_timesteps : int
        Total number of time steps N_t.
    dt_minutes : float
        Duration of each time step in minutes. Default is 10.

    Returns
    -------
    float
        Capacity factor (dimensionless, 0-1). Multiply by 100 for %.
    """
    dt_h = dt_minutes / 60.0
    denom = n_turbines * rated_power_kw * n_timesteps * dt_h
    if denom == 0.0:
        return np.nan
    return float(energy_kwh / denom)


# ---------------------------------------------------------------------------
# 5.  Wake-effect metrics  (Sections 5.2-5.3)
# ---------------------------------------------------------------------------


def wind_speed_deficit(u_resource, u_wake):
    """
    Wind speed deficit delta_u due to upstream wind farm wake effects
    (Section 5.2, Fig. 13).

    .. math::
        \\Delta u = u_w - u_r

    Negative values indicate a speed reduction caused by the upwind farm.

    Parameters
    ----------
    u_resource : array_like
        Wind speed from resource-only (no wake) WRF simulation (m/s).
    u_wake : array_like
        Wind speed from WRF simulation with wake effects modelled (m/s).

    Returns
    -------
    ndarray
        Wind speed deficit (m/s). Negative = speed reduction.
    """
    u_r = np.asarray(u_resource, dtype=float)
    u_w = np.asarray(u_wake, dtype=float)
    return u_w - u_r


def wind_speed_deficit_fraction(u_resource, u_wake):
    """
    Fractional wind speed deficit at each time step or grid point.

    .. math::
        \\delta_u = \\frac{u_r - u_w}{u_r}

    Parameters
    ----------
    u_resource : array_like
        Resource-only wind speed (m/s). Zero values yield NaN.
    u_wake : array_like
        Wake-simulation wind speed (m/s).

    Returns
    -------
    ndarray
        Fractional deficit (dimensionless). Positive = speed reduction.
    """
    u_r = np.asarray(u_resource, dtype=float)
    u_w = np.asarray(u_wake, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(u_r != 0.0, (u_r - u_w) / u_r, np.nan)


def power_deficit_pct(p_resource, p_wake):
    """
    Percentage reduction in power output due to upstream wind farm wakes
    (Section 5.2.2, Fig. 15-16).

    .. math::
        \\Delta P (\\%) = \\frac{P_r - P_w}{P_r} \\times 100

    Parameters
    ----------
    p_resource : array_like
        Power output (kW) from resource-only simulation (no wakes).
    p_wake : array_like
        Power output (kW) from simulation with wake effects.

    Returns
    -------
    ndarray
        Power deficit in percent. Positive = reduction.
    """
    p_r = np.asarray(p_resource, dtype=float)
    p_w = np.asarray(p_wake, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p_r != 0.0, (p_r - p_w) / p_r * 100.0, np.nan)


def energy_yield_error_pct(e_wrf, e_scada):
    """
    Percentage error of WRF-predicted energy yield relative to SCADA data
    (Section 4.2, Fig. 10).

    .. math::
        \\varepsilon (\\%) = \\frac{E_{WRF} - E_{SCADA}}{E_{SCADA}} \\times 100

    Parameters
    ----------
    e_wrf : float or array_like
        WRF-predicted energy yield (kWh or GWh).
    e_scada : float or array_like
        SCADA-measured energy yield (same units).

    Returns
    -------
    float or ndarray
        Relative error in percent. Positive = overprediction.
    """
    e_w = np.asarray(e_wrf, dtype=float)
    e_s = np.asarray(e_scada, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(e_s != 0.0, (e_w - e_s) / e_s * 100.0, np.nan)


def energy_deficit_pct(e_resource, e_wake):
    """
    Percentage energy deficit at a downwind farm due to upstream wake effects
    (Tables 4 and 6).

    .. math::
        \\text{Deficit} (\\%) = \\frac{E_r - E_w}{E_r} \\times 100

    Parameters
    ----------
    e_resource : float
        Energy yield from resource-only simulation (kWh or GWh).
    e_wake : float
        Energy yield from wake-effect simulation (same units).

    Returns
    -------
    float
        Energy deficit in percent. Positive = reduction due to wakes.
    """
    if e_resource == 0.0:
        return np.nan
    return float((e_resource - e_wake) / e_resource * 100.0)


# ---------------------------------------------------------------------------
# 6.  Representative subset selection  (Section 2.4)
# ---------------------------------------------------------------------------


def select_representative_subset(
    candidate_wind_speeds,
    annual_wind_speeds,
    candidate_directions=None,
    annual_directions=None,
    n_speed_bins=25,
    du=1.0,
    n_dir_bins=36,
    metric="occurrence",
):
    """
    Select the most representative subset of wind data relative to the annual
    distribution by minimising RMSE_f (or RMSE_W) over a list of candidates.

    The paper (Section 2.4) identifies weekly subsets from the annual time
    series whose wind speed (and direction) distribution most closely matches
    the annual statistics. This function computes the error metric for each
    candidate and returns the index of the best-matching one.

    Parameters
    ----------
    candidate_wind_speeds : list of array_like
        Wind speed arrays for each candidate subset (e.g. one per week).
    annual_wind_speeds : array_like
        Full-year wind speed record used as the reference distribution.
    candidate_directions : list of array_like or None
        Direction arrays (degrees) for each candidate. When supplied, the 2-D
        occurrence RMSE (wind rose) is used instead of the 1-D speed RMSE.
    annual_directions : array_like or None
        Full-year direction record (degrees). Required if candidate_directions
        is supplied.
    n_speed_bins : int
        Number of wind speed bins (paper: n = 25).
    du : float
        Wind speed bin width (m/s). Default is 1.0.
    n_dir_bins : int
        Number of direction bins (paper: m = 36).
    metric : {"occurrence", "weibull"}
        Error metric. "occurrence" uses RMSE_f (Eq. 2);
        "weibull" compares Weibull PDFs (RMSE_W).

    Returns
    -------
    best_idx : int
        Index into candidate_wind_speeds of the best-matching subset.
    errors : ndarray
        RMSE value computed for every candidate.
    """
    annual_u = np.asarray(annual_wind_speeds, dtype=float)
    use_2d = (candidate_directions is not None) and (annual_directions is not None)

    if metric == "weibull":
        k_ref, c_ref = fit_weibull_mle(annual_u)
        u_centers = np.arange(0.5 * du, n_speed_bins * du, du)
    else:
        if use_2d:
            f_ref, _, _ = wind_speed_occurrence(
                annual_u, annual_directions, n_speed_bins, du, n_dir_bins
            )
        else:
            f_ref, _, _ = wind_speed_occurrence(annual_u, None, n_speed_bins, du)

    errors = np.empty(len(candidate_wind_speeds))
    for idx, cand_u in enumerate(candidate_wind_speeds):
        cand_u = np.asarray(cand_u, dtype=float)
        if metric == "weibull":
            k_sim, c_sim = fit_weibull_mle(cand_u)
            errors[idx] = rmse_weibull(k_ref, c_ref, k_sim, c_sim, u_centers)
        else:
            if use_2d:
                cand_d = np.asarray(candidate_directions[idx], dtype=float)
                f_sim, _, _ = wind_speed_occurrence(
                    cand_u, cand_d, n_speed_bins, du, n_dir_bins
                )
            else:
                f_sim, _, _ = wind_speed_occurrence(cand_u, None, n_speed_bins, du)
            errors[idx] = rmse_occurrence(f_ref, f_sim)

    best_idx = int(np.argmin(errors))
    return best_idx, errors


# ---------------------------------------------------------------------------
# 7.  WRF spatial interpolation  (Section 3.2)
# ---------------------------------------------------------------------------


def bilinear_interpolate_horizontal(
    field, field_lats, field_lons, target_lat, target_lon
):
    """
    Bi-linear interpolation of a 2-D WRF field to a (lat, lon) point.

    The paper (Section 3.2) extracts WRF wind speed at met-mast locations
    using bi-linear interpolation on the horizontal grid of the highest-
    resolution domain (400 m).

    Parameters
    ----------
    field : array_like, shape (ny, nx)
        2-D WRF output field (e.g. wind speed at a single height level).
    field_lats : array_like, shape (ny,)
        Latitude coordinates of the WRF grid (degrees North), monotonic.
    field_lons : array_like, shape (nx,)
        Longitude coordinates of the WRF grid (degrees East), monotonic.
    target_lat : float
        Latitude of the target point (degrees North).
    target_lon : float
        Longitude of the target point (degrees East).

    Returns
    -------
    float
        Interpolated field value at (target_lat, target_lon).
    """
    lats = np.asarray(field_lats, dtype=float)
    lons = np.asarray(field_lons, dtype=float)
    Z = np.asarray(field, dtype=float)
    interp = RegularGridInterpolator(
        (lats, lons), Z, method="linear", bounds_error=False, fill_value=np.nan
    )
    return float(interp([[target_lat, target_lon]])[0])


def linear_interpolate_vertical(heights, values, target_height):
    """
    Linear interpolation of a WRF vertical profile to a target height.

    Used together with :func:`bilinear_interpolate_horizontal` to extract
    hub-height wind speeds from multi-level WRF output (Section 3.2).

    Parameters
    ----------
    heights : array_like
        Heights of the WRF model levels (m above ground), ascending order.
    values : array_like
        Wind speed (or other quantity) at each model level.
    target_height : float
        Target height (m), e.g. turbine hub height of 78 m or LR met-mast
        anemometer at 60 m.

    Returns
    -------
    float
        Linearly interpolated value at target_height.
    """
    h = np.asarray(heights, dtype=float)
    v = np.asarray(values, dtype=float)
    f = interp1d(h, v, kind="linear", fill_value="extrapolate")
    return float(f(target_height))


def extract_wrf_wind_speed_at_point(
    u_component,
    v_component,
    wrf_lats,
    wrf_lons,
    wrf_heights,
    target_lat,
    target_lon,
    target_height,
):
    """
    Extract WRF wind speed magnitude at a (lat, lon, height) point using
    bi-linear horizontal and linear vertical interpolation (Section 3.2).

    Replicates the extraction of wind speed values at met-mast locations
    (La Rumorosa at 60 m, La Zacatosa at 80 m) and at turbine hub height
    (78 m).

    Parameters
    ----------
    u_component : array_like, shape (n_levels, ny, nx)
        WRF eastward wind component (m/s) for a single time step.
    v_component : array_like, shape (n_levels, ny, nx)
        WRF northward wind component (m/s) for the same time step.
    wrf_lats : array_like, shape (ny,)
        Latitude of WRF grid rows (degrees North), monotonically increasing.
    wrf_lons : array_like, shape (nx,)
        Longitude of WRF grid columns (degrees East), monotonically increasing.
    wrf_heights : array_like, shape (n_levels,)
        Height of each WRF vertical level above ground (m).
    target_lat : float
        Latitude of the target location (degrees North).
    target_lon : float
        Longitude of the target location (degrees East).
    target_height : float
        Target height above ground (m).

    Returns
    -------
    float
        Wind speed magnitude (m/s) at the target point.
    """
    u_3d = np.asarray(u_component, dtype=float)
    v_3d = np.asarray(v_component, dtype=float)
    h = np.asarray(wrf_heights, dtype=float)

    u_profile = np.array(
        [
            bilinear_interpolate_horizontal(
                u_3d[k], wrf_lats, wrf_lons, target_lat, target_lon
            )
            for k in range(len(h))
        ]
    )
    v_profile = np.array(
        [
            bilinear_interpolate_horizontal(
                v_3d[k], wrf_lats, wrf_lons, target_lat, target_lon
            )
            for k in range(len(h))
        ]
    )

    u_at_h = linear_interpolate_vertical(h, u_profile, target_height)
    v_at_h = linear_interpolate_vertical(h, v_profile, target_height)
    return float(np.sqrt(u_at_h**2 + v_at_h**2))


# ---------------------------------------------------------------------------
# 8.  Utility helpers
# ---------------------------------------------------------------------------


def wind_direction_from_components(u, v):
    """
    Compute meteorological wind direction (degrees) from u and v wind
    components.

    Meteorological convention: 0 deg = wind blowing from North, 90 deg =
    from East.

    Parameters
    ----------
    u : array_like
        Eastward wind component (m/s). Positive = blowing eastward.
    v : array_like
        Northward wind component (m/s). Positive = blowing northward.

    Returns
    -------
    ndarray
        Wind direction in degrees [0, 360), meteorological convention.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def annual_energy_production(power_kw, dt_minutes=10.0):
    """
    Compute total energy from a power time series (AEP when the series spans
    one year).

    Parameters
    ----------
    power_kw : array_like
        Power output time series in kW. NaN values are ignored.
    dt_minutes : float
        Duration of each time step in minutes. Default is 10.

    Returns
    -------
    float
        Total energy in kWh.
    """
    p = np.asarray(power_kw, dtype=float)
    dt_h = dt_minutes / 60.0
    return float(np.nansum(p) * dt_h)


def filter_by_wind_direction(data, directions, heading, tolerance=5.0):
    """
    Retain only samples within a heading +/- tolerance directional band.

    Used to isolate wake effects for a specific wind direction, e.g. the 67 deg
    bearing from La Zacatosa to La Rumorosa (Section 5.2).

    Parameters
    ----------
    data : array_like
        Data array to filter; first axis must align with directions.
    directions : array_like
        Wind direction at each sample (degrees, 0-360).
    heading : float
        Target heading (degrees).
    tolerance : float
        Half-width of the direction band (degrees). Paper uses +/-5 deg.

    Returns
    -------
    filtered_data : ndarray
        Subset of data within the heading band.
    mask : ndarray of bool
        Boolean mask applied to the data.
    """
    dirs = np.asarray(directions, dtype=float) % 360.0
    diff = np.abs((dirs - heading + 180.0) % 360.0 - 180.0)
    mask = diff <= tolerance
    return np.asarray(data)[mask], mask
