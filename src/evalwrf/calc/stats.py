import numpy as np
from scipy.optimize import minimize


def weibull(u: np.ndarray, c: float, k: float) -> np.ndarray:
    """
    Weibull probability density function for wind speed (Eq. 1).

        u : windspeed array # x-achse
        c : shape factor # 50th percentile
        k : scale factor # breite

    .. math::
        f(u) = \\frac{k}{c}\\left(\\frac{u}{c}\\right)^{k-1}
               \\exp\\!\\left[-\\left(\\frac{u}{c}\\right)^k\\right]

    Parameters
    ----------
    u : array_like
        Wind speed values (m/s). Must be >= 0.
    k : float
        Shape factor (dimensionless).
    c : float
        Scale factor (m/s).

    Returns
    -------
    ndarray
        Probability density at each wind speed value.
    """

    return (k / c) * (u / c) ** (k - 1) * np.exp(-((u / c) ** k))


def fit_weibull_mle(wind_speeds):
    """
    Estimate Weibull shape (k) and scale (c) parameters from wind speed data
    using Maximum Likelihood Estimation (MLE), as used in Section 2.2.

    Parameters
    ----------
    wind_speeds : array_like
        Observed wind speed samples (m/s). Zero and NaN values are excluded.

    Returns
    -------
    k : float
        Fitted Weibull shape factor.
    c : float
        Fitted Weibull scale factor (m/s).
    """
    u = np.asarray(wind_speeds, dtype=float)
    u = u[np.isfinite(u) & (u > 0)]

    def neg_log_likelihood(params):
        k, c = params
        if k <= 0 or c <= 0:
            return np.inf
        return -np.sum(np.log(weibull(u, k, c) + 1e-300))

    result = minimize(
        neg_log_likelihood,
        x0=[2.0, float(np.mean(u))],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10_000},
    )
    k, c = result.x
    return float(k), float(c)


def coefficient_of_variation(u_mast, u_wrf):
    """
    Coefficient of variation (cv) between two synchronised wind speed time
    series (Eq. 3).

    .. math::
        cv = \\frac{1}{\\bar{u}} \\sqrt{\\frac{1}{N_t}
             \\sum_{t=1}^{N_t} (u_{1,t} - u_{2,t})^2}

    where u_bar is the mean of the reference (met-mast) series.

    Parameters
    ----------
    u_mast : array_like
        Measured wind speed time series from the met-mast (m/s).
    u_wrf : array_like
        Simulated wind speed time series from WRF at the same time steps (m/s).

    Returns
    -------
    float
        Coefficient of variation (dimensionless). Multiply by 100 for %.
    """
    u1 = np.asarray(u_mast, dtype=float)
    u2 = np.asarray(u_wrf, dtype=float)
    mask = np.isfinite(u1) & np.isfinite(u2)
    u1, u2 = u1[mask], u2[mask]
    u_mean = float(np.mean(u1))
    if u_mean == 0.0:
        return np.nan
    return float((1.0 / u_mean) * np.sqrt(np.mean((u1 - u2) ** 2)))
