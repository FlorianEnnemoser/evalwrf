from pathlib import Path
from typing import Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np


def parse_namelist_wps(path: Union[str, Path]) -> dict:
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


def _meter2lat(meter: float) -> float:
    """Convert a north-south distance in metres to degrees latitude.

    Uses the approximation 1° latitude ≈ 110 574 m.

    Parameters
    ----------
    meter : float
        Distance in metres.

    Returns
    -------
    float
        Equivalent latitude in degrees.
    """
    return meter / (110.574e3)


def _meter2lon(meter: float, lat: float) -> float:
    """Convert an east-west distance in metres to degrees longitude.

    Accounts for the cosine factor at the given latitude.

    Parameters
    ----------
    meter : float
        Distance in metres.
    lat : float
        Reference latitude in degrees North.

    Returns
    -------
    float
        Equivalent longitude in degrees at *lat*.
    """
    return meter / (111.32e3 * np.cos(np.deg2rad(lat)))


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


def find_closest_city(
    lat: float,
    lon: float,
    pop_size: int = 50_000,
) -> dict:
    """Find the nearest populated place above a population threshold.

    Parameters
    ----------
    lat : float
        Query latitude in degrees North.
    lon : float
        Query longitude in degrees East.
    pop_size : int, optional
        Minimum population for consideration.  Defaults to ``50_000``.

    Returns
    -------
    dict
        ``{"name": str, "lat": float, "lon": float}`` for the closest city.
    """
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="populated_places"
    )
    cities = list(shpreader.Reader(shpfilename).records())

    def _distance(city):
        return np.sqrt((lat - city.geometry.y) ** 2 + (lon - city.geometry.x) ** 2)

    large_cities = [c for c in cities if c.attributes.get("POP_MAX", 0) > pop_size]
    closest = min(large_cities, key=_distance)
    return {
        "name": closest.attributes["NAME"],
        "lat": closest.geometry.y,
        "lon": closest.geometry.x,
    }


def plot_grids(
    domain: dict,
    grids: list,
    plot_grid: bool = True,
) -> plt.Figure:
    """Plot WRF domain boundaries on a Lambert Conformal map.

    Parameters
    ----------
    domain : dict of str -> list of str
        Parsed WPS namelist dictionary (from :func:`parse_namelist_wps` or
        :meth:`Namelist._as_wps_dict`).
    grids : list of dict
        Computed grid data as returned by :func:`compute_grid`.
    plot_grid : bool, optional
        Draw individual grid-cell lines within each domain.
        Defaults to ``True``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed domain plot.  The caller is responsible for saving
        or displaying the figure.

    Notes
    -----
    Country borders are rendered with ``edgecolor="black"`` and
    ``linewidth=0.5``.  Coastlines, land fill, rivers, and lakes are also
    included for geographical context.
    """

    base_config = dict(linewidth=1.5, transform=ccrs.PlateCarree())
    grid_config = dict(
        color="blue", linewidth=0.5, transform=ccrs.PlateCarree(), alpha=0.5
    )

    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={
            "projection": ccrs.LambertConformal(
                central_longitude=grids[0]["center_lon"]
            )
        },
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "50m", edgecolor="blue", facecolor="none"
    )
    # physical_labels = cfeature.NaturalEarthFeature('physical', 'geography_regions_polys', '10m',
    #                                     edgecolor='red', facecolor='none')
    # ax.add_feature(physical_labels, linewidth=0.5)
    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor="blue", facecolor="lightblue", alpha=0.5)

    dx = int(domain["dx"][0]) / 1000

    for i, grid in enumerate(grids):
        lons, lats = grid["lons"], grid["lats"]

        if plot_grid:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            ax.plot(lon_grid, lat_grid, **grid_config)
            ax.plot(lon_grid.T, lat_grid.T, **grid_config)

        color = plt.get_cmap(name="tab10")(i / len(grids))
        ax.plot(lons, [max(lat_grid[:, 0])] * len(lons), **base_config, color=color)
        ax.plot(lons, [min(lat_grid[:, 0])] * len(lons), **base_config, color=color)
        ax.plot([min(lon_grid[0])] * len(lats), lats, **base_config, color=color)
        ax.plot(
            [max(lon_grid[0])] * len(lats),
            lats,
            **base_config,
            color=color,
            label=f"Domain {i + 1}",
        )

        dx /= int(domain["parent_grid_ratio"][i])

    ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    ax.legend(loc="upper left")

    title_string = ["Center of domains:"]
    title_string.extend(
        [
            f"Domain {i + 1}: {d['center_lat']:.1f}° {d['center_lon']:.1f}°"
            for i, d in enumerate(grids)
        ]
    )
    ax.set_title("\n".join(title_string))
    #########
    #########
    #########
    ######### new below

    # center_lon = grids[0]["center_lon"]
    # projection = ccrs.LambertConformal(central_longitude=center_lon)

    # fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": projection})

    # ax.add_feature(cfeature.LAND, facecolor="lightgray")
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    # ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)

    # rivers = cfeature.NaturalEarthFeature(
    #     "physical",
    #     "rivers_lake_centerlines",
    #     "50m",
    #     edgecolor="steelblue",
    #     facecolor="none",
    # )
    # ax.add_feature(rivers, linewidth=0.4)
    # ax.add_feature(
    #     cfeature.LAKES, edgecolor="steelblue", facecolor="lightblue", alpha=0.5
    # )

    # transform = ccrs.PlateCarree()
    # colors = plt.get_cmap("tab10")

    # for i, grid in enumerate(grids):
    #     lons = grid["lons"]
    #     lats = grid["lats"]
    #     color = colors(i / max(len(grids), 1))
    #     lon_grid, lat_grid = np.meshgrid(lons, lats)

    #     if plot_grid:
    #         grid_kw = dict(color=color, linewidth=0.3, transform=transform, alpha=0.4)
    #         ax.plot(lon_grid, lat_grid, **grid_kw)
    #         ax.plot(lon_grid.T, lat_grid.T, **grid_kw)

    #     border_kw = dict(linewidth=1.5, color=color, transform=transform)
    #     ax.plot(lons, [lat_grid[:, 0].max()] * len(lons), **border_kw)
    #     ax.plot(lons, [lat_grid[:, 0].min()] * len(lons), **border_kw)
    #     ax.plot([lon_grid[0].min()] * len(lats), lats, **border_kw)
    #     ax.plot(
    #         [lon_grid[0].max()] * len(lats),
    #         lats,
    #         label=f"Domain {i + 1}  (dx={grid['dx'] / 1000:.1f} km)",
    #         **border_kw,
    #     )

    # ax.gridlines(
    #     draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    # )
    # ax.legend(loc="upper left")

    # title_lines = ["WRF domain configuration"] + [
    #     f"  D{i + 1}: centre {g['center_lat']:.2f}°N  {g['center_lon']:.2f}°E"
    #     for i, g in enumerate(grids)
    # ]
    # ax.set_title("\n".join(title_lines))

    return fig


def from_namelist(path: Union[str, Path] = "namelist.wps") -> plt.Figure:
    """Parse, compute, and plot WRF domains from a ``namelist.wps`` file.

    End-to-end convenience function that combines :func:`parse_namelist_wps`,
    :func:`compute_grid`, and :func:`plot_grids`.

    Parameters
    ----------
    path : str or Path, optional
        Path to ``namelist.wps``.  Defaults to ``"namelist.wps"``.

    Returns
    -------
    matplotlib.figure.Figure
        The domain plot figure.
    """
    domain = parse_namelist_wps(path)
    grids = compute_grid(domain)
    return plot_grids(domain, grids, plot_grid=True)
