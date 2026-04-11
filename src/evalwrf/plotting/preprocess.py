import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from ..preprocess.namelist import compute_grid, parse_namelist_wps


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


def plot_namelist(path: str = "namelist.wps") -> plt.Figure:
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
