from itertools import cycle
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["font.family"] = "Courier New"


def centerpoint_extend(center_lat: float, center_lon: float, extend: float) -> list:
    left = center_lon - extend / 2
    right = center_lon + extend / 2
    lower = center_lat - extend / 2
    upper = center_lat + extend / 2
    return [left, right, lower, upper]


def plot_field(
    da: xr.DataArray,
    lat: str = "lat",
    lon: str = "lon",
    title: str | None = None,
    levels: int | str = 10,
    cmap: str = "jet",
    extent: list[float] = [9.0, 17.5, 46.0, 49.1],
    figsize: tuple[float, float] = (10, 10 / 3),
) -> plt.Figure:
    """
    Create a cartopy map plot of a NetCDF field with a colorbar.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with coordinates (lon, lat, time).
    timestep : dict[str,int]
        Time index to plot. Use the time coordinate / dimension with the desired index.
    levels : int | str, optional
        Colorbar levels. If int, creates `levels` equally-spaced levels from min to max.
        If str, parsed as "start:stop:step" (e.g., "0:10:2" → [0, 2, 4, 6, 8, 10]).
        Default: 10.
    cmap : str, optional
        Colormap name (default: "jet").
    extent : list[float], optional
        Map extent [lon_min, lon_max, lat_min, lat_max] in degrees.
        Default: [9.0, 17.5, 46.0, 49.1] (Austria region).
    figsize : tuple[float, float], optional
        Figure size (width, height) in inches. Default: (10, 10/3).

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Cartopy GeoAxes object.

    Notes
    -----
    - The colorbar width is set to exactly match the plot width.
    - All text uses "Courier New" font.
    - Title shows the date and hour of the selected timestep (UTC).
    """
    if not isinstance(da, xr.DataArray):
        raise ValueError("Only for dataarrays not datasets!")

    if da.ndim > 2:
        raise ValueError("Only 2D Data plotable! Select only one timestamp?")

    fontstyle_cfg = dict(size=10, color="k", rotation=0, ha="center")
    gridline_cfg = dict(
        draw_labels=["left", "bottom"], linewidth=0.7, color="dimgray", linestyle=":"
    )

    if isinstance(levels, str):
        start, stop, step = map(float, levels.split(":"))
        level_array = np.arange(start, stop + step, step)
    else:
        level_array = np.linspace(da.values.min(), da.values.max(), levels + 1)

    if "Time" in da.dims:
        time_str = pd.to_datetime(da.Time.values).strftime("%d-%m-%Y %H:%M") + " UTC"
    elif title:
        time_str = title
    else:
        time_str = "No Time"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(frameon=False, projection=ccrs.Robinson(central_longitude=0))

    ax.coastlines(zorder=3, linewidth=0.5, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

    N_COLORS = 256
    norm = mcolors.BoundaryNorm(level_array, ncolors=N_COLORS)
    im = ax.pcolormesh(
        da[lon],
        da[lat],
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    ax.gridlines(
        xlabel_style=fontstyle_cfg,
        ylabel_style=fontstyle_cfg,
        auto_update=False,
        **gridline_cfg,
    )

    CBAR_THICKNESS = 0.035
    pos = ax.get_position()
    cbar_y_pos = pos.y0 - 0.1
    cax = fig.add_axes([pos.x0, cbar_y_pos, pos.width, CBAR_THICKNESS])

    long_name = da.attrs.get("long_name", False)
    unit = da.attrs.get("units", False)
    if long_name and unit:
        cbar_label = f"{long_name} in {unit}"
    else:
        cbar_label = ""

    if not (da == 0).all(None):
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=level_array)
        cb.set_label(label=cbar_label)

        for t in cb.ax.get_xticklabels():
            t.set_fontfamily("Courier New")

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    fig.suptitle(time_str)

    return fig


def timeseries_station(
    df: xr.Dataset,
    y: str | List[str],
    x: str = "time",
    title: str | None = None,
    y_lim: tuple | None = None,
    figsize: tuple[float, float] = (10, 10 / 3),
):
    if isinstance(y, str):
        y = [y]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%x %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())

    CMAP_3 = ["#2E2E6A", "#872E49", "#56872E"]

    for station in df["station"]:
        for _y, c in zip(y, cycle(CMAP_3)):
            ax.plot(
                x,
                _y,
                label=df[_y].attrs["long_name"],
                data=df.sel(station=station),
                color=str(c),
            )

        for label in ax.get_xticklabels(which="major"):
            label.set(rotation=30, horizontalalignment="right")

        if len(y) == 1:
            attrs = df[_y].attrs
            ax.set_ylabel(f"{attrs['long_name']} in {attrs['unit']}")

        ax.axhline(y=0, c="black", ls="-", zorder=99)
        ax.grid(ls="--", c="grey", which="major")
        ax.set_ylim(y_lim)

        ax.legend(bbox_to_anchor=(0, 1), loc="lower left", fontsize="small")

    fig.suptitle(title)
    return fig
