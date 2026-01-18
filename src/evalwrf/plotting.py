from typing import List
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.dates as mdates
import warnings

# import colormaps as cmaps
from itertools import cycle

warnings.filterwarnings("ignore")

CMAP_3 = ["#2E2E6A", "#872E49", "#56872E"]


def _get_daynight_img(df: xr.Dataset, x: str, y: str | tuple, blackest_hour=1):
    if isinstance(y, tuple):
        minmax = np.linspace(*y)
    else:
        minmax = np.linspace(*df[y].agg(["min", "max"]).squeeze())
    normalized_hours = (df[x].dt.hour - blackest_hour) / 23
    hours = np.deg2rad(normalized_hours * 360)
    real_xx, fake_yy = np.meshgrid(df[x], minmax)
    xx, yy = np.meshgrid(hours, minmax)
    zz = np.cos(xx)
    return real_xx, yy, zz


def timeseries_station(
    df: xr.Dataset,
    y: str | List[str],
    x: str = "time",
    title: str | None = None,
    add_daynight: bool = False,
    y_lim: tuple | None = None,
    filename: str | None = None,
):
    fig, axs = plt.subplot_mosaic("a", figsize=(10, 10 / 3))

    if isinstance(y, str):
        y = [y]

    for ax in axs:
        axs[ax].xaxis.set_major_formatter(mdates.DateFormatter("%x %H:%M"))
        axs[ax].xaxis.set_major_locator(mdates.DayLocator())
        for station in df["station"]:
            for _y, c in zip(y, cycle(CMAP_3)):
                axs[ax].plot(x, _y, label=df[_y].attrs["long_name"], data=df.sel(station=station), color=str(c))

            for label in axs[ax].get_xticklabels(which="major"):
                label.set(rotation=30, horizontalalignment="right")

            axs[ax].axhline(y=0, c="black", ls="-", zorder=-99)
            axs[ax].grid(ls="--", c="grey", which="major")

            if add_daynight:
                if y_lim:
                    y = y_lim
                axs[ax].pcolormesh(
                    *_get_daynight_img(df, x, y), clim=(0, 1), cmap="grey_r", alpha=0.5, zorder=-999, shading="nearest"
                )

            axs[ax].set_ylim(y_lim)
            axs[ax].legend(bbox_to_anchor=(0, 1), loc="lower left", fontsize="small")

    fig.suptitle(title)
    if filename:
        fig.savefig(filename, transparent=True, dpi=400, bbox_inches="tight")
    return fig, axs
