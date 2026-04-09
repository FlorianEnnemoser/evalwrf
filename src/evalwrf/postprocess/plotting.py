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
