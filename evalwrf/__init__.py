from .api import GFSAPI, ERA5API, ZAMGAPI
from  . import preprocess as pre
from  . import plotting as post
from .config import CONFIG
from importlib_metadata import version


__version__ = version("evalwrf")

del version