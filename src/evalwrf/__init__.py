from .api import GFSAPI, ERA5API, ZAMGAPI
from  . import preprocess as pre
from  . import plotting as post
from importlib_metadata import version

__version__ = version("evalwrf")

__all__ = [GFSAPI, ERA5API, ZAMGAPI, pre, post]