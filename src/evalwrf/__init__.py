from .api import GFSAPI, ERA5API, ZAMGAPI
from  . import preprocess as pre
from  . import plotting as post
from importlib_metadata import version

# import sys
# import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# make installable via:
# 1) go to folder were different python script is located
# 2) activate conda environment
# 3) cd into this folder (were pyproject.toml is located)
# 4) uv pip install -e .
# 5) ...
# 6) profit
# 7) meh, no typehints in other folder idk why
# 8) doch, geht, aber nur über «pip install -e . --config-settings editable_mode=strict»
# siehe https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#pip--setuptools


__version__ = version("evalwrf")

del version