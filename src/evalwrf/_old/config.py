import tomllib
from pathlib import Path

with open(Path(__file__).parent / "configuration.toml", "rb") as f:
    CONFIG = tomllib.load(f)