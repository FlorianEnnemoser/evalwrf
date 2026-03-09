# 🌦️ evalwrf

```python
import evalwrf as ew
```

> _evalwrf_ is a toolkit for pre- and postprocessing WRF (Weather Research and Forecasting Model) data, featuring intuitive plotting utilities and a GeoSphere Austria API integration.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Status](https://img.shields.io/badge/status-active-brightgreen)]() -->

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Preprocessing](#preprocessing)
- [Postprocessing](#postprocessing)
- [Plotting](#plotting)
- [GeoSphere Austria API](#geosphere-austria-api)
- [Contributing](#contributing)
- [License](#license)

<!-- - [Configuration](#configuration)
- [Examples](#examples) -->

---

## Overview

**evalwrf** is a data science toolkit designed to simplify workflows around the [WRF model](https://www.mmm.ucar.edu/models/wrf/). It provides modular, scriptable utilities for:

- **Preprocessing** WRF input data (domain setup, namelist generation, initial/boundary conditions)
- **Postprocessing** WRF output (variable extraction, unit conversion, regridding)
- **Visualizing** meteorological fields with minimal boilerplate
- **Querying** observational and reanalysis data from the **GeoSphere Austria** (ZAMG) API for validation and forcing

---

## Features

| Category | Capabilities |
|---|---|
| **Preprocessing** | Namelist generation, WPS automation, Nudging Inputs |
| **Postprocessing** | NetCDF variable extraction, unit conversion, regridding, derived variables (e.g. Windpark Power Output) |
| **Plotting** | Spatial maps (Cartopy), time series, vertical profiles |
| **GeoSphere API** | Station data queries, gridded analysis fields, historical reanalysis access |
| **Config-driven** | TOML-based configuration for reproducible experiment setups |

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Using uv (recommended)

```bash
git clone https://github.com/FlorianEnnemoser/evalwrf.git
cd evalwrf

uv sync
```

### Using pip (WIP NOT WORKING)

```bash
pip install evalwrf
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `netCDF4` / `xarray` | Reading WRF NetCDF output |
| `colormaps` | Colormaps for plotting multidimensional data |
| `cartopy` | Geographic map projections |
| `matplotlib` | Plotting backend |
| `httpx` | GeoSphere API access |

---

## Quickstart

```python
import evalwrf as ew

...
```

---

## Preprocessing (WIP)

The preprocessing module automates WPS and namelist configuration for WRF runs.

### Generate a `namelist.input`

```python
import evalwrf as ew

...
```

---

## Postprocessing (WIP)

### Extract variables from `wrfout` files

```python
import evalwrf as ew

...
```

---

## Plotting (WIP)

### Spatial map

```python
import evalwrf as ew

...
```

### Time series

```python
import evalwrf as ew

...
```

---

## GeoSphere Austria API (WIP)

This toolkit includes a client for the [GeoSphere Austria Open Data API](https://data.hub.geosphere.at/), Austria's national weather and climate service.

### Station observations

```python
import evalwrf as ew

API_RESOURCE = "klima-v2-10min"
url = ew.load_url_from_resource("Datasets.json", API_RESOURCE)
ew.save_csv(
    url,
    filename="TS_Murau_NewYear2026.csv",
    params=dict(
        start="2025-12-30",
        end="2026-01-01",
        parameters=["TL", "RR", "cglo"],
        station_ids=15920,
        output_format="csv",
    ),
)
```

### Gridded analysis data (WIP)

```python
import evalwrf as ew
...
```

### Available datasets (WIP)

| Resource ID | Description | Resolution |
|---|---|---|
| `klima-v2-1h` | Station climate observations | 1h |
| `klima-v2-10min` | Station climate observations | 10 min |
| `inca-v2-1h-2d` | INCA gridded analysis (2D fields) | 1h, ~1 km |
| `snowgrid-v2-1d` | Snow analysis grid | daily |
| `spartacus-v2-1d` | Gridded climate dataset | daily |

> Full API documentation: [data.hub.geosphere.at](https://data.hub.geosphere.at/)

---

##  Configuration (WIP)

Project settings are managed via a `.toml` file:

```toml
[IO]
input = "..."
output = "..."

[NAMELIST]
domain = "alpine"
```

Override config programmatically:

```python
import evalwrf as ew

ew.load_config("my_config.toml")
ew.config.set("io.input", "abc")
```

---

## Examples (WIP)

Explore the [notebooks/](notebooks/) directory for end-to-end worked examples:

| Notebook | Description |
|---|---|
| `01_preprocessing.ipynb` | Domain setup, namelist generation, WPS run |
| `02_postprocessing.ipynb` | Variable extraction, derived vars, regridding |
| `03_plotting_examples.ipynb` | Maps, time series, Skew-T diagrams |
| `04_geosphere_api_demo.ipynb` | Querying GeoSphere data, WRF vs. obs comparison |

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
