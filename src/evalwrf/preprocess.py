from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import cartopy.io.shapereader as shpreader
import xarray as xr
import pandas as pd
from .mathfuncs import _meter2lat, _meter2lon
from .config import CONFIG

def parse_namelist_wps(file_path):
    """
    Parse a namelist.wps file to extract domain boundaries and grid information.
    
    Parameters:
        file_path (str): Path to the namelist.wps file.
    
    Returns:
        dict: A dictionary containing the domain information.
    """

    domain = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('!')[0].strip()
            
            if line.startswith("/") or len(line) == 0:
                continue

            if not line.startswith("&"):
                arr = [v.strip().split(",") for v in line.split("=")]
                name = arr[0][0]
                values = arr[1:][0]
                values = [v.strip().strip("'").strip('"') for v in values if v]
                domain[name] = values

    return domain

def compute_grid(domain):
    grids = []
    
    for i in range(int(domain["max_dom"][0])):    
        parent_grid_ratio = int(domain["parent_grid_ratio"][i])
        if i <= 1:
            dx = int(domain["dx"][0]) / parent_grid_ratio
            dy = int(domain["dy"][0]) / parent_grid_ratio
        else:
            dx = int(domain["dx"][0]) / (parent_grid_ratio * (i+1))
            dy = int(domain["dy"][0]) / (parent_grid_ratio * (i+1))

        ref_lat  = float(domain["ref_lat"][0])
        ref_lon  = float(domain["ref_lon"][0])
        e_we = int(domain["e_we"][i])
        e_sn = int(domain["e_sn"][i])

        if (e_we - 1) % parent_grid_ratio != 0:
            min_n = (e_we - 1) // parent_grid_ratio
            suggested_e_we = [(n * parent_grid_ratio + 1) for n in range(min_n, min_n + 5)]
            raise ValueError(f"Domain {i+1}: e_we={e_we} does not satisfy the nesting criterion. Try: {suggested_e_we}")
        if (e_sn - 1) % parent_grid_ratio != 0:
            min_n = (e_sn - 1) // parent_grid_ratio
            suggested_e_sn = [(n * parent_grid_ratio + 1) for n in range(min_n, min_n + 5)]            
            raise ValueError(f"Domain {i+1}: e_sn={e_sn} does not satisfy the nesting criterion. Try: {suggested_e_sn}")

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

            width = _meter2lon((e_we - 1) * dx,start_lat)
            height = _meter2lat((e_sn - 1) * dy)

            center_lat = start_lat + height/2. 
            center_lon = start_lon + width/2. 

        grid_spacing_lon = _meter2lon(dx,center_lat)
        grid_spacing_lat = _meter2lat(dy)
        
        lons = center_lon + (np.arange(e_we) - e_we / 2) * grid_spacing_lon
        lats = center_lat + (np.arange(e_sn) - e_sn / 2) * grid_spacing_lat
        grids.append({"lons": lons, "lats": lats, "center_lat": center_lat, "center_lon": center_lon, "dx":dx,"dy":dy})
    return grids

def plot_grids(domain, grids, plot_grid : bool = True):
    """
    Plot the WRF grids using Cartopy.
    """

    base_config = dict(linewidth=1.5, transform=ccrs.PlateCarree())
    grid_config = dict(color="blue", linewidth=0.5, transform=ccrs.PlateCarree(), alpha=0.5)

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.LambertConformal(central_longitude=grids[0]["center_lon"])})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m',
                                        edgecolor='blue', facecolor='none')
    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='lightblue', alpha=0.5)

    dx = int(domain["dx"][0]) /  1000

    for i, grid  in enumerate(grids):
        lons, lats = grid["lons"], grid["lats"]

        if plot_grid:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            ax.plot(lon_grid, lat_grid, **grid_config)
            ax.plot(lon_grid.T, lat_grid.T, **grid_config)
    
        color = plt.get_cmap(name="tab10")(i/len(grids))
        ax.plot(lons, [max(lat_grid[:,0])] * len(lons),  **base_config, color=color)
        ax.plot(lons, [min(lat_grid[:,0])] * len(lons),  **base_config, color=color)
        ax.plot([min(lon_grid[0])] * len(lats), lats,    **base_config, color=color)
        ax.plot([max(lon_grid[0])] * len(lats), lats,    **base_config, color=color, label=f"Domain {i+1}")

        dx /= int(domain["parent_grid_ratio"][i])
        print(f"Domain {i+1}:",find_closest_city(grid["center_lat"],grid["center_lon"]), f" Gridsize: {dx:.1f} km")
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    ax.legend(loc="upper left")

    title_string = "Center of domains:\n"+"".join([f'Domain {i+1}: {d["center_lat"]:.1f}° {d["center_lon"]:.1f}°\n' for i,d in enumerate(grids)])
    plt.title(title_string)    
    plt.savefig("grid.jpg")

def find_closest_city(lat : float, lon : float, pop_size : int = 50_000) -> dict:
    shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='populated_places')
    cities = shpreader.Reader(shpfilename).records()
    
    def _distance_function(city):
        return np.sqrt((lat - city.geometry.y) ** 2 + (lon - city.geometry.x) ** 2)
    
    large_cities = (city for city in cities if city.attributes.get("POP_MAX", 0) > pop_size)
    closest_city = min(large_cities, key=_distance_function)

    city_lat = closest_city.geometry.y
    city_lon = closest_city.geometry.x
    return {"name":closest_city.attributes["NAME"],"lat":city_lat,"lon":city_lon}

def from_namelist(namelist_wps_path : str = "namelist.wps") -> None:
    domain_info = parse_namelist_wps(namelist_wps_path)
    grids = compute_grid(domain_info)
    plot_grids(domain_info,grids,plot_grid=True)

def wrf_to_xr(ds : str, ds_input : str) -> xr.Dataset:
    ds_data = xr.open_dataset(ds)
    ds_coords = xr.open_dataset(ds_input)

    time_ = [t.values.astype(str) for t in ds_data["Times"]]
    time_ = pd.to_datetime([str(i).replace("_"," ") for i in time_])
    ds_wrf_timedim = ds_data.rename({"Time":"time"})
    ds_wrf_timecoord = ds_wrf_timedim.assign(time=time_)
    ds_wrf_dropped_Times = ds_wrf_timecoord.drop_vars('Times')

    ds_wrf_w_latlon = ds_wrf_dropped_Times.assign_coords(
    lat=ds_coords.coords['XLAT'].squeeze('Time'),
    lon=ds_coords.coords['XLONG'].squeeze('Time'),
    landmask=ds_coords.LANDMASK.squeeze('Time'),
    lakemask=ds_coords.LAKEMASK.squeeze('Time'))
    ds_wrf_rename_latlon = ds_wrf_w_latlon.rename({'south_north':'y', 'west_east':'x'})
    ds_wrf_cf = ds_wrf_rename_latlon.drop_vars(['XLAT', 'XLONG', 'XTIME','XLAT_U','XLONG_U','XLAT_V','XLONG_V'])  #['XLAT', 'XLONG']
    return ds_wrf_cf

def read_windturbine_file(filename='windturbines.tbl') -> pd.DataFrame:
    """
    https://github.com/wrf-model/WRF/blob/master/doc/README.windturbine

    ### Interpretation of ct:
        * Tells you how much of the wind’s momentum is extracted by the turbine.
        * A higher ct means more force on the turbine, which can increase wake effects and loading on the structure.
    
    """

    wind_speed = []
    thrust_coefficient = []
    power_production = []

    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            if i == 0:
                max_lines = int(line.strip())

            if i > max_lines+1:
                print(f"more lines than max lines! exiting file at entry {max_lines}!")
                break
            
            parts = line.strip().split()
            parts = [float(p) for p in parts]

            if i == 1: 
                wind_turbine = dict(zip(CONFIG["WINDTURBINE"]["ATTRIBUTE_COLS"],parts))

            if i > 1:
                wind_speed.append(parts[0])
                thrust_coefficient.append(parts[1])
                power_production.append(parts[2])
            

        turbines = pd.DataFrame({"wind_speed_ms-1":wind_speed,"thrust_coeff":thrust_coefficient,"power_produktion_kW":power_production})
        turbines.attrs = wind_turbine
        return turbines
    
def to_windturbine(df : pd.DataFrame, filename="wind-turbine-XXX.tbl") -> None:
    if not filename.split(".")[-1].endswith("tbl"):
        raise ValueError("File must end with `.tbl`!")
    
    _req_columns = ["wind_speed_ms-1","thrust_coeff","power_produktion_kW"]
    if not df.columns.isin(_req_columns).all() or len(df.columns) != len(_req_columns):
        raise ValueError(f"Provide required columns {_req_columns} and remove potentially unused columns!")

    _req_header_info = CONFIG["WINDTURBINE"]["ATTRIBUTE_COLS"]
    header = [str(df.attrs.get(c)) for c in _req_header_info]
    if len(header) != len(_req_header_info):
        raise ValueError(f"Missing data in dataframe attrs! Provide all of: {_req_header_info}")

    if Path(filename).is_file():
        raise ValueError("File already exists!")
    
    df.to_csv(filename,sep=" ",index=False,header=False)

    with open(filename, 'r') as f:
        content = f.read()

    max_lines_line = str(df.index.size)
    turbine_attributes_line = " ".join(header)
    header_info = [max_lines_line,turbine_attributes_line]
    with open(filename, 'w') as f:
        for line in header_info:
            f.write(line + '\n')
        f.write(content)
    return None

def create_windfarm_locations(df : pd.DataFrame, filename : str = "windturbines.txt") -> None:
    """
    Example `pd.DataFrame`:
    >>> df = pd.DataFrame(
        [
            {"Latitude":convert("46-52-07.2N"),"Longitude":convert("15-00-32.8E"),"Index Value":1},
            {"Latitude":convert("46-52-21.1N"),"Longitude":convert("15-00-32.9E"),"Index Value":1},
            {"Latitude":convert("46-52-29.1N"),"Longitude":convert("15-00-26.8E"),"Index Value":1},
            {"Latitude":convert("46-52-40.2N"),"Longitude":convert("15-00-25.3E"),"Index Value":1},
            {"Latitude":convert("46-52-48.9N"),"Longitude":convert("15-00-15.4E"),"Index Value":1},
            {"Latitude":convert("46-52-57.4N"),"Longitude":convert("15-00-06.4E"),"Index Value":1},
            {"Latitude":convert("46-53-06.1N"),"Longitude":convert("14-59-56.7E"),"Index Value":2},
            {"Latitude":convert("46-53-14.3N"),"Longitude":convert("14-59-47.5E"),"Index Value":2},
         ]
    )    
    >>> ew.pre.reate_windfarm_locations(df, filename="windturbines_XXX.txt")
    """

    if filename != "windturbines.txt":
        print("WARNING: WRF always needs the windfarm locations file to be named 'windturbines.txt'!")

    _req_columns = CONFIG["WINDTURBINE"]["LOCATION_COLS"]
    if not df.columns.isin(_req_columns).all() or len(df.columns) != len(_req_columns):
        raise ValueError(f"Provide required columns {_req_columns} and remove potentially unused columns!")

    df.to_csv(filename,sep=" ",index=False,header=False)
    return None

