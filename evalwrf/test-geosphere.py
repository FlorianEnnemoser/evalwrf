from dataclasses import dataclass
from typing import Literal, Tuple
import requests
import pandas as pd
import numpy as np

@dataclass
class ZAMGAPI:
    """
    https://dataset.api.hub.geosphere.at/v1/docs/getting-started.html
    https://dataset.api.hub.geosphere.at/v1/docs/user-guide/resource.html#resources
    """
    type : Literal["grid","timeseries","station"]
    mode : Literal["historical","current","forecast"]
    resource : str

    def __post_init__(self):
        self.url = "https://dataset.api.hub.geosphere.at"
        self.version = "v1"
        self.dataset_url = "/".join([self.url,self.version,self.type,self.mode,self.resource])
        self.output_format = "csv"
        self.parameters = []
        self.time = ""
        self.stationslist = []
        self.request_url = ""

    def metadata(self) -> str:
        self.request_url = "/".join([self.dataset_url,"metadata"])
        self.output_format = "json"
        return self.request_url

    def parameter(self, *p : str) -> None:
        self.parameters.extend(p)
        return None
    
    def timeframe(self, start : str = "2024-12-01 00:00:00", end : str = "2024-12-02 00:00:00") -> None:
        dt_start = pd.to_datetime(start).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        dt_end = pd.to_datetime(end).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        self.time = f"start={dt_start}&end={dt_end}&"
        return None

    def stations(self, *ids : int) -> None:
        self.stationslist.extend(ids)
        return None

    def compile(self) -> str:
        self.parameter_string = "".join([f"parameters={para}&" for para in self.parameters])
        self.stations_string = "".join([f"station_ids={station}&" for station in self.stationslist])
        self.output_format = "csv"
        self.request_url = self.dataset_url + "?" + \
                           self.parameter_string  + \
                           self.time + \
                           self.stations_string + \
                           f"output_format={self.output_format}"
        return self.request_url
    
    def download(self,filename: str, schwimmflügerl : bool = True):
        if schwimmflügerl:
            really_download = input("Wirklich runterladen? [y/n] ")
            if really_download != "y":
                raise ValueError("stopped downloading.")
        
        self.full_filename = f"{filename.split('.')[0]}.{self.output_format}"
        response = requests.get(self.request_url)
        print(response.headers)
        with open(self.full_filename, "wb") as f:
            f.write(response.content)
        print("finished downloading")
        
    def load_metadata(self, filename : str = "") -> Tuple[pd.DataFrame,pd.DataFrame]:
        if not filename:
            filename = self.full_filename
        df = pd.read_json(filename ,orient="index").T.drop(columns=["type","mode","title","frequency","response_formats","start_time","end_time","id_type"])
        self.df_stations = pd.json_normalize(df["stations"])
        self.df_parameters = pd.json_normalize(df["parameters"]).dropna(how="all",axis="index").dropna(how="all",axis="columns")
        return self.df_stations, self.df_parameters

    def plot(self):
        NotImplementedError("...")

#################
api = ZAMGAPI(type="station",mode="historical",resource="klima-v2-10min") #klima-v2-1h

api.metadata()
# api.download("klima-v2-10min_METADATA.json")
api.load_metadata("klima-v2-10min_METADATA.json")
available_stations = api.df_stations.query("state == 'Steiermark' & is_active == True & type != 'COMBINED' & has_global_radiation == True")
print(available_stations)
print(api.df_parameters.query("~name.str.contains('flag')"))


api.parameter("ff","dd","tl","rf")
api.timeframe(start="2024-06-01 00:00:00", end="2024-06-10 18:00:00")
api.stations(*available_stations["id"])
req = api.compile()

# api.download("Steiermark_2024-06-01_2024-06-10_FDDA.csv")


import plotly.express as px
from PIL import Image

def update_plot_layout(
    fig, 
    image_path=None, 
    opacity=0.5, 
    background_color='white', 
    secondary_background_color='lightgray',
    font_family='Arial',
    font_size=12
):
    """
    Update Plotly Express figure layout with background image and colors
    
    :param fig: Plotly Express figure
    :param image_path: Path to background image
    :param opacity: Image opacity
    :param background_color: Main background color
    :param secondary_background_color: Plot area background color
    :param font_family: Font family to use
    :param font_size: Base font size    
    :return: Updated figure
    """
    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=secondary_background_color,
        font=dict(
            family=font_family,
            size=font_size
        )        
    )
    
    if image_path:

        fig.update_layout(
            images=[{
                'source': Image.open(image_path),
                'xref': 'paper',
                'yref': 'paper',
                'x': 0,
                'y': 1,
                'sizex': 0.2,
                'sizey': 0.2,
                'opacity': opacity,
                'layer': 'below'
            }]
        )
    
    return fig

df = pd.read_csv("Steiermark_2024-06-01_2024-06-10_FDDA.csv")
df["time"] = pd.to_datetime(df["time"])
stationdata = api.df_stations.set_index("id")
df["name"] = df["station"].map(stationdata["name"])
df["altitude"] = df["station"].map(stationdata["altitude"])
df["lat"] = df["station"].map(stationdata["lat"])
df["lon"] = df["station"].map(stationdata["lon"])

def saturation_water_vapor_pressure(T : float) -> float:
    """
    https://en.wikipedia.org/wiki/Vapour_pressure_of_water > August-Roche-Magnus 
    """
    return 0.61094*np.exp(17.625*T / (T+243.04)) * 1e3 # (Pa)

def saturation_vapor_density(T : float, molecular_weight : float = 18.) -> float:
    """Ideal Gas law with 18 as molecular weight for water (in g/mol)"""
    R = 8.31 # J/molK
    VP = saturation_water_vapor_pressure(T)
    return VP / (R*(T+273.15)) * molecular_weight

def u_from_vector(magnitude : float,degree : float):
    return np.cos(np.deg2rad(degree)) * magnitude

def v_from_vector(magnitude : float,degree : float):
    return np.sin(np.deg2rad(degree)) * magnitude

df["S"] = saturation_vapor_density(df["tl"])
df["Q"] = df["S"] * df["rf"] / 100

df["u"] = u_from_vector(df["ff"],df["dd"])
df["v"] = v_from_vector(df["ff"],df["dd"])

print(df)

def plot_wind_vectors(df,lon_min=None, lon_max=None, lat_min=None, lat_max=None) -> None:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Create figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(12, 8), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    
    # Determine plot boundaries
    if lon_min is None: lon_min = df['lon'].min() - 1
    if lon_max is None: lon_max = df['lon'].max() + 1
    if lat_min is None: lat_min = df['lat'].min() - 1
    if lat_max is None: lat_max = df['lat'].max() + 1
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot wind vectors
    q = ax.quiver(df['lon'], df['lat'], df['u'], df['v'], 
                  transform=ccrs.PlateCarree(),
                  color='red', 
                  scale=50,  # Adjust scale for vector length
                  width=0.002)
    
    # Add quiver key for reference
    ax.quiverkey(q, 0.9, 0.9, 10, r'10 m/s', 
                 labelpos='E', coordinates='figure')
    
    # Set title and labels
    plt.title('Wind Vectors')
    ax.set_global()
    
    # Add gridlines
    ax.gridlines(draw_labels=True, linewidth=1, 
                 color='gray', alpha=0.5, linestyle='--')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    return None

# fig = px.scatter(df,x="time",y="ff",color="name")
# fig.update_traces(marker=dict(line=dict(width=1,color='DarkSlateGrey'),size=10),selector=dict(mode='markers'))
# fig.update_traces(mode='lines+markers')
# fig = update_plot_layout(fig, 'Picture2.png')
# fig.show()