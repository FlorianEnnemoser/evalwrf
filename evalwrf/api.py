from pathlib import Path
import cdsapi
from itertools import product
from typing import List, Literal, Optional, Self, Tuple, Union
import requests
import time as t
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from . import mathfuncs as mf

@dataclass
class BaseAPI:
    daterange : str
    grid_size : Literal["1p00","0p50","0p25"]
    savefolder : str = field(init=True,default="download_wrf",repr=True)

    base_url : str = field(init=True,default="",repr=True)

    def _download(self, url : str, file : str, max_sleep : int = 2) -> None:
        self.savefolder : Path = Path(self.savefolder)
        if not self.savefolder.is_dir():
            self.savefolder.mkdir(parents=True, exist_ok=True)
        response = requests.get(self.base_url + url)
        
        if response.status_code != 200:
            raise ValueError(f"Statuscode not 200, can't download file. Requested URL:\n{self.base_url + url}")

        with open(self.savefolder / file, "wb") as f:
            f.write(response.content)
        
        time_to_sleep = np.random.uniform(1,max_sleep) #seconds
        t.sleep(time_to_sleep)

    def _filename(self,prefix : str,*args) -> str:
        return f"{prefix}_" + '_'.join([str(a) for a in args]) + ".grib2"

    @property
    def today(self):
        return pd.Timestamp.now()

    
    @property
    def date_range(self) -> pd.DatetimeIndex:
        splitted_range = self.daterange.split("|",maxsplit=1)
        return pd.date_range(*splitted_range,freq="D")

@dataclass
class GFSAPI(BaseAPI):

    def __post_init__(self):
        self.noaa_start_date = self.today - pd.Timedelta(days=9)
        self.is_forecast = (self.today < self.date_range).any()
        self.valid_current = (self.date_range > self.noaa_start_date).all()
        self.date_format = "%Y%m%d" #YYYYMMDD
        self.forecast_times = ["00", "06", "12", "18"] 
        self.file_date = self._get_date_range()
        self.file_urls = []
        self.filenames = []

        if self.valid_current or self.is_forecast:
            self.base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl"
            self.set_noaa = True
        else:
            self.base_url = "https://data.rda.ucar.edu/d083002/grib2/"
            self.set_noaa = False
        
        print(f"Using saved data from {self.base_url}")
        
        if self.grid_size != "1p00":
            raise NotImplementedError("Only 1° Gridsize implemented!")

        if len(self.file_date) > 20:
            raise ValueError("Currently max. dates is 20.")

    def _get_date_range(self) -> list | tuple:
        condition_past = self.date_range < (self.today - pd.Timedelta(days=1))
        past_times = self.date_range[condition_past].strftime(self.date_format)
        forecast_times = self.date_range[~condition_past]

        if forecast_times.empty:
            return past_times
        else:
            return past_times, forecast_times

    def _factory_noaa(self, bottom : int = 10, top : int= 70, left : int= 0, right: int= 360) -> None:
        if any(np.sign([left,right]) == -1):
            raise ValueError("Only positive values for longitude allowed! Like 30°E to 280°E")

        if self.is_forecast:
            self._factory_forecast(bottom,top,left,right)
        else:
            self._factory_non_forecast(self.file_date,bottom,top,left,right)
        # for day,hour in product(self.file_date,self.forecast_times):
        #     single_url = (
        #         f"{self.base_url}?dir=%2Fgfs.{day}%2F{hour}%2Fatmos"
        #         f"&file=gfs.t{hour}z.pgrb2.1p00.f000&all_var=on&all_lev=on"
        #         f"&subregion=&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
        #     )
        #     self.file_urls.append(single_url)
        #     self.filenames.append(self._filename("GFS",day,hour,top,bottom,left,right))
        return None

    def _factory_forecast(self, bottom : int = 10, top : int= 70, left : int= 0, right: int= 360) -> None:
        past_dates, forecast_dates = self.file_date

        n_forecast_days = len(forecast_dates)
        six_hour_interval = np.arange(0,n_forecast_days*24,6)

        self._factory_non_forecast(past_dates,bottom,top,left,right)

        for future_hour in six_hour_interval:
            single_url = (
                f"?dir=%2Fgfs.{forecast_dates[0].strftime(self.date_format)}%2F00%2Fatmos"
                f"&file=gfs.t00z.pgrb2.1p00.f{future_hour:03}&all_var=on&all_lev=on"
                f"&subregion=&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
            )
            date = forecast_dates[0] + pd.Timedelta(hours=future_hour)
            hour = date.hour

            self.file_urls.append(single_url)
            self.filenames.append(self._filename("GFS",date.strftime(self.date_format),f"{hour:02}",top,bottom,left,right))
        return None

    def _factory_non_forecast(self, dates : list, bottom : int = 10, top : int= 70, left : int= 0, right: int= 360) -> None:
        for day,hour in product(dates,self.forecast_times):
            single_url = (
                f"?dir=%2Fgfs.{day}%2F{hour}%2Fatmos"
                f"&file=gfs.t{hour}z.pgrb2.1p00.f000&all_var=on&all_lev=on"
                f"&subregion=&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
            )
            self.file_urls.append(single_url)
            self.filenames.append(self._filename("GFS",day,hour,top,bottom,left,right))
        return None        

    def _factory_ncar(self) -> None:
        file_year = self.date_range.year
        file_yearmonth = self.date_range.strftime("%Y.%m")
        
        for (day,year,ym), hour in product(zip(self.file_date,file_year,file_yearmonth), self.forecast_times):
            single_url = f"{year}/{ym}/fnl_{day}_{hour}_00.grib2"
            
            self.file_urls.append(single_url)
            self.filenames.append(self._filename("GFS",day,hour))
        return None

    def download(self, bottom : int = 10, top : int= 70, left : int= 0, right: int= 360, as_test : bool = True) -> Self:
        if self.set_noaa:
            self._factory_noaa(bottom,top,left,right)
        else:
            self._factory_ncar()
        
        print(f"Downloading to: {self.savefolder}")
        for url, file in zip(self.file_urls, self.filenames):
            print(file)
            if not as_test:
                self._download(url,file)
        return self

@dataclass
class ERA5API(BaseAPI):

    def __post_init__(self):
        self.pressure_dataset = "reanalysis-era5-pressure-levels" 
        self.surface_dataset = "reanalysis-era5-single-levels"
        self._pressure = {
            "product_type": ["reanalysis"],
            "variable": [
                    "geopotential",
                    "relative_humidity",
                    "specific_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind"
                ],
                # "year": ["2022"],
                # "month": ["03"],
                # "day": ["13", "14", "15"],
                "time": [
                    "00:00", "03:00", "06:00",
                    "09:00", "12:00", "15:00",
                    "18:00", "21:00"
                ],
                "pressure_level": [
                    "1", "2", "3",
                    "5", "7", "10",
                    "20", "30", "50",
                    "70", "100", "125",
                    "150", "175", "200",
                    "225", "250", "300",
                    "350", "400", "450",
                    "500", "550", "600",
                    "650", "700", "750",
                    "775", "800", "825",
                    "850", "875", "900",
                    "925", "950", "975",
                    "1000"
                ],
                "data_format": "grib",
                "download_format": "unarchived",
                # "area": [45, 35, 20, 70] #North, West, South, East                        
        }
        self._surface = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "sea_surface_temperature",
                "surface_pressure",
                "total_precipitation",
                "skin_temperature",
                "surface_latent_heat_flux",
                "top_net_solar_radiation_clear_sky",
                "snow_depth",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "soil_type",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "leaf_area_index_high_vegetation",
                "geopotential",
                "land_sea_mask",
                "sea_ice_cover"
            ],
            # "year": ["2022"],
            # "month": ["03"],
            # 'day': [
                # '13', '14', '15',
            # ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
            "data_format": "grib",
            "download_format": "unarchived",
            # 'area': [
                # 45, 35, 20, 70, #North, West, South, East
            # ],
        }

    @property
    def pressure(self) -> dict:
        return self._pressure
        
    @property
    def surface(self) -> dict:
        return self._surface

    def _update_area(self, bottom : int , top : int, left : int, right: int) -> Self:
        area = [top, left, bottom, right]
        self._surface.update({"area":area})
        self._pressure.update({"area":area})
        return self

    def _update_time(self, year : list, month : list, day : list) -> Self:
        self._surface.update({"year":year.astype(str).tolist()})
        self._surface.update({"month":month.astype(str).tolist()})
        self._surface.update({"day":day.astype(str).tolist()})

        self._pressure.update({"year":year.astype(str).tolist()})
        self._pressure.update({"month":month.astype(str).tolist()})
        self._pressure.update({"day":day.astype(str).tolist()})
        return self

    def _update_grid(self) -> Self:
        grid_translator = {"1p00":"1.0/1.0","0p50":"0.5/0.5","0p25":"0.25/0.25"}
        used_grid = grid_translator.get(self.grid_size)
        self._surface.update({"grid":used_grid})
        self._pressure.update({"grid":used_grid})
        return self

    def download(self, bottom : int = 10, top : int= 70, left : int= 0, right: int= 360, as_test : bool = True, as_netcdf : bool = False):
        """
        This for Info which variables: 
        https://www.youtube.com/watch?v=M91ec7EdCic
        
        This for how to install api key:
        https://cds.climate.copernicus.eu/how-to-api
        API Key is located under `C:\\User\\USERNAME\\.cdsapirc`
        If this file (`.cdsapirc`) does not exist, create it!
        """
        self._update_area(bottom,top,left,right)
        self._update_time(np.unique(self.date_range.year),
                          np.unique(self.date_range.month),
                          np.unique(self.date_range.day))
        self._update_grid()

        client = cdsapi.Client()

        if as_netcdf:
            file_ending = "nc"
            self._surface["data_format"] = "netcdf"
            self._pressure["data_format"]= "netcdf"
        else:
            file_ending = self._surface["data_format"]
        

        for dataset, request in zip([self.surface_dataset,self.pressure_dataset],[self.surface,self.pressure]):
            filename = f'{dataset}.{file_ending}'
            print(filename)
            if not as_test:
                client.retrieve(dataset, request,filename)
        return None
    
@dataclass
class ZAMGAPI:
    """
    https://dataset.api.hub.geosphere.at/v1/docs/getting-started.html
    https://dataset.api.hub.geosphere.at/v1/docs/user-guide/resource.html#resources

    Example:
    ---------
    >>> api = ew.ZAMGAPI(type="station",mode="historical",resource="klima-v2-10min")
    >>> api.metadata()
    >>> api.download("klima-v2-10min_METADATA.json")

    >>> api.load_metadata("klima-v2-10min_METADATA.json")
    >>> available_stations = api.get_available_stations("Steiermark") 

    >>> api.parameter(*api.wrf_fdda_properties)
    >>> api.timeframe(start="2024-06-01 00:00:00", end="2024-06-10 18:00:00")
    >>> api.stations(*available_stations["id"])
    >>> api.compile()
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
        self.timestamp_format = '%Y-%m-%dT%H:%M:%S.000Z'

    def metadata(self) -> str:
        """sets request_url to metadata for given resource"""
        self.request_url = "/".join([self.dataset_url,"metadata"])
        self.output_format = "json"
        return self.request_url

    def parameter(self, *p : str) -> None:
        self.parameters.extend(p)
        return None
    
    def timeframe(self, start : str = "2024-12-01 00:00:00", end : str = "2024-12-02 00:00:00") -> None:
        dt_start = pd.to_datetime(start).strftime(self.timestamp_format)
        dt_end = pd.to_datetime(end).strftime(self.timestamp_format)
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
            really_download = input("Start download? [y/n] ")
            if really_download != "y":
                raise ValueError("stopped downloading.")
        
        self.full_filename = f"{filename.split('.')[0]}.{self.output_format}"
        response = requests.get(self.request_url)
        print(response.headers)
        with open(self.full_filename, "wb") as f:
            f.write(response.content)
        print("finished downloading")
        
    def load_metadata(self, filename : str = "") -> Tuple[pd.DataFrame,pd.DataFrame]:
        """Returns `stations` dataframe and `parameters` dataframe as a tuple"""
        if not filename:
            filename = self.full_filename
        standard_columns = ["type","mode","title","frequency","response_formats","start_time","end_time","id_type"]
        df = pd.read_json(filename ,orient="index").T.drop(columns=standard_columns)
        self.df_stations = pd.json_normalize(df["stations"]).set_index("id")
        self.df_parameters = pd.json_normalize(df["parameters"]).dropna(how="all",axis="index").dropna(how="all",axis="columns")
        return self.df_stations, self.df_parameters

    def get_available_stations(
        self, 
        states: Union[str, List[str]],
        is_active_only : bool = True,
        expression: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query stations based on state(s) with optional additional filtering.
        
        Parameters:
        -----------
        states : str or List[str]
            Single state or list of states to query
        expression : str, optional
            Additional pandas query expression for filtering
        
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame of stations
        
        Raises:
        -------
        ValueError
            If any provided state is not in the available states
        """
        if isinstance(states, str):
            states = [states]
        
        invalid_states = [s not in self.states for s in states]
        if any(invalid_states):
            raise ValueError(f"Invalid states: {invalid_states}")
        
        active = "& is_active == True" if is_active_only else ""
        base_query = f"state in {tuple(states)} {active} & type != 'COMBINED'"
        
        if expression:
            full_query = f"{base_query} & {expression}"
        else:
            full_query = base_query

        return self.df_stations.query(full_query)
    
    def _converter(self, filename : str) -> pd.DataFrame:
        zamg_mapping = {
            "ff":"Wind", #wind
            "dd":"Wind direction", #windrichtung
            "tl":"2m Temperature", #lufttemp
            "rf":"Relative Humidity", #relative feuchte
            "p":"Surface Pressure",  #luftdruck
            "pred":"SLP Pressure",
            "rr":"Precipitation (mm)"  #regen
        }
        MISSING_CHAR = -888888
        df = pd.read_csv(filename)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time",ignore_index=True)
        df = df.rename(columns={"station":"ID"})

        df["Name"] = df["ID"].map(self.df_stations["name"])
        df['Name'] = df['Name'].str.replace('ä', 'ae')
        df['Name'] = df['Name'].str.replace('ö', 'oe')
        df['Name'] = df['Name'].str.replace('ü', 'ue')
        df['Name'] = df['Name'].str.replace('Ä', 'Ae')
        df['Name'] = df['Name'].str.replace('Ö', 'Oe')
        df['Name'] = df['Name'].str.replace('Ü', 'Ue')

        df["Elevation"] = df["ID"].map(self.df_stations["altitude"])
        df["Latitude"] = df["ID"].map(self.df_stations["lat"])
        df["Longitude"] = df["ID"].map(self.df_stations["lon"])

        df["Saturation_Vapor_Density"] = mf.saturation_vapor_density(df["tl"])
        df["Humidity"] = df["Saturation_Vapor_Density"] * df["rf"] / 100

        df["t2m_K"] = df["tl"] + 273.15
        df["u"] = mf.u_from_vector(df["ff"],df["dd"])
        df["v"] = mf.v_from_vector(df["ff"],df["dd"])
        df["pred"] = np.where(df["pred"].isna(),mf.slp_from_station_pressure(df["p"],df["Elevation"]+2,df["t2m_K"]),df["pred"]) *100
        df["p"] *= 100
        df = df.rename(columns=zamg_mapping)

        df["FM-Code"] = "FM-12" # FM-35 oder FM-12 idk https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html#FM
        df["Source"] = "Geosphere Austria"
        df["Sequence number"] = df.index[::-1]
        df["Bogus"] = "F"
        df["ref_pres"] = MISSING_CHAR
        # df["Unix Time"] = df["time"].astype(int) / 10**9
        # df["Julian day"] = df["time"].apply(lambda t: t.to_julian_date()).astype(int)
        df["Date string"] = df["time"].dt.strftime('%Y%m%d%H%M%S')
        return df

    def to_obsdomain(self,filename : str, domain_number : int, output_filepath = "OBS_DOMAIN", is_sound=False):
        """
        Converts a Pandas DataFrame to the OBS_DOMAIN format for WRF ObsNudging.

        This function takes a Pandas DataFrame, presumably created from observational data,
        and writes it to a file in the OBS_DOMAIN format, which is used by the Weather
        Research and Forecasting (WRF) model's Observation Nudging (ObsNudging) system.
        The OBS_DOMAIN format is a specific text-based format detailed in the WRF
        ObsNudging Guide (see references below).

        The function performs the following steps:
        1. **Data Loading and Preprocessing:**
           - Calls the internal `_converter` method (``self._converter(filename)``) to load data
             from the specified `filename` into a Pandas DataFrame.  It is assumed that
             `_converter` returns a DataFrame with the necessary columns for OBS_DOMAIN format.
           - Converts the 'ID' column in the DataFrame to string type.
           - Sorts the DataFrame chronologically by the 'time' column to ensure proper time order
             in the output file.

        2. **OBS_DOMAIN File Writing:**
           - Determines observation type parameters:
             - `is_sound_char` is set to "F" (likely representing surface observation).
             - `QC_char` (Quality Control character) is set to 0.
             - `meas_count` (measurement count) is set to 1 for surface observations.
               (Note: Sounding implementation is not yet available and will raise a `NotImplementedError`).
           - Constructs the output filename using `output_filepath` and `domain_number`.
           - Opens the output file for writing.
           - Iterates through each row of the DataFrame and writes the data in the
             OBS_DOMAIN format, including:
             - Date and time string.
             - Latitude and Longitude.
             - Observation ID and Name.
             - FM-Code, Source, Elevation, is_sound_char, Bogus flag, and meas_count.
             - Surface data line with pressure, temperature, wind components (u, v),
               relative humidity, surface pressure, and precipitation, along with QC flags
               for each variable.  The variables are assumed to be in specific units
               (e.g., Pressure in hPa, Temperature in Kelvin, wind components in m/s,
               precipitation in mm).

        3. **Output and Return:**
           - Prints a message indicating the path to the created OBS_DOMAIN file.
           - Returns `None`.

        Note:
           - This function currently only supports surface observations (`is_sound=False`).
             Setting `is_sound=True` will raise a `NotImplementedError`.
           - The function relies on the DataFrame returned by `self._converter` having
             specific columns named: 'Date string', 'Latitude', 'Longitude', 'ID', 'Name',
             'FM-Code', 'Source', 'Elevation', 'Bogus', 'SLP Pressure', 'ref_pres',
             't2m_K', 'u', 'v', 'Relative Humidity', 'Surface Pressure', 'Precipitation (mm)'.
             Ensure your input data and `_converter` method produce a DataFrame with these columns
             and in the expected units for WRF ObsNudging.
           - The output file format is fixed-width, as required by the OBS_DOMAIN format.
           - Progress of row processing is displayed using `tqdm`.

        Parameters
        ----------
        filename : str
            The path to the input data file that will be processed by the internal
            `_converter` method to create a Pandas DataFrame.
        domain_number : int
            The domain number for the WRF model. This number is used in the output
            OBS_DOMAIN filename (e.g., OBS_DOMAIN**domain_number**01).
        output_filepath : str, optional
            The base file path for the output OBS_DOMAIN file. The domain number and
            a suffix '01' will be appended to this path to create the final filename.
            Defaults to "OBS_DOMAIN".
        is_sound : bool, optional
            A flag indicating whether the input data represents sounding observations.
            Currently, only surface observations (`is_sound=False`) are supported.
            Setting this to `True` will raise a `NotImplementedError`. Defaults to `False`.

        Returns
        -------
        None
            This function does not return any value. It writes the OBS_DOMAIN formatted
            data to a file.

        Raises
        ------
        NotImplementedError
            If `is_sound` is set to `True`, as sounding observation conversion is not
            currently implemented.

        References
        ----------
        - WRF Observation Nudging Guide:
          https://www2.mmm.ucar.edu/wrf/users/docs/ObsNudgingGuide.pdf
        - WRFDA Online Tutorial - OBS_DOMAIN format:
          https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html#FM
        """
        df = self._converter(filename)
        df["ID"] = df["ID"].astype(str)
        
        #ensure chronological order:
        df = df.sort_values(by="time",ignore_index=True)
        
        is_sound_char = "F"
        QC_char = 0
        # For surface obs, meas_count is 1, for soundings, it's the number of levels in the sounding (assuming each row is a level for sounding)
        if not is_sound:
            meas_count = 1
        else:
            meas_count = len(df)
            raise NotImplementedError("sounding is not implemented")
        
        OUTFILE = f"{output_filepath}{domain_number}01"
        with open(OUTFILE, 'w') as outfile:
            for _, row in tqdm(df.iterrows(),desc="Creating rows",ascii=True,unit=" rows",total=len(df.index)):
                outfile.write(f" {row['Date string']:14}\n")
                outfile.write((f"  {row['Latitude']:9.4f} {row['Longitude']:9.4f}\n"))  #latlon
                outfile.write(f"  {row['ID'].ljust(40)}   {row['Name'].ljust(40)}   \n") #line 3
                outfile.write(f"  {row['FM-Code']:16}  {row['Source']:16}  {row['Elevation']:8.0f}.  {is_sound_char.rjust(4)}  {row['Bogus'].rjust(4)}  {meas_count:5d}\n")
                surface_data_line_values = [
                        f"{row['SLP Pressure']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['ref_pres']:11.3f}", f"{row['ref_pres']:11.3f}",
                        f"{row['Elevation']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['t2m_K']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['u']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['v']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['Relative Humidity']:11.3f}", f"{QC_char:11.3f}",
                        f"{row['Surface Pressure']:11.3f}", f"{QC_char:11.3f}"
                        f"{row['Precipitation (mm)']:11.3f}", f"{QC_char:11.3f}"
                    ]
                outfile.write(" ".join(surface_data_line_values) + "\n")
        print(f"OBS_DOMAIN file written to: {OUTFILE}")
        return None

    @property
    def states(self) -> list:
        return self.df_stations["state"].unique()

    @property
    def wrf_fdda_properties(self) -> list:
        return ["ff","dd","tl","rf","p","pred","rr"]

    @property
    def available_parameters(self) -> pd.DataFrame:
        return self.df_parameters.query("~name.str.contains('flag')")

    def plot(self):
        NotImplementedError("...")