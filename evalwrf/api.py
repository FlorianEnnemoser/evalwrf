from pathlib import Path
import cdsapi
from itertools import product
from typing import Literal, Self
import requests
import time as t
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from itertools import product

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
    """
    type : Literal["grid","timeseries","station"]
    mode : Literal["historical","current","forecast"]
    resource : str

    def __post_init__(self):
        self.url = "https://dataset.api.hub.geosphere.at"
        self.version = "v1"
        self.request_url = "/".join([self.url,self.version,self.type,self.mode,self.resource])
