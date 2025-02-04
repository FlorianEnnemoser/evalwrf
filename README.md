"""
The following are mandatory input data fields:

3D Data (for data on constant pressure levels)
Temperature
U and V components of wind
Geopotential height
Relative humidity (or specific humidity)


3D Data (for data on native model levels)
Temperature
U and V components of wind
Geopotential height
Relative humidity (or specific humidity)
Pressure


2D Data
Surface pressure
Mean sea-level pressure
Skin temperature/SST
2 meter temperature
2 meter relative humidity
10 meter U and V components of wind
Soil temperature
Soil moisture
Soil height (or terrain height)
*Note: the 2m temperature, RH, and wind fields may be optional if one sets the namelist option use_surface = F before running real.exe


The following are recommended fields:
LANDSEA mask field for input data
Water equivalent snow depth
SEAICE (especially good for a high-latitude winter case)
Additional SST data (this is mandatory for climate runs)

https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20241104/00/atmos/
"""
