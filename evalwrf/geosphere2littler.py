import pandas as pd
import numpy as np
import fortranformat as ff

# https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html
# https://pypi.org/project/fortranformat/

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

df = pd.read_json("klima-v2-10min_METADATA.json" ,orient="index").T.drop(columns=["type","mode","title","frequency","response_formats","start_time","end_time","id_type"])
df_stations = pd.json_normalize(df["stations"]).set_index("id")






df = pd.read_csv("Steiermark_2024-06-01_2024-06-10_FDDA.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(by="time",ignore_index=True)
df["name"] = df["station"].map(df_stations["name"])
df["altitude"] = df["station"].map(df_stations["altitude"])
df["lat"] = df["station"].map(df_stations["lat"])
df["lon"] = df["station"].map(df_stations["lon"])



df["S"] = saturation_vapor_density(df["tl"])
df["Q"] = df["S"] * df["rf"] / 100

df["u"] = u_from_vector(df["ff"],df["dd"])
df["v"] = v_from_vector(df["ff"],df["dd"])

df["FM-Code"] = "FM-35" # FM-35 oder FM-12 idk https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html#FM

df = df.rename(columns={"lat":"Latitude","lon":"Longitude","station":"ID","name":"Name","altitude":"Elevation"})

# Spalten die "Mandatory" sind, müssen drin und ausgefüllt sein. Die die Optional sind, müssen auch drin sein und sollen "missing" sein wenns ned passt
LITTLE_R_COLUMNS = [
    "Latitude","Longitude",
    "ID","Name",
    "FM-Code","Source",
    "Elevation",
    "Bogus", "Discard",
    "Unix time", "Julian day",
    "Date string", "SLP","SLP_QC",
    "SFC Pressure","SFC Pressure_QC"
]
dfr = df.reindex(columns=LITTLE_R_COLUMNS)
dfr["Source"] = "Geosphere Austria"
dfr["Sequence number"] = dfr.index[::-1] #	This field is only used as one of the "tiebreakers" for deciding which observation header will be kept when merging duplicate obs. It is assumed that lower numbers are more recent observations.
dfr["Bogus"] = False # oder `"T"`
dfr["Discard"] = False
dfr["Unix Time"] = df["time"].astype(int) / 10**9
dfr["Julian day"] = df["time"].apply(lambda t: t.to_julian_date()).astype(int)
dfr["Date string"] = df["time"].dt.strftime('%Y%m%d%H%M%S')
dfr = dfr.fillna(777777)
print(dfr)
print(dfr.columns)
print(dfr.dropna(axis="columns"))

HEADER_FORMAT = '( 2f20.5 , 2a40 , 2a40 , 1f20.5 , 1i10 , 2L10 , 2i10 , a20 ,  13( f13.5 , i7 ) )'
DATA_FORMAT = '( 10( f13.5 , i7 ) )'
END_FORMAT = '( 3 ( i7 ) )'
header_line = ff.FortranRecordWriter(HEADER_FORMAT)
data_writer = ff.FortranRecordWriter(DATA_FORMAT)
end_writer = ff.FortranRecordWriter(END_FORMAT)

print(dfr.loc[0])
print(dfr.loc[0].tolist())
whatisthis = header_line.write(dfr.loc[0].tolist())
print(whatisthis)

