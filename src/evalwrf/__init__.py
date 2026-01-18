from plotting import timeseries_station
from api import load_url_from_resource, save_csv, load_metadata
from data import load_dataset_from_csv

# API_RESOURCE = "klima-v2-10min"
# url = load_url_from_resource("Datasets.json", API_RESOURCE)
# save_csv(
#     url,
#     filename="murau_data.csv",
#     params=dict(
#         start="2025-12-30",
#         end="2026-01-01",
#         parameters=["TL", "RR", "cglo"],
#         station_ids=15920,
#         output_format="csv",
#     ),
# )

# df = pd.read_csv("murau_data.csv").dropna()
# df["time"] = pd.to_datetime(df["time"], utc=True)


# timeseries(df, y="tl", filename="murau.svg", add_daynight=True, y_lim=(-10, 3))
