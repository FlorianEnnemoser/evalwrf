from pathlib import Path
import pandas as pd
from warnings import warn


ATTRIBUTE_COLS = [
    "Nabenhöhe (m)",
    "Durchmesser (m)",
    "Standing Thrust Coeffcient",
    "Nennleistung (MW)",
]


def read_windturbines_file(filename="windturbines.txt") -> pd.DataFrame:
    """
    https://github.com/wrf-model/WRF/blob/master/doc/README.windturbine

    thrust coefficient (ct): ct = T/(0.5 * rho*A*V**2)

    ### Interpretation of ct:
        * Tells you how much of the wind’s momentum is extracted by the turbine.
        * A higher ct means more force on the turbine, which can increase wake effects and loading on the structure.

    """

    wind_speed = []
    thrust_coefficient = []
    power_production = []

    with Path(filename).open() as f:
        for i, line in enumerate(f):
            if i == 0:
                max_lines = int(line.strip())

            if i > max_lines + 1:
                print(f"more lines than max lines! exiting file at entry {max_lines}!")
                break

            parts = line.strip().split()
            parts = [float(p) for p in parts]

            if i == 1:
                wind_turbine = dict(zip(ATTRIBUTE_COLS, parts))

            if i > 1:
                wind_speed.append(parts[0])
                thrust_coefficient.append(parts[1])
                power_production.append(parts[2])

    turbines = pd.DataFrame(
        {
            "wind_speed_ms-1": wind_speed,
            "thrust_coeff": thrust_coefficient,
            "power_production_kW": power_production,
        }
    )
    turbines.attrs = wind_turbine
    return turbines


def to_windturbine(df: pd.DataFrame, filename: str = "wind-turbine-XXX.tbl") -> None:
    filename: Path = Path(filename)

    if not filename.suffix.endswith("tbl"):
        raise ValueError("File must end with `.tbl`!")

    if filename.is_file():
        raise ValueError("File already exists!")

    _req_columns = ["wind_speed_ms-1", "thrust_coeff", "power_production_kW"]
    if not df.columns.isin(_req_columns).all() or len(df.columns) != len(_req_columns):
        raise ValueError(
            f"Provide required columns {_req_columns} and remove potentially unused columns!"
        )

    header = [str(df.attrs.get(c)) for c in ATTRIBUTE_COLS]
    if len(header) != len(ATTRIBUTE_COLS):
        raise ValueError(
            f"Missing data in dataframe attrs! Provide all of: {ATTRIBUTE_COLS}"
        )

    df.to_csv(filename, sep=" ", index=False, header=False)

    with filename.open() as f:
        content = f.read()

    max_lines_line = str(df.index.size)

    turbine_attributes_line = " ".join(header)
    header_info = [max_lines_line, turbine_attributes_line]

    with filename.open(mode="w") as f:
        for line in header_info:
            f.write(line + "\n")
        f.write(content)
    return None


def save_windfarm(df: pd.DataFrame, filename: str = "windturbines.txt") -> None:
    """
    Beispiel df:
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
    >>> save_windfarm(df, filename="windturbines_baernofen.txt")
    """

    if filename != "windturbines.txt":
        warn(
            "WRF always needs the windfarm locations file to be named 'windturbines.txt'!"
        )

    _req_columns = ["Latitude", "Longitude", "Index Value"]
    if not df.columns.isin(_req_columns).all() or len(df.columns) != len(_req_columns):
        raise ValueError(
            f"Provide required columns {_req_columns} and remove potentially unused columns!"
        )

    df.to_csv(filename, sep=" ", index=False, header=False)
    return None


def save_tbl():
    df = pd.read_excel("Windturbines.xlsx", sheet_name="Leistungskurven", usecols="A:H")

    configs = {
        "V117": dict(sheet_name="Vestas", usecols="A:C"),
        "V122": dict(sheet_name="Vestas", usecols="D:F"),
        "V126": dict(sheet_name="Vestas", usecols="G:I"),
        "V162": dict(sheet_name="Vestas", usecols="J:L"),
        "V136": dict(sheet_name="Vestas", usecols="M:O"),
        "V150": dict(sheet_name="Vestas", usecols="P:R"),
        "V112": dict(sheet_name="Vestas", usecols="S:U"),
        "N149": dict(sheet_name="Nordex", usecols="A:C"),
        "N163": dict(sheet_name="Nordex", usecols="D:F"),
        "E82-E4": dict(sheet_name="Enercon", usecols="A:C"),
    }

    for k, v in configs.items():
        data: pd.DataFrame = pd.read_excel(
            "Windturbines.xlsx", skiprows=1, **v
        ).dropna()
        data.columns = [c.split(".")[0] for c in data.columns]
        attribute_data = df[df["Callname"] == k].drop_duplicates(subset="Callname")
        attribute_data = attribute_data.drop(columns=["Marke", "Name", "Callname"])

        for col in attribute_data:
            val = attribute_data[col].item()
            data.attrs[col] = val

        to_windturbine(
            df=data, filename=f"wind-turbine-{data.attrs['Index Value']}.tbl"
        )
    return None
