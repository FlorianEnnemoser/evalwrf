from typing import List, Literal
import httpx
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def load_json(filename: str) -> dict:
    with Path(filename).open("r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return data


class URL(httpx.URL):
    def __truediv__(self, other: str | List[str]):
        if isinstance(other, str):
            other = [other]
        return __class__(self.__str__() + "/" + "/".join(other))


def save_json_from_URL(url: URL, filename: str, **kwargs) -> None:
    """
    url = load_url_from_resource("Datasets.json", API_RESOURCE)
    save_json_from_URL(url / "metadata", f"Metadata_{API_RESOURCE}")
    """
    response = httpx.get(url, **kwargs)

    if not response.status_code == 200:
        raise ValueError("Incorrect status code!")

    with open(f"{filename}.json", "w") as f:
        json.dump(response.json(), f, indent=4)
    return None


def save_netcdf(url: URL, filename: str, **kwargs) -> None:
    with httpx.stream("GET", url, timeout=60, **kwargs) as response:
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in tqdm(response.iter_bytes(), ascii=True, desc="Saving NetCDF"):
                f.write(chunk)
    print(f"Saved file: {filename}")


def save_csv(url: URL, filename: str, **kwargs):
    """
    save_csv(
        url,
        filename="murau_data.csv",
        params=dict(
            start="2025-12-30",
            end="2026-01-01",
            parameters=["TL", "RR"],
            station_ids=15920,
            output_format="csv",
        ),
    )
    """
    with httpx.stream("GET", url, timeout=10, **kwargs) as response:
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    print(f"Saved file: {filename}")
    return None


def load_url_from_resource(filename: str, resource: str) -> URL:
    """
    >>> url = load_url_from_resource("config/Datasets.json", resource="klima-v2-10min")
    >>> URL('https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-10min')
    """
    data = load_json(filename)
    for k, v in data.items():
        if resource in k:
            return URL(v.get("url", ""))
    return None


def load_metadata(filename: str, record: Literal["parameters", "stations"]) -> pd.DataFrame:
    data = load_json(filename)
    idx = "name" if record == "parameters" else "id"
    df = pd.json_normalize(data, record_path=record).set_index(idx)
    return df
