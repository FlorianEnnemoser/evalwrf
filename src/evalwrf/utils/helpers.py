from pathlib import Path
import json
from tqdm import tqdm
from httpx import Response


def create_folder(folder_name: str) -> Path:
    folder_name: Path = Path(folder_name)
    if not folder_name.is_dir():
        folder_name.mkdir(parents=True, exist_ok=True)
    return folder_name


def load_json(filename: str) -> dict:
    """Load a JSON file from disk.

    Parameters
    ----------
    filename : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If *filename* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.

    Examples
    --------
    >>> data = load_json("config/Datasets.json")
    """
    with Path(filename).open("r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return data


def save_data_stream(response: Response, filename: str) -> None:
    with open(filename, "wb") as f:
        for chunk in tqdm(response.iter_bytes(), ascii=True, desc=f"Saving {filename}"):
            f.write(chunk)
    return None
