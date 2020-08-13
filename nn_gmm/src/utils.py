import json
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd


def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """This is the same as a np.isin,
    however is significantly faster for large arrays

    https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0


class GenericObjJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as ex:
            return str(obj)


def get_station_lookup(X: pd.DataFrame):
    """Creates a station - id lookup dataframe"""
    X = X.loc[:, ["lon", "lat"]].copy()

    X["station"] = get_station_from_id(X.index.values.astype(str))
    X.drop_duplicates("station", inplace=True)
    station_lookup = X.set_index("station")

    return station_lookup


def get_station_from_id(ids: np.ndarray) -> List[str]:
    """Computes the stations from station_rupture ids"""
    return [cur_split[-1] for cur_split in np.char.split(ids, "_")]


def to_path(input: Union[str, List[str], List[Path]] = None):
    if isinstance(input, str):
        return Path(input)
    elif isinstance(input, List):
        return [Path(cur_input) if isinstance(cur_input, str) else cur_input for cur_input in input]

    return input