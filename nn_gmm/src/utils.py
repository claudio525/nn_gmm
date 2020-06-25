import json
from typing import Any

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


def sel_rand_locations(df: pd.DataFrame, n_locs: int):
    """Selects a set of random locations from the
    specified dataframe

    Parameters
    ----------
    df: dataframe
        Contains the locations from which to select
        Must have the columns lon & lat
    n_locs: int
        Number of locations to select

    Returns
    -------
    dataframe
        with the selected locations
        columns: [lon, lat]
    """
    locations = np.unique(df.loc[:, ("lon", "lat")].values, axis=0)
    return pd.DataFrame(
        data=locations[np.random.randint(0, locations.shape[0], n_locs), :],
        columns=["lon", "lat"],
    )
