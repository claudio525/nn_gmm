import json
from pathlib import Path
from typing import Any, List, Union
from PIL import Image

import numpy as np
import pandas as pd
from scipy import interpolate


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
        return [
            Path(cur_input) if isinstance(cur_input, str) else cur_input
            for cur_input in input
        ]

    return input

def to_list(input: Union[Any, List[Any]]):
    return input if isinstance(input, list) else [input]


def load_dfs(df_ffps: List[Union[str, Path]], **kwargs):
    return pd.concat([pd.read_csv(df_fp, **kwargs) for df_fp in df_ffps])


def combine_imgs(img_ffp_1: Path, img_ffp_2: Path, output_ffp: Path):
    cur_img_1 = Image.open(img_ffp_1)
    cur_img_2 = Image.open(img_ffp_2)

    width = cur_img_1.size[0] + cur_img_2.size[0]

    new_im = Image.new("RGB", (width, cur_img_1.size[1]))
    new_im.paste(cur_img_1, (0, 0))
    new_im.paste(cur_img_2, (cur_img_1.size[0], 0))

    new_im.save(output_ffp)


def interpolate_pSA_periods(im_df: pd.DataFrame, target_ims: np.ndarray):
    """Selects the pSA periods of interest if available in the specified
    IM dataframe, otherwise interpolates to get the desired range of
    pSA periods"""
    if np.all(np.isin(target_ims, im_df.columns)):
        return im_df

    ims = im_df.columns.values.astype(str)
    pSA_mask = np.char.startswith(ims, "pSA_")
    pSA_periods = np.stack(np.char.split(ims[pSA_mask], "_"))[:, 1].astype(float)

    target_mask = np.char.startswith(target_ims, "pSA_")
    target_periods = np.sort(
        np.stack(np.char.split(target_ims[target_mask], "_"))[:, 1].astype(float)
    )

    # Interpolate
    assert np.all(np.sort(pSA_periods) == pSA_periods)
    f = interpolate.interp1d(
        np.log(pSA_periods),
        im_df.loc[:, ims[pSA_mask]].values,
        kind="linear",
        bounds_error=True,
    )
    target_values = f(np.log(target_periods))

    pSA_df = pd.DataFrame(
        columns=np.char.add("pSA_", target_periods.astype(str)),
        data=target_values,
        index=im_df.index,
    )
    result_df = pd.merge(
        im_df.loc[:, ims[~pSA_mask]],
        pSA_df,
        left_index=True,
        right_index=True,
        how="inner",
    )

    assert result_df.shape[0] == im_df.shape[0]
    return result_df
