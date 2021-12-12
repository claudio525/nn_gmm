from pathlib import Path
from typing import Any, List, Union, Dict
from PIL import Image

import numpy as np
import pandas as pd
from scipy import interpolate

from . import data


def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """This is the same as a np.isin,
    however is significantly faster for large arrays

    https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0


def get_station_lookup(X: pd.DataFrame):
    """Creates a station - id lookup dataframe"""
    X = X.loc[:, ["lon", "lat"]].copy()

    X["station"] = get_station_from_id(X.index.values.astype(str))
    X.drop_duplicates("station", inplace=True)
    station_lookup = X.set_index("station")

    return station_lookup


def get_station_from_id(ids: np.ndarray) -> List[str]:
    """Computes the stations from station_rupture ids"""
    return np.stack(np.char.rsplit(ids, "_", maxsplit=1))[:, -1]
    # return [cur_split[-1] for cur_split in np.char.split(ids, "_")]


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
    pSA_periods = np.char.replace(
        np.stack(np.char.split(ims[pSA_mask], "_"))[:, 1], "p", "."
    ).astype(float)

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


def convert_pre_config(config: Dict, stats_df: pd.DataFrame):
    """Adds the correct stats parameters to a input or output config"""
    for key, item in config.items():
        if item is None:
            continue
        elif item == "standard":
            config[key] = (item, stats_df.loc[key, "mean"], stats_df.loc[key, "std"])
        elif item == "min_max":
            config[key] = (item, stats_df.loc[key, "min"], stats_df.loc[key, "max"])
        elif item == "ln":
            config[key] = (item,)
        else:
            raise ValueError(f"{item} is not a valid preprocessing config value")

    return config


def convert_io_config(io_config: Dict, stats_df: pd.DataFrame):
    """Converts an input & output config"""
    if io_config["multi_input"]:
        feature_config_dict = io_config["feature_config"]
        for cur_input_name, cur_config in feature_config_dict.items():
            feature_config_dict[cur_input_name] = convert_pre_config(
                cur_config, stats_df
            )
        io_config["feature_config"] = feature_config_dict
    else:
        io_config["feature_config"] = convert_pre_config(
            io_config["feature_config"], stats_df
        )

    io_config["im_config"] = convert_pre_config(io_config["im_config"], stats_df)

    return io_config


def get_repo_version():
    """Gets the current commit hash"""
    import git
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def find_record_ffp(data_dirs: List[Path], event: str):
    """Finds the tfrecord file for the given event in
    the specified directories, raises ValueError if no
    file is found
    """
    results = []
    for cur_dir in data_dirs:
        cur_r = list(cur_dir.glob(f"{event}.tfrecord"))

        if len(cur_r) > 0:
            cur_feature_details = data.load_feature_details(cur_dir)
            results.append((cur_r[0], cur_feature_details))

    if len(results) == 0:
        raise ValueError(
            f"No tfrecord file could be found for the specified event: {event}"
        )

    assert len(results) == 1, "More than one tfrecord file found"

    return results[0]


# def pa_column_types(column_types: Dict[str, Type]):
#     pa_column_types = {}
#     for cur_key, cur_type in column_types.items():
#         if cur_type is None or cur_type is np.dtype(object):
#             continue
#         elif cur_type in (np.float64, np.dtype(np.float64)):
#             pa_column_types[cur_key] = pa.float64()
#         elif cur_type in (float, np.float32, np.dtype(np.float32)):
#             pa_column_types[cur_key] = pa.float32()
#         elif cur_type in (np.float16, np.dtype(np.float16)):
#             pa_column_types[cur_key] = pa.float16()
#         elif cur_type in (np.int64, np.dtype(np.int64)):
#             pa_column_types[cur_key] = pa.int64()
#         elif cur_type in (int, np.int32, np.dtype(np.int32)):
#             pa_column_types[cur_key] = pa.int32()
#         elif cur_type in (np.int16, np.dtype(np.int16)):
#             pa_column_types[cur_key] = pa.int16()
#         elif cur_type in [bool, np.dtype(bool)]:
#             pa_column_types[cur_key] = pa.bool_()
#         else:
#             raise NotImplementedError(f"No converstion type for type {cur_type} specified")
#     return pa_column_types