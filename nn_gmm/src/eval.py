from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from .data import load_tfrecord, load_feature_details
from .model import GMM


def get_realisation_residuals(
    data_dirs: Union[List[Path], Path],
    model: Union[GMM, Path],
    abs_residual: bool = True,
):
    """Computes realisation residuals for all realisations found in the
    specified data directories"""
    # Find all .tfrecord files
    if isinstance(data_dirs, list):
        tf_files = []
        for cur_dir in data_dirs:
            tf_files.extend(list(cur_dir.glob("*.tfrecord")))
    else:
        tf_files = list(data_dirs.glob("*.tfrecord"))

    # Load the model
    model = model if isinstance(model, GMM) else GMM.load(model)

    feature_details = (
        load_feature_details(data_dirs[0])
        if isinstance(data_dirs, list)
        else load_feature_details(data_dirs)
    )

    rel_res_dfs = []
    for ix, cur_tf_ffp in enumerate(tf_files):
        print(f"Processing {ix + 1}/{len(tf_files)}")

        cur_data = load_tfrecord(str(cur_tf_ffp), feature_details)
        cur_mean_df, _ = model.predict(cur_data)
        assert np.all(cur_mean_df.index == cur_data.index)

        ims = cur_mean_df.columns.values.astype(str)

        split_ids = np.stack(np.char.split(cur_mean_df.index.values.astype(str), "_"))
        cur_mean_df["realisation"] = np.char.add(
            np.char.add(split_ids[:, 0], "_"), split_ids[:, 1]
        )
        cur_data["realisation"] = cur_mean_df.realisation

        cur_data["station"] = cur_mean_df["station"] = split_ids[:, -1]

        cur_mean_df["fault"] = split_ids[:, 0]
        rel_fault = cur_mean_df.groupby("realisation").first()["fault"].to_frame()

        rel_mag = cur_data.groupby("realisation").first()["mag"]
        rel_n_stations = cur_data.groupby("realisation").count()["station"]
        rel_n_stations.name = "n_stations"

        cur_rel_res_dfs = []
        for im in ims:
            res = (
                cur_data.loc[:, im] - cur_mean_df.loc[:, im].apply(np.log)
            ).to_frame()
            if abs_residual:
                res[im] = res[im].apply(np.abs)

            res["realisation"] = cur_mean_df.realisation

            cur_rel_res_dfs.append(res.groupby("realisation").mean())

        cur_rel_res_df = pd.concat(cur_rel_res_dfs, axis=1)
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_mag, left_index=True, right_index=True
        )
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_fault, left_index=True, right_index=True
        )
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_n_stations, left_index=True, right_index=True
        )
        rel_res_dfs.append(cur_rel_res_df)

    rel_res_df = pd.concat(rel_res_dfs)
    return rel_res_df
