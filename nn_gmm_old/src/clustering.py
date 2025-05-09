from typing import List, Dict, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from qcore import geo

from . import eval


def compute_clusters(
    model_dir: Path,
    ims: Sequence[str],
    eps: float = 0.1,
    min_samples: int = 4,
    bias_threshold: float = 0.1,
    spatial_radius: float = 12000
):
    """
    Performs clustering based on the model's spatial bias
    on the training data

    Parameters
    ----------
    model_dir: Path
    ims: sequence of strings
    eps: float, optional
        DBSCAN argument, neighborhood radius
        In this context, this parameter only affects
        the bias "radius", i.e. increasing this will
        result in cluster with a larger "variance"
    min_samples: int, optional
        Minimum samples to form a cluster
    bias_threshold: float, optional
        Threshold below/above which no clustering is performed
        Use this to prevent clustering in areas with a small bias
    spatial_radius: float, optional
        The spatial neighborhood radius (in metres)

    Returns
    -------
    dictionary of dataframes:
        One dataframe per specified IM
    """
    train_spatial_metrics, _ = eval.load_spatial_metrics(model_dir)

    data_df = train_spatial_metrics["bias"]

    # Only use base stations
    data_df = data_df.loc[
        np.char.startswith(data_df.site.values.astype(str), "0")
    ].copy()
    data_df = data_df.set_index("site")

    # Convert to NZGD2000, which is in metres
    station_coords = geo.wgs_nztm2000x(
        np.stack((data_df.lon.values, data_df.lat.values), axis=1)
    )

    data_df.loc[:, "X"] = station_coords[:, 0]
    data_df.loc[:, "Y"] = station_coords[:, 1]

    results = {}
    for cur_im in ims:
        cur_df = data_df.copy()
        cur_im_adj = f"{cur_im}_adj"

        # Only use stations that have a large bias
        large_bias_mask = (cur_df[cur_im] > bias_threshold) | (
            cur_df[cur_im] < -bias_threshold
        )
        cur_df[f"{cur_im}_adj"] = np.where(~large_bias_mask, 0.0, cur_df[cur_im].values)

        invalid_value = 9999999.0

        # Compute custom distance matrix
        # Compute distance between all stations
        dist_df = pd.DataFrame(
            index=cur_df.loc[large_bias_mask].index,
            columns=cur_df.loc[large_bias_mask].index,
            data=cdist(
                cur_df.loc[large_bias_mask, ["X", "Y"]].values,
                cur_df.loc[large_bias_mask, ["X", "Y"]].values,
            ),
        )
        # Ignore stations with distance > 12km away
        # (this means that for the 8km base grid,
        # the diagonal stations are also included)
        dist_matrix = np.where(dist_df.values > spatial_radius, np.inf, dist_df.values)

        # Compute the bias distance (which is the actual distance used for clustering)
        bias_dist = np.abs(
            cur_df.loc[large_bias_mask, cur_im_adj].values
            - cur_df.loc[large_bias_mask, cur_im_adj].values[:, None]
        )
        dist_matrix = np.where(dist_matrix == np.inf, invalid_value, bias_dist)
        dist_df = pd.DataFrame(
            index=dist_df.index, columns=dist_df.columns, data=dist_matrix
        )

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(
            dist_df.values
        )

        cur_df["cluster"] = np.nan
        cur_df.loc[large_bias_mask, "cluster"] = clustering.labels_
        print(f"{cur_im} - Number of clusters {np.count_nonzero(np.unique(clustering.labels_) >= 0)}")

        results[cur_im] = cur_df.loc[:, ["lon", "lat", "cluster"]]

    return results
