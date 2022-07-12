from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

from qcore import geo, nhm

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import nn_gmm

model_dir = Path(
    "/home/claudy/dev/work/data/nn_gmm/results/keep_runs/z1p0/0705_130827_location_model-1_base-grid_comb-delta-z1p0"
)
site_params_ffp = Path(
    "/home/claudy/dev/work/data/nn_gmm/input_data/site_params/site_params.csv"
)

# site_df = pd.read_csv(site_params_ffp, index_col=0)
train_spatial_metrics, _ = nn_gmm.load_spatial_metrics(model_dir)

cur_df = train_spatial_metrics["bias"]

# Only use base stations
cur_df = cur_df.loc[np.char.startswith(cur_df.site.values.astype(str), "0")].copy()
cur_df = cur_df.set_index("site")

# Convert to NZGD2000, which is in metres
station_coords = geo.wgs_nztm2000x(
    np.stack((cur_df.lon.values, cur_df.lat.values), axis=1)
)

cur_df.loc[:, "X"] = station_coords[:, 0]
cur_df.loc[:, "Y"] = station_coords[:, 1]

cur_im = "pSA_3.0"
cur_im_adj = f"{cur_im}_adj"

large_bias_mask = (cur_df[cur_im] > 0.1) | (cur_df[cur_im] < -0.1)
cur_df[f"{cur_im}_adj"] = np.where(~large_bias_mask, 0.0, cur_df[cur_im].values)

# # Have to scale for clustering
# scaler = MinMaxScaler(feature_range=(-1, 1))
# cur_df.loc[
#     large_bias_mask, ["X_scaled", "Y_scaled", f"{cur_im}_adj_scaled"]
# ] = scaler.fit_transform(
#     cur_df.loc[large_bias_mask, ["X", "Y", f"{cur_im}_adj"]].values
# )

invalid_value = 9999999.0

# Custom distance matrix
dist_df = pd.DataFrame(index=cur_df.loc[large_bias_mask].index, columns=cur_df.loc[large_bias_mask].index,
                       data=cdist(cur_df.loc[large_bias_mask, ["X", "Y"]].values, cur_df.loc[large_bias_mask, ["X", "Y"]].values))
dist_matrix = np.where(dist_df.values > 12000, np.inf, dist_df.values)


bias_dist = np.abs(cur_df.loc[large_bias_mask, cur_im_adj].values - cur_df.loc[large_bias_mask, cur_im_adj].values[:, None])
dist_matrix = np.where(dist_matrix == np.inf, invalid_value, bias_dist)
dist_df = pd.DataFrame(index=dist_df.index, columns=dist_df.columns, data=dist_matrix)

print("wtf")

# Custom scaling attempt
# cur_df[f"{cur_im}_adj_scaled"] = (cur_df[f"{cur_im}_adj"] / 0.1) * 8000
#
# clustering = DBSCAN(eps=10000, min_samples=2).fit(
#     cur_df.loc[large_bias_mask, ["X", "Y", f"{cur_im}_adj_scaled"]].values
# )
# clustering = DBSCAN(eps=0.1, min_samples=2).fit(
#     cur_df.loc[large_bias_mask, ["X_scaled", "Y_scaled", f"{cur_im}_adj_scaled"]].values
# )


clustering = DBSCAN(eps=0.1, min_samples=4, metric="precomputed").fit(dist_df.values)

cur_df["cluster"] = np.nan
cur_df.loc[large_bias_mask, "cluster"] = clustering.labels_
print(f"Number of clusters {np.unique(clustering.labels_).size - 1}")

# clusterer = hdbscan.HDBSCAN()
# clustering = clusterer.fit(cur_df.loc[large_bias_mask, ["X", "Y", f"{cur_im}_adj"]].values)
# cur_df.loc[large_bias_mask, "cluster"] = clustering.labels_

print("wtf")


# Original
fig = plt.figure(figsize=(16, 10), dpi=400)
plt.scatter(
    cur_df.X, cur_df.Y, s=1.0, c=cur_df[cur_im], cmap="seismic_r", vmin=-0.4, vmax=0.4
)
plt.colorbar(pad=0.0)
plt.tight_layout()
plt.show()

# Adjusted
fig = plt.figure(figsize=(16, 10), dpi=400)
plt.scatter(
    cur_df.loc[~large_bias_mask, "X"],
    cur_df.loc[~large_bias_mask, "Y"],
    s=1.0,
    c="k",
    marker="x",
    alpha=0.3,
)
plt.scatter(
    cur_df.loc[large_bias_mask, "X"],
    cur_df.loc[large_bias_mask, "Y"],
    s=1.0,
    c=cur_df.loc[large_bias_mask, f"{cur_im}_adj"],
    cmap="seismic_r",
    vmin=-0.4,
    vmax=0.4,
)
plt.colorbar(pad=0.0)
plt.tight_layout()
plt.show()


# Plot cluster
fig = plt.figure(figsize=(16, 10), dpi=400)
cur_mask = (cur_df[f"{cur_im}_adj"] > 0.15) | (cur_df[f"{cur_im}_adj"] < -0.15)
plt.scatter(
    cur_df.loc[~cur_mask, "X"],
    cur_df.loc[~cur_mask, "Y"],
    s=1.0,
    c="k",
    marker="x",
    alpha=0.3,
)

for cur_cluster in np.unique(clustering.labels_):
    cur_mask = cur_df.cluster == cur_cluster
    plt.scatter(
        cur_df.loc[cur_mask, "X"],
        cur_df.loc[cur_mask, "Y"],
        s=1.0,
        c="k" if cur_cluster == -1 else None,
    )

plt.colorbar(pad=0.0)
plt.tight_layout()
plt.show()
