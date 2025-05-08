import time
from pathlib import Path

import ml_tools.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import pygmt
import typer

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

MAG_N_BINS = 20
MAG_SAMPLE_WEIGHT_THRESHOLD = 20.0

RRUP_N_BINS = 20
RRUP_SAMPLE_WEIGHT_THRESHOLD = 15.0

RRUP_MAG_SAMPLE_WEIGHT_THRESHOLD = 200.0

def mag_hist(mag_values: np.ndarray, weights: np.ndarray, output_ffp: Path):
    fig = plt.figure(figsize=(20, 12))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.hist(mag_values, bins=25)
    ax_1.set_title("Original")

    ax_3 = fig.add_subplot(2, 1, 2, sharex=ax_1)
    ax_3.hist(mag_values, weights=weights, bins=25)
    ax_3.set_title("Weighted")

    ax_3.set_xlabel(f"Magnitude")
    ax_3.set_ylabel(f"Count")
    ax_3.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

def rrup_hist(rrup_values: np.ndarray, weights: np.ndarray, output_ffp: Path):
    fig = plt.figure(figsize=(20, 12))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.hist(rrup_values, bins=25)
    ax_1.set_title("Original")

    ax_3 = fig.add_subplot(2, 1, 2, sharex=ax_1)
    ax_3.hist(rrup_values, weights=weights, bins=25)
    ax_3.set_title("Weighted")

    ax_3.set_xlabel(f"Rrup")
    ax_3.set_ylabel(f"Count")
    ax_3.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)


def main(
    output_dir: Path,
    source_params_dir: Path,
    site_params_dir: Path,
    distance_db_dir: Path,
    im_dbs_dir: Path,
    base_grid_only: bool = False
):
    print("Loading realisation params")
    rel_df = nn_gmm.load_dfs(
        list(source_params_dir.glob("*.csv")), index_col="realisation"
    )
    rel_df["source"] = [
        split_list[0]
        for split_list in np.char.split(rel_df.index.values.astype(str), "_")
    ]
    assert np.unique(rel_df.index.values.astype(str)).shape[0] == rel_df.shape[0]

    # Get full set of sample combinations
    print("Generating full set of sample combinations")
    idx_values = []
    im_db_ffps = im_dbs_dir.glob("*.h5")
    for cur_imdb_ffp in im_db_ffps:
        with pd.HDFStore(str(cur_imdb_ffp), mode="r") as db:
            for cur_key in db.keys():
                idx_values.append(db[cur_key].index.values.astype(str))

    idx_values = np.concatenate(idx_values)
    sample_df = nn_gmm.create_sample_comb(pd.DataFrame(index=idx_values))

    if base_grid_only:
        station_ids = sample_df.site.values.astype(str)
        mask = (
            np.char.startswith(station_ids, "1")
            | np.char.startswith(station_ids, "2")
            | np.char.startswith(station_ids, "3")
            | np.char.startswith(station_ids, "4")
        )
        sample_df = sample_df.loc[~mask]

    # Add magnitude
    sample_df.loc[:, "mag"] = rel_df.loc[sample_df.realisation, "mag"].values

    # Load sites
    site_df = nn_gmm.load_dfs(list(site_params_dir.glob("*.csv")))

    # Check & Drop duplicates
    n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
    site_df = site_df.drop_duplicates()
    assert n_unique_stations == site_df.shape[0]
    site_df.set_index("station", inplace=True)
    site_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

    print("Load and merge distance data")
    distance_db_ffps = list(distance_db_dir.glob("*.db"))
    distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=8)

    # Merge with distance parameters
    sample_df["id"] = sample_df.index.values.astype(str)
    sample_df = pd.merge(
        sample_df,
        distance_df,
        how="inner",
        left_on=["source", "site"],
        right_on=["source", "site"],
        copy=False,
    )
    sample_df.set_index("id", inplace=True)

    print("Computing combined magnitude-rrup weighting")
    sample_df["rrup_mag_weights"] = np.nan
    mag_bins = np.linspace(sample_df.mag.min(), sample_df.mag.max(), MAG_N_BINS + 1)
    rrup_bins = np.linspace(0.0, sample_df.rrup.max(), RRUP_N_BINS + 1)

    prev_mag_edge, prev_rrup_edge = mag_bins[0], rrup_bins[0]
    for cur_mag_edge in mag_bins[1:]:
        for cur_rrup_edge in rrup_bins[1:]:
            cur_mask = (sample_df.mag.values >= prev_mag_edge) & (sample_df.mag.values <= cur_mag_edge) & \
                       (sample_df.rrup.values >= prev_rrup_edge) & (sample_df.rrup.values <= cur_rrup_edge)

            if np.any(cur_mask):
                sample_df.loc[cur_mask, "rrup_mag_weights"] = 1.0 / np.count_nonzero(cur_mask)

            prev_rrup_edge = cur_rrup_edge

        prev_rrup_edge = rrup_bins.min()
        prev_mag_edge = cur_mag_edge

    # Normalise
    sample_df.rrup_mag_weights = sample_df.rrup_mag_weights * (sample_df.shape[0] / sample_df.rrup_mag_weights.sum())
    sample_df.loc[sample_df.rrup_mag_weights > RRUP_MAG_SAMPLE_WEIGHT_THRESHOLD, "rrup_mag_weights"] = RRUP_MAG_SAMPLE_WEIGHT_THRESHOLD

    # 2D Hist - Orig vs Weighted
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 1, 1)
    _, _, __, cb_map = ax.hist2d(sample_df.mag, sample_df.rrup, bins=(MAG_N_BINS, RRUP_N_BINS), norm=matplotlib.colors.LogNorm())
    plt.colorbar(cb_map, ax=ax)
    ax.set_ylabel(f"Rrup")

    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    _, _, __, cb_map = ax.hist2d(sample_df.mag, sample_df.rrup, weights=sample_df.rrup_mag_weights, bins=(MAG_N_BINS, RRUP_N_BINS), norm=matplotlib.colors.LogNorm())
    plt.colorbar(cb_map, ax=ax)

    ax.set_xlabel(f"Magnitude")
    ax.set_ylabel(f"Rrup")

    fig.tight_layout()
    fig.savefig(output_dir / "mag_rrup_2d_hist.png")

    # Rrup and Magnitude histogram
    mag_hist(sample_df.mag.values, sample_df.rrup_mag_weights.values, output_dir / f"rrup_mag_weights_mag_hist.png")
    rrup_hist(sample_df.rrup.values, sample_df.rrup_mag_weights.values, output_dir / f"rrup_mag_weights_rrup_hist.png")

    # Compute magnitude sample bins
    print("Computing magnitude weighting")
    mag_count, mag_bins = np.histogram(sample_df.mag, bins=MAG_N_BINS)

    mag_sample_bin_ind = np.digitize(sample_df.mag.values, mag_bins)

    # Histogram includes the right bin edge, but
    # digitize does not
    mag_sample_bin_ind[mag_sample_bin_ind > MAG_N_BINS] = MAG_N_BINS

    # Digitize also indices from 1
    # (as 0 is used for values out of bounds)
    mag_sample_bin_ind -= 1

    # Compute the bin weights
    mag_bin_weights = 1 / mag_count
    mag_sample_weights = mag_bin_weights[mag_sample_bin_ind]

    # Normalise to number of data points
    mag_sample_weights = mag_sample_weights * (
        sample_df.shape[0] / mag_sample_weights.sum()
    )

    # Apply a max sample weight threshold
    mag_sample_weights[
        mag_sample_weights > MAG_SAMPLE_WEIGHT_THRESHOLD
    ] = MAG_SAMPLE_WEIGHT_THRESHOLD

    sample_df["mag_weights"] = mag_sample_weights

    # Create weighted magnitude histogram
    mag_hist(sample_df.mag.values, sample_df.mag_weights.values, output_dir / "mag_weights_hist.png")

    # Compute rrup sample bins
    print("Computing Rrup sample weighting")
    count, bins = np.histogram(sample_df.rrup, bins=RRUP_N_BINS)

    rrup_sample_bin_ind = np.digitize(sample_df.rrup.values, bins)

    # Histogram includes the right bin edge, but
    # digitize does not
    rrup_sample_bin_ind[rrup_sample_bin_ind > RRUP_N_BINS] = RRUP_N_BINS

    # Digitize also indices from 1
    # (as 0 is used for values out of bounds)
    rrup_sample_bin_ind -= 1

    # Compute the bin weights
    rrup_bin_weights = 1 / count
    rrup_sample_weights = rrup_bin_weights[rrup_sample_bin_ind]

    # Normalise
    rrup_sample_weights = rrup_sample_weights * (
        sample_df.shape[0] / rrup_sample_weights.sum()
    )

    # Remove extreme sample weights
    rrup_sample_weights[
        rrup_sample_weights > RRUP_SAMPLE_WEIGHT_THRESHOLD
    ] = RRUP_SAMPLE_WEIGHT_THRESHOLD

    sample_df["rrup_weights"] = rrup_sample_weights

    # Rrup hist
    rrup_hist(sample_df.rrup.values, sample_df.rrup_weights.values, output_dir / "rrup_weights_hist.png")

    print(f"Writing the results")
    sources = np.unique(sample_df.source.values.astype(str))
    with pd.HDFStore(str(output_dir / "mag_rrup_weights.db"), mode="w") as db:
        for ix, cur_source in enumerate(sources):
            print(f"Processing {ix + 1}/{sources.size}")
            db[cur_source] = sample_df.loc[
                sample_df.source == cur_source, ["mag_weights", "rrup_weights", "rrup_mag_weights"]
            ]

    ml_tools.utils.write_to_yaml({
        "MAG_N_BINS": MAG_N_BINS,
        "MAG_SAMPLE_WEIGHT_THRESHOLD": MAG_SAMPLE_WEIGHT_THRESHOLD,
        "RRUP_N_BINS": RRUP_N_BINS,
        "RRUP_SAMPLE_WEIGHT_THRESHOLD": RRUP_SAMPLE_WEIGHT_THRESHOLD,
        "RRUP_MAG_SAMPLE_WEIGHT_THRESHOLD": RRUP_MAG_SAMPLE_WEIGHT_THRESHOLD,
    }, output_dir / "details.yaml")

if __name__ == "__main__":
    typer.run(main)
