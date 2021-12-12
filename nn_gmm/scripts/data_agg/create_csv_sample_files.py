"""Creates csv sample files, one per source"""
from pathlib import Path

import numpy as np
import pandas as pd

import nn_gmm
from nn_gmm import console

IMs = np.asarray(
    [
        "AI",
        "CAV",
        "Ds575",
        "Ds595",
        "MMI",
        "PGA",
        "PGV",
        "pSA_0.01",
        "pSA_0.02",
        "pSA_0.03",
        "pSA_0.04",
        "pSA_0.05",
        "pSA_0.075",
        "pSA_0.1",
        "pSA_0.12",
        "pSA_0.15",
        "pSA_0.17",
        "pSA_0.2",
        "pSA_0.25",
        "pSA_0.3",
        "pSA_0.4",
        "pSA_0.5",
        "pSA_0.6",
        "pSA_0.7",
        "pSA_0.75",
        "pSA_0.8",
        "pSA_0.9",
        "pSA_1.0",
        "pSA_1.25",
        "pSA_1.5",
        "pSA_2.0",
        "pSA_2.5",
        "pSA_3.0",
        "pSA_4.0",
        "pSA_5.0",
        "pSA_6.0",
        "pSA_7.5",
        "pSA_10.0",
    ]
)

n_procs = 8

site_params_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/site_params")
site_params_ffps = list(site_params_dir.glob("*.csv"))

distance_data_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/distance_data")
distance_db_ffps = list(distance_data_dir.glob("*.db"))

site_source_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/site_source_data")
site_source_ffps = list(site_source_dir.glob("*.db"))

rel_source_params_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/source_rel_params")
rel_source_params_ffps = list(rel_source_params_dir.glob("*.csv"))

imdb_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/im_data")
imdb_ffps = list(imdb_dir.glob("*.h5"))

# val_events_ffp = Path("/home/claudy/dev/work/data/nn_gmm/validation_events.txt")

output_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/sample_files/20211122_csv_all_files")

tect_type_one_hot_dict = nn_gmm.TECT_TYPE_ONE_HOT_DICT

# Load the data

# Site
site_df = nn_gmm.load_dfs(site_params_ffps)
n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
site_df = site_df.drop_duplicates()
assert n_unique_stations == site_df.shape[0]
site_df.set_index("station", inplace=True)

# Site-source params
console.log("Loading distance params")
distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=n_procs)

console.log("Loading site-source params")
site_source_df = nn_gmm.load_site_source_df(site_source_ffps, n_procs=n_procs)
assert (
    np.unique(site_source_df.index.values.astype(str)).shape[0]
    == site_source_df.shape[0]
)

# Source params
console.log("Loading realisation params")
rel_df = nn_gmm.load_dfs(
    list(rel_source_params_dir.glob("*.csv")), index_col="realisation"
)
rel_df["source"] = [
    split_list[0]
    for split_list in np.char.split(rel_df.index.values.astype(str), "_")
]

# Generate CSV files
sources = np.unique(rel_df.source.values.astype(str))

for ix, cur_source in enumerate(sources):
    console.log(f"Processing {ix+1}/{sources.size}")
    cur_output_ffp = output_dir / f"{cur_source}.csv"
    if cur_output_ffp.exists():
        console.log(f"[orange]Skipping {cur_source} as output file already exists[/]")
        continue

    cur_im_df = nn_gmm.load_fault_im_df(cur_source, imdb_ffps)
    if cur_im_df is None:
        console.log(f"[orange]No IM data found for source {cur_source}, skipping[/]")
        continue
    cur_im_df.sort_index(inplace=True)

    # Create sample combinations
    cur_input_df = nn_gmm.create_sample_comb(cur_im_df)
    cur_input_df["id"] = cur_input_df.index.values.astype(str)

    # Merge with realisation source parameters
    cur_input_df = pd.merge(
        cur_input_df,
        rel_df,
        how="inner",
        left_on="realisation",
        right_index=True,
        suffixes=(None, "_rel_df"),
    )

    # Merge with site parameters
    cur_input_df = pd.merge(
        cur_input_df, site_df, how="inner", left_on="site", right_index=True
    )

    # Merge with site-source parameters
    # cur_input_df = pd.merge(
    #     cur_input_df,
    #     site_source_df.loc[:, ["theta", "s"]],
    #     how="inner",
    #     left_index=True,
    #     right_index=True,
    # )

    # Merge with distance parameters
    cur_input_df = pd.merge(
        cur_input_df,
        distance_df.loc[distance_df.source == cur_source],
        how="inner",
        left_on=["source", "site"],
        right_on=["source", "site"],
    )
    cur_input_df.set_index("id", inplace=True)
    cur_input_df.sort_index(inplace=True)
    cur_input_df = cur_input_df.drop(
        columns=["realisation", "source", "source_rel_df", "site", "rtvz"]
    )
    cur_input_df = nn_gmm.apply_one_hot_enc(
        cur_input_df, "tect_type", tect_type_one_hot_dict
    )

    cur_im_df = nn_gmm.interpolate_pSA_periods(cur_im_df, IMs)

    # Combine and save
    cur_input_df.sort_index(inplace=True)
    cur_im_df.sort_index(inplace=True)
    assert np.all(cur_input_df.index == cur_im_df.index)
    cur_sample_df = pd.concat((cur_input_df, cur_im_df), axis=1)

    cur_sample_df.to_csv(cur_output_ffp, index_label="id")
