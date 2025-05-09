"""Adds the Z1.0 variations for the specified site_params.csv"""
from pathlib import Path

import pandas as pd
import numpy as np

import nn_gmm

def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30**8.0 + 378.7**8.0)) / 1000.0
    )  # CY08 estimate in KM

input_ffp = Path("/home/claudy/dev/work/data/nn_gmm/input_data/site_params/old/20220629_site_params.csv")
basin_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/raw_data/site_data/basin_stations")

output_ffp = Path("/home/claudy/dev/work/data/nn_gmm/input_data/site_params/site_params.csv")

site_df = pd.read_csv(input_ffp, index_col=0)

basin_stations = nn_gmm.load_basin_stations(basin_dir)

# Add the vs30 based Z1.0
site_df["est_z1p0"] = estimate_z1p0(site_df.vs30)

# Use vs30 based for outside basin stations
# and use VM Z1.0 for basin stations
site_df["comb_z1p0"] = site_df.est_z1p0

for cur_basin, cur_stations in basin_stations.items():
    site_df.loc[cur_stations, "comb_z1p0"] = site_df.loc[cur_stations, "z1p0"].values

# Z1.0 Delta
site_df["delta_z1p0"] = site_df.z1p0 - site_df.est_z1p0

# Combined Z1.0
# 0 everwhere, except for the basins
site_df["comb_delta_z1p0"] = 0

for cur_basin, cur_stations in basin_stations.items():
    site_df.loc[cur_stations, "comb_delta_z1p0"] = site_df.loc[cur_stations, "delta_z1p0"].values


site_df.to_csv(output_ffp)

