"""Combines the site parames from station list and vs30 into a single csv file,
also adds Z1.0 & Z2.5"""
import os
import argparse

import pandas as pd
import numpy as np

import qcore.formats as formats
from empirical.util.classdef import estimate_z1p0, estimate_z2p5


def main(site_loc_ffp: str, site_vs30_ffp: str, output_ffp: str):
    if os.path.isfile(output_ffp):
        print("The output file already exist, quitting!")
        exit()

    loc_df = formats.load_station_file(site_loc_ffp)
    vs30_df = formats.load_vs30_file(site_vs30_ffp)

    df = loc_df.merge(vs30_df, how="left", left_index=True, right_index=True)

    # Compute Z1.0 & Z2.5
    df["z1p0"] = estimate_z1p0(df.vs30.values)
    df["z2p5"] = estimate_z2p5(z1p0=df.z1p0.values)

    df.to_csv(output_ffp, index=True, index_label="station")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("site_loc_ffp", type=str, help="Station lon/lat list file path")
    parser.add_argument("site_vs30_ffp", type=str, help="Station vs30 file path")
    parser.add_argument("output_ffp", type=str, help="Output csv file path")

    args = parser.parse_args()

    main(args.site_loc_ffp, args.site_vs30_ffp, args.output_ffp)
