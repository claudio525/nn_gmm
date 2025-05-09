"""Combines the site parames from station list and vs30 into a single csv file,
also adds Z1.0 & Z2.5"""
import os
import argparse

import pandas as pd
import numpy as np

import qcore.formats as formats


def main(
    site_loc_ffp: str,
    site_vs30_ffp: str,
    site_vs500_ffp: str,
    site_Z1p0_ffp: str,
    site_Z2p5_ffp: str,
    output_ffp: str,
):
    if os.path.isfile(output_ffp):
        print("The output file already exist, quitting!")
        exit()

    loc_df = formats.load_station_file(site_loc_ffp)
    vs30_df = formats.load_vs30_file(site_vs30_ffp)
    vs500_df = pd.read_csv(site_vs500_ffp, index_col=0)
    z1p0_df = pd.read_csv(site_Z1p0_ffp, index_col=0)
    z2p5_df = pd.read_csv(site_Z2p5_ffp, index_col=0)

    df = loc_df.merge(vs30_df, how="inner", left_index=True, right_index=True)
    df = df.merge(vs500_df.loc[:, "vs500"], how="inner", left_index=True, right_index=True)
    df = df.merge(z1p0_df.loc[:, "z1p0"], how="inner", left_index=True, right_index=True)
    df = df.merge(z2p5_df.loc[:, "z2p5"], how="inner", left_index=True, right_index=True)

    assert df.shape[0] == loc_df.shape[0]

    df.to_csv(output_ffp, index=True, index_label="station")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("site_loc_ffp", type=str, help="Station lon/lat list file path")
    parser.add_argument("site_vs30_ffp", type=str, help="Station vs30 file path")
    parser.add_argument("site_vs500_ffp", type=str, help="Station vs500 file path")
    parser.add_argument("site_Z1p0_ffp", type=str, help="Station Z1.0 file path")
    parser.add_argument("site_Z2p5_ffp", type=str, help="Station Z2.5 file path")
    parser.add_argument("output_ffp", type=str, help="Output csv file path")

    args = parser.parse_args()

    main(
        args.site_loc_ffp,
        args.site_vs30_ffp,
        args.site_vs500_ffp,
        args.site_Z1p0_ffp,
        args.site_Z2p5_ffp,
        args.output_ffp,
    )
