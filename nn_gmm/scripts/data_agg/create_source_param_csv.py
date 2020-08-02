"""Creates a source attribute csv file for a given Cybershake sources directory
Note: This is all done at the fault level, i.e. assumes all attributes retrieved
are the same across all realisation (Todo: need to check this..)
"""
import os
import glob
import argparse
import multiprocessing as mp

import h5py
import pandas as pd
import numpy as np


def get_fault_data(sources_dir: str, fault_name: str):
    # Get the first info file
    info_file = glob.glob(os.path.join(sources_dir, fault_name, "Srf", "*.info"))[0]

    with h5py.File(info_file, "r") as f:
        tect_type = (
            f.attrs["tect_type"] if "tect_type" in f.attrs.keys() else "ACTIVE_SHALLOW"
        )
        return (
            fault_name,
            [
                np.mean(f.attrs["dip"]),
                np.mean(f.attrs["rake"]),
                np.mean(f.attrs["strike"]),
                np.mean(f.attrs["dtop"]),
                np.mean(f.attrs["dbottom"]),
                np.min(f.attrs["dtop"]),  # Ztor
                np.max(f.attrs["dtop"]),  # Zbot
                f.attrs["mag"],
                tect_type,
                f.attrs["hdepth"],
                f.attrs["hlon"],
                f.attrs["hlat"],
                # f.attrs["dhyp"],
                # f.attrs["shyp"],
                np.mean(f.attrs["width"]),
                np.mean(f.attrs["length"]),
                f.attrs["type"] == 1,
            ],
        )


def main(sources_dir: str, output_ffp: str, n_procs: int = 4):
    if os.path.isfile(output_ffp):
        print("The output file already exist, quitting!")
        exit()

    # Get all faults
    faults = np.asarray(next(os.walk(sources_dir))[1])

    # Collect data for each fault
    if n_procs == 1:
        fault_src_data = [
            get_fault_data(sources_dir, cur_fault) for cur_fault in faults
        ]
    else:
        with mp.Pool(processes=n_procs) as p:
            fault_src_data = p.starmap(
                get_fault_data, [(sources_dir, cur_fault) for cur_fault in faults]
            )

    # Combine and create a dataframe
    fault_src_dict = {name: src_attrs for name, src_attrs in fault_src_data}
    src_df = pd.DataFrame.from_dict(
        fault_src_dict,
        orient="index",
        columns=[
            "dip",
            "rake",
            "strike",
            "dtop",
            "dbottom",
            "ztor",
            "zbot",
            "mag",
            "tect_type",
            "hdepth",
            "hlon",
            "hlat",
            # "dhyp",
            # "shyp",
            "width",
            "length",
            "is_point_source"
        ],
    )

    # Write dataframe
    src_df.to_csv(output_ffp, index=True, index_label="fault")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sources_dir", type=str, help="The cybershake sources directory"
    )
    parser.add_argument(
        "output_ffp", type=str, help="The output file path for the source params csv"
    )
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )

    args = parser.parse_args()

    main(args.sources_dir, args.output_ffp, n_procs=args.n_procs)
