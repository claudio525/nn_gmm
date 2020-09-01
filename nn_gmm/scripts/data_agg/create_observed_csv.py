import multiprocessing as mp
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

import nn_gmm

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

def _process_im_file(im_csv_ffp: Path):
    cur_fault = im_csv_ffp.name.split(".")[0]
    cur_df = pd.read_csv(im_csv_ffp, index_col="station")

    cur_df = nn_gmm.interpolate_pSA_periods(cur_df, IMs)
    cur_df.set_index(np.char.add(f"{cur_fault}_{cur_fault}_", cur_df.index.values.astype(str)), inplace=True)
    cur_df.drop(columns=["component"], inplace=True)

    return cur_df

def main(data_dirs: List[Path], output_ffp: Path, n_procs: int = 4):
    dfs = []
    for cur_dir in data_dirs:
        im_files = cur_dir.glob("*.csv")

        if n_procs == 1:
            for im_csv_ffp in im_files:
                dfs.append(_process_im_file(im_csv_ffp))
        else:
            with mp.Pool(processes=n_procs) as pool:
                dfs = pool.map(_process_im_file, [im_csv_ffp for im_csv_ffp in im_files])

    df = pd.concat(dfs)
    df.to_csv(output_ffp, index_label="id")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_ffp", type=str, help="Output file path")
    parser.add_argument(
        "--im_data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="The directories that contain the IM csv files for the observed records",
    )
    parser.add_argument("--n_procs", type=int, help="Number of processes to use", default=4)

    args = parser.parse_args()

    main(nn_gmm.to_path(args.im_data_dirs), Path(args.output_ffp), args.n_procs)

