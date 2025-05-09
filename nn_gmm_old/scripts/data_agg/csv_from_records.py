"""Creates a csv from the specified tfrecord files"""
import pickle
import argparse
from pathlib import Path
from typing import List

import pandas as pd

import nn_gmm


def main(
    data_dirs: List[Path], output_ffp: Path, file_filter: str = "*.tfrecord", event_list_ffp: Path = None
):
    # Load the events of interest if specified
    events = None
    if event_list_ffp is not None:
        with open(event_list_ffp, "r") as f:
            events = [line.strip() for line in f.readlines()]

    # Feature details (have to be same across all specified data dirs)
    with open(data_dirs[0] / "feature_details.pickle", "rb") as f:
        feature_details = pickle.load(f)

    dfs, n_processed = [], 0
    for cur_data_dir in data_dirs:
        tfrecord_files = cur_data_dir.glob(file_filter)
        n_files = len(events) if events is not None else len(tfrecord_files)
        for cur_ffp in tfrecord_files:
            if events is not None and cur_ffp.name.split(".")[0] not in events:
                continue

            cur_df = nn_gmm.load_tfrecord(str(cur_ffp), feature_details)
            dfs.append(cur_df)

            n_processed += 1
            print(f"Processed {n_processed}/{n_files}")

    df = pd.concat(dfs)

    df.to_csv(output_ffp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="The data directories from which to load .tfrecords, "
        "expects a feature_details.pickle in each directory",
    )
    parser.add_argument(
        "--output_ffp", required=True, help="File path for the resulting csv file"
    )
    parser.add_argument(
        "--file_filter", help="The glob filter to use", default="*.tfrecord"
    )
    parser.add_argument("--event_list_ffp", type=str, help="File that contains the events to export to the csv", default=None)
    parser.add_argument(
        "-r",
        "--recursive",
        help="If set then the data directories are searched recursively",
        action="store_true",
    )

    args = parser.parse_args()

    main(nn_gmm.to_path(args.data_dirs), nn_gmm.to_path(args.output_ffp), file_filter=args.file_filter, event_list_ffp=nn_gmm.to_path(args.event_list_ffp))
