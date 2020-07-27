"""Creates a csv from the specified tfrecord files"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd

import nn_gmm


def main(
    data_dirs: List[str], output_ffp: str, file_filter: str = "*.tfrecord"
):
    dfs = []
    for cur_data_dir in data_dirs:
        cur_data_dir = Path(cur_data_dir)
        cur_ds = nn_gmm.load_dataset(
            cur_data_dir,
            nn_gmm.load_feature_details(cur_data_dir),
            10000,
            file_filter=file_filter,
            shuffle_buffer=None,
            block_size=30000,
        )

        for data in cur_ds.as_numpy_iterator():
            dfs.append(pd.DataFrame.from_dict(data))

    df = pd.concat(dfs)
    df.id = df.id.str.decode("UTF-8")
    df = df.set_index("id")

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
    parser.add_argument(
        "-r",
        "--recursive",
        help="If set then the data directories are searched recursively",
        action="store_true",
    )

    args = parser.parse_args()

    main(args.data_dirs, args.output_ffp, file_filter=args.file_filter)
