"""Script for collectiong stats required for preprocessing, such as min, max,
mean and standard deviation.

This script is pretty slow and not really ideal, however given that this
will only be required in-frequently it will do for now.
"""
import glob
import time
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf


def main(data_dir: Path, glob_filter: str, feature_details_ffp: Path, output_ffp):
    data_files = glob.glob(str(data_dir / glob_filter))

    with feature_details_ffp.open("rb") as f:
        feature_description = pickle.load(f)

    raw_dataset = tf.data.TFRecordDataset(list(data_files))

    def _parse_fn(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed

    parsed_dataset = raw_dataset.map(_parse_fn)

    def _get_stats(cur_state, item):
        cur_state["count"] += 1
        for key in item.keys():
            cur_state["min"][key] = cur_state["min"][key] if cur_state["min"][key] > item[
                key] else item[key]
            cur_state["max"][key] = cur_state["min"][key] if cur_state["min"][key] < item[
                key] else item[key]
            cur_state["sum"][key] += item[key]

        return cur_state

    initial_state = {"count": 0, "min": {}, "max": {}, "sum": {}}
    std_initial_state = {}
    for item in parsed_dataset.take(1):
        for key in item.keys():
            initial_state["min"][key] = -99999.0
            initial_state["max"][key] = 99999.0
            initial_state["sum"][key] = 0.0
            std_initial_state[key] = 0.0

    start_time = time.time()
    stats = parsed_dataset.prefetch(tf.data.experimental.AUTOTUNE).reduce(initial_state,
                                                                          _get_stats)
    print(f"Took {time.time() - start_time}")
    # stats = initial_state

    count = stats["count"].numpy()
    del stats["count"]

    stats_df = pd.DataFrame.from_dict(stats)
    stats_df = stats_df.applymap(lambda t: t.numpy())
    stats_df["mean"] = stats_df["sum"] / count

    def _get_sigma_sum(cur_state, item):
        for key in item.keys():
            cur_state[key] += tf.math.pow(item[key] - stats_df.loc[key, "mean"],
                                          tf.constant(2, dtype=tf.float32))

        return cur_state

    start_time = time.time()
    std_sum = parsed_dataset.prefetch(tf.data.experimental.AUTOTUNE).reduce(
        std_initial_state, _get_sigma_sum)
    print(f"Took {time.time() - start_time}")

    std_sum_df = pd.Series(std_sum).to_frame("std_sum")
    std_sum_df = std_sum_df.applymap(lambda t: t.numpy())
    std_sum_df["std"] = np.sqrt(std_sum_df["std_sum"] / (count - 1))

    stats_df = pd.merge(stats_df, std_sum_df, how="left", left_index=True,
                        right_index=True)
    stats_df.to_csv(output_ffp, index_label="feature")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Directoy that contains the tfrecord files", type=Path
    )
    parser.add_argument(
        "feature_details_ffp",
        help="Path to the feature details pickle file, "
        "as produced by create_sample_files.py",
        type=Path
    )
    parser.add_argument(
        "output_ffp", help="File path of the output csv file", type=Path
    )
    parser.add_argument(
        "--glob_filter", help="The glob filter to use", type=str, default="*.tfrecord"
    )

    args = parser.parse_args()

    main(args.data_dir, args.glob_filter, args.feature_details_ffp, args.output_ffp)
