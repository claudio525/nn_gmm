import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from nn_gmm import agg_utils


def main(
    site_params_ffp: str,
    site_source_ffp: str,
    source_params_ffp: str,
    im_db_ffp: str,
    output_dir: str,
    n_procs: int = 4,
):
    output_dir = Path(output_dir)

    # Load Site params
    print("Loading site params")
    site_df = pd.read_csv(site_params_ffp, index_col="station")

    # Load site-source params
    print("Loading site-source params")
    site_source_df = agg_utils.load_site_source_df(
        site_df, site_source_ffp, n_procs=n_procs
    )

    # Load source params
    print("Loading source params")
    source_df = pd.read_csv(source_params_ffp, index_col="fault")
    tect_type_one_hot_dict = {cur_type: cur_type.lower() for cur_type in np.unique(source_df.tect_type)}

    for ix, cur_source in enumerate(source_df.index.values.astype(str)):
        print(f"Processing {ix + 1}/{source_df.shape[0]}")
        cur_im_df = agg_utils.load_fault_im_df(cur_source, im_db_ffp)
        if cur_im_df is None:
            print(f"No IM data found for source {cur_source}, skipping.")
            continue
        cur_im_df.sort_index(inplace=True)

        cur_input_df = agg_utils.create_sample_comb(cur_im_df)
        cur_input_df = pd.merge(
            cur_input_df, source_df, how="inner", left_on="source", right_index=True
        )
        cur_input_df = pd.merge(
            cur_input_df, site_df, how="inner", left_on="site", right_index=True
        )
        cur_input_df["id"] = cur_input_df.index.values.astype(str)

        cur_input_df = pd.merge(
            cur_input_df,
            site_source_df,
            how="inner",
            left_on=["source", "site"],
            right_on=["source", "site"],
        )
        cur_input_df.set_index("id", inplace=True)
        cur_input_df.sort_index(inplace=True)
        cur_input_df = cur_input_df.drop(columns=["id", "source", "site"])
        cur_input_df = agg_utils.apply_one_hot_enc(cur_input_df, "tect_type", tect_type_one_hot_dict)

        assert np.all(cur_input_df.index.values.astype(str) == cur_im_df.index.values.astype(str))

        cur_input_df.to_csv(output_dir / f"{cur_source}_inputs.csv", index_label="id")
        cur_im_df.to_csv(output_dir / f"{cur_source}_ims.csv", index_label="id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "site_params_ffp", type=str, help="The path to the site params csv file to use"
    )
    parser.add_argument(
        "site_source_ffp", type=str, help="The path to the site-source db to use"
    )
    parser.add_argument(
        "source_params_ffp",
        type=str,
        help="The path to the source params csv file to use",
    )
    parser.add_argument("im_db_ffp", type=str, help="The path to the IM labels db")
    parser.add_argument("output_dir", type=str, help="Path of the output directory")
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )

    args = parser.parse_args()

    main(
        args.site_params_ffp,
        args.site_source_ffp,
        args.source_params_ffp,
        args.im_db_ffp,
        args.output_dir,
        n_procs=args.n_procs,
    )
