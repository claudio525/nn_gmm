import time
import argparse

import pandas as pd
import numpy as np

from nn_gmm import agg_utils


def main(
    site_params_ffp: str,
    site_source_ffp: str,
    source_params_ffp: str,
    im_db_ffp: str,
    output_ffp: str,
    n_procs: int = 4,
):
    # Load Site params
    print("Loading site params")
    site_df = pd.read_csv(site_params_ffp, index_col="station")

    # Load site-source params
    print("Loading site-source params")
    site_source_df = agg_utils.load_site_source_dict(
        site_df, site_source_ffp, n_procs=n_procs
    )

    # Load source params
    print("Loading source params")
    source_df = pd.read_csv(source_params_ffp, index_col="fault")

    # Load IM data
    print("Loading IM data")
    im_df = agg_utils.load_im_dict(im_db_ffp, n_procs=n_procs)
    im_df.sort_index(inplace=True)

    # Create all possible sample combinations based on the available labels
    print("Creating possible sample combinations")
    sample_combs = agg_utils.create_sample_comb(im_df)

    # Create the input dataframe (X)
    input_df = sample_combs.copy()
    input_df = pd.merge(
        input_df, source_df, how="inner", left_on="source", right_index=True
    )
    input_df = pd.merge(
        input_df, site_df, how="inner", left_on="site", right_index=True
    )
    input_df["id"] = input_df.index.values.astype(str)

    input_df = pd.merge(
        input_df,
        site_source_df,
        how="inner",
        left_on=["source", "site"],
        right_on=["source", "site"],
    )
    input_df.set_index("id", inplace=True)
    input_df.sort_index(inplace=True)

    assert np.all(input_df.index.values.astype(str) == im_df.index.values.astype(str))

    # Get all float columns for conversion to np.float32
    float_cols = {
        col: np.float32 for col in input_df.columns if input_df[col].dtype == float
    }

    print(f"Writing to {output_ffp}")
    with pd.HDFStore(output_ffp) as store:
        store["X"] = input_df.astype(float_cols)
        store["y"] = im_df.astype(np.float32)


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
    parser.add_argument("output_ffp", type=str, help="Path for the output h5")
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )

    args = parser.parse_args()

    main(
        args.site_params_ffp,
        args.site_source_ffp,
        args.source_params_ffp,
        args.im_db_ffp,
        args.output_ffp,
        n_procs=args.n_procs,
    )
