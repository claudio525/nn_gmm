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
    site_source_dict = agg_utils.load_site_source_dict(
        site_df, site_source_ffp, n_procs=n_procs
    )

    # Load source params
    print("Loading source params")
    source_df = pd.read_csv(source_params_ffp, index_col="fault")

    # Load IM data
    print("Loading IM data")
    im_dict = agg_utils.load_im_dict(im_db_ffp, n_procs=n_procs)

    # Create all possible sample combinations based on the available labels
    print("Creating possible sample combinations")
    sample_combs = agg_utils.create_sample_comb(im_dict)

    # Reduce sample set based on the input data availability
    print("Dropping samples with missing data")
    sample_combs = agg_utils.drop_missing_data(
        sample_combs, site_df, source_df, site_source_dict
    )

    input_samples = {}
    labels = {}

    start_time = time.time()
    print("Creating samples")
    for cur_id, cur_row in sample_combs.iterrows():
        cur_site_params = site_df.loc[cur_row.site].values
        cur_source_params = source_df.loc[cur_row.source].values
        cur_site_source_params = site_source_dict[cur_row.site].loc[cur_row.source].values
        cur_im_data = im_dict[cur_row.source].loc[cur_row.site].values

        input_samples[cur_id] = np.concatenate(
            [cur_site_params, cur_source_params, cur_site_source_params]
        )
        labels[cur_id] = cur_im_data

        continue
    print(f"Took {time.time() - start_time} seconds")

    print("Creating dataframe")
    input_df = pd.DataFrame.from_dict(
        input_samples,
        orient="index",
        columns=np.concatenate(
            [
                site_df.columns.values.astype(str),
                source_df.columns.values.astype(str),
                site_source_dict[cur_station].loc[cur_source].index.values.astype(str),
            ]
        ),
    )
    labels_df = pd.DataFrame.from_dict(
        labels,
        orient="index",
        columns=im_dict[cur_source].loc[cur_station].index.values.astype(str),
    )

    print(f"Writing to {output_ffp}")
    with pd.HDFStore(output_ffp) as store:
        store["X"] = input_df
        store["y"] = labels_df


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
    parser.add_argument("--n_procs", type=int, help="Number of processes to use", default=4)

    args = parser.parse_args()

    main(
        args.site_params_ffp,
        args.site_source_ffp,
        args.source_params_ffp,
        args.im_db_ffp,
        args.output_ffp,
        n_procs=args.n_procs,
    )
