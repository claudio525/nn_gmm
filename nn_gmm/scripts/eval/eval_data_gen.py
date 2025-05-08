import multiprocessing as mp
from pathlib import Path

import pandas
import pandas as pd
import numpy as np
import typer

import nn_gmm
from nn_gmm import console
import ml_tools
from empirical.util import openquake_wrapper_vectorized as oqw

app = typer.Typer()

PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]

# PERIODS = [
#     1.0, 3.0, 5.0, 10.0
# ]


def _get_source_sites(db_ffp: Path, source: str, valid_sites: np.ndarray):
    with pd.HDFStore(str(db_ffp), "r") as db:
        source_key = f"/{source}"
        im_data = db[source_key]
        im_data["site"] = np.stack(
            np.char.rsplit(im_data.index.values.astype(str), "_", maxsplit=1), axis=0
        )[:, -1]

        sites = np.unique(im_data.site.values.astype(str))
        sites = sites[nn_gmm.pandas_isin(sites, valid_sites)]

        return sites



@app.command("br13-predictions")
def gen_br13_predictions(
    output_dir: Path,
    site_params_ffp: Path,
    distance_dir: Path,
    imdb_dir: Path,
    source_params_dir: Path,
    train_data_dir: Path,
    val_sources_ffp: Path,
    n_procs: int = 6,
):
    val_sources = ml_tools.utils.load_txt(val_sources_ffp)

    # Check & Drop duplicates
    site_df = pd.read_csv(site_params_ffp)
    n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
    site_df = site_df.drop_duplicates()
    assert n_unique_stations == site_df.shape[0]
    site_df.set_index("station", inplace=True)
    site_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

    # Load site-source params
    console.print("Loading distance params")
    distance_db_ffps = list(distance_dir.glob("*.db"))
    distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=n_procs)
    assert (
        np.unique(distance_df.index.values.astype(str)).shape[0] == distance_df.shape[0]
    )

    print(f"Creating source lookup")
    site_source_lookup = []
    imdb_ffps = list(imdb_dir.glob("*.h5"))
    for ix, cur_ffp in enumerate(imdb_ffps):
        print(f"Processing db {ix}")
        with pd.HDFStore(str(cur_ffp), "r") as db:
            cur_sources = [cur_key.lstrip("/") for cur_key in db.keys()]

        with mp.Pool(n_procs) as p:
            cur_source_sites = p.starmap(
                _get_source_sites,
                [
                    (cur_ffp, cur_source, site_df.index.values.astype(str))
                    for cur_source in cur_sources
                ],
            )

        cur_site_source_lookup = pd.DataFrame(
            data=np.zeros((len(cur_sources), site_df.shape[0]), dtype=bool),
            columns=site_df.index,
            index=cur_sources,
        )
        for cur_source, cur_sites in zip(cur_sources, cur_source_sites):
            cur_site_source_lookup.loc[cur_source, cur_sites] = True

        site_source_lookup.append(cur_site_source_lookup)

    site_source_lookup = pd.concat(site_source_lookup, axis=0)

    # Load source params
    console.print("Loading realisation params")
    rel_df = nn_gmm.load_dfs(
        list(source_params_dir.glob("*.csv")), index_col="realisation"
    )
    rel_df["source"] = [
        split_list[0]
        for split_list in np.char.split(rel_df.index.values.astype(str), "_")
    ]
    rel_df["realisation"] = rel_df.index.values
    assert np.unique(rel_df.index.values.astype(str)).shape[0] == rel_df.shape[0]

    console.print("Computing results")
    results = []
    for ix, cur_site in enumerate(site_df.index.values):
        console.print(f"Processing site {cur_site}, {ix}/{site_df.shape[0]}")

        # Create dataframe for current site
        cur_data_df = distance_df.loc[
            distance_df.site == cur_site, ["rrup", "rjb", "rx", "source"]
        ]

        # Drop sources for which there is no IM data
        cur_sources = site_source_lookup.loc[
            site_source_lookup.loc[:, cur_site], cur_site
        ].index.values.astype(str)
        cur_data_df = cur_data_df.loc[np.isin(cur_data_df.source, cur_sources)]

        cur_data_df = pd.merge(
            rel_df.loc[:, ["realisation", "dip", "rake", "mag", "ztor", "source"]],
            cur_data_df,
            how="inner",
            left_on="source",
            right_on="source",
        )
        cur_data_df.set_index("realisation", inplace=True, drop=False)

        cur_data_df["vs30"] = site_df.loc[cur_site, "vs30"]
        cur_data_df["vs30measured"] = False
        cur_data_df["z1pt0"] = site_df.loc[cur_site, "z1p0"]
        cur_data_df["lat"] = site_df.loc[cur_site, "lat"]
        cur_data_df["lon"] = site_df.loc[cur_site, "lon"]

        # Fix the id for historic events (bit of a hack)
        cur_mask = cur_data_df.realisation.values == cur_data_df.source.values
        t = np.char.add(
            np.char.add(
                np.char.add(cur_data_df.loc[cur_mask].source.values.astype(str), "_"),
                cur_data_df.loc[cur_mask].source.values.astype(str),
            ),
            f"_{cur_site}",
        )
        cur_id = np.empty(cur_mask.shape, dtype=t.dtype)
        cur_id[cur_mask] = t
        cur_id[~cur_mask] = np.char.add(cur_data_df.index.values[~cur_mask].astype(str), f"_{cur_site}")

        cur_data_df.index = cur_id

        # Compute results
        cur_result_df = oqw.oq_run(
            oqw.GMM.Br_10, oqw.TectType.ACTIVE_SHALLOW, cur_data_df, "pSA", PERIODS
        ).astype(np.float32)
        cur_result_df = cur_result_df.loc[
            :, [col for col in cur_result_df.columns if col.endswith("_mean")]
        ]
        cur_result_df.rename(
            columns={
                col: f"{col.rsplit('_', maxsplit=1)[0]}_est"
                for col in cur_result_df.columns
            },
            inplace=True,
        )
        cur_result_df.index = cur_data_df.index

        cur_result_df["PGA_est"] = oqw.oq_run(
            oqw.GMM.Br_10, oqw.TectType.ACTIVE_SHALLOW, cur_data_df, "PGA"
        )["PGA_mean"].values.astype(np.float32)

        # Merge with inputs and add site
        cur_result_df = cur_result_df.merge(
            cur_data_df, left_index=True, right_index=True
        )
        cur_result_df["site"] = cur_site

        # Renaming
        cur_result_df.rename(
            columns={"realisation": "rupture", "source": "fault"}, inplace=True
        )
        cur_result_df.drop(columns=["vs30measured"], inplace=True)

        results.append(cur_result_df)

    results_df = pd.concat(results, axis=0)
    ims = [f"pSA_{cur_period}" for cur_period in PERIODS] + ["PGA", "PGV"]

    # Get the simulation IM values
    # train_db_path = Path(
    #     "/home/claudy/dev/work/data/nn_gmm/results/keep_runs/20220830_model_1_tuning/0830_100143_base/train_predictions.hdf5"
    # )
    # val_db_path = Path(
    #     "/home/claudy/dev/work/data/nn_gmm/results/keep_runs/20220830_model_1_tuning/0830_100143_base/val_predictions.hdf5"
    # )
    # val_df = nn_gmm.ResultDB.get_data_static(
    #     val_db_path, columns=ims
    # )
    # train_df = nn_gmm.ResultDB.get_data_static(
    #     train_db_path, columns=ims
    # )
    #
    # val_df = val_df.merge(results_df, how="left", left_index=True, right_index=True)
    # train_df = train_df.merge(results_df, how="left", left_index=True, right_index=True)

    # print("wtf")
    # results_df = pd.concat([train_df, val_df], axis=0)
    # del val_df, train_df
    # print(results_df.dtypes)

    feature_details = nn_gmm.load_feature_details(train_data_dir / "train")
    for ix, cur_source in enumerate(site_source_lookup.index.values):
        print(f"{ix}/{site_source_lookup.shape[0]}")
        if cur_source in val_sources:
            cur_df = nn_gmm.load_tfrecord(
                train_data_dir / "val" / f"{cur_source}.tfrecord", feature_details
            )
        else:
            cur_df = nn_gmm.load_tfrecord(
                train_data_dir / "train" / f"{cur_source}.tfrecord", feature_details
            )

        # print(cur_df.dtypes)

        try:
            results_df.loc[cur_df.index.values, ims] = cur_df.loc[:, ims].values.astype(np.float32)
        except KeyError:
            continue

        # print(results_df.dtypes)
        # print("wtf")

    # Writing results
    val_mask = nn_gmm.pandas_isin(results_df.fault, val_sources)
    nn_gmm.ResultDB.write_data(
        results_df.loc[val_mask], output_dir / "val_predictions.hdf5"
    )
    nn_gmm.ResultDB.write_data(
        results_df.loc[~val_mask], output_dir / "train_predictions.hdf5"
    )


if __name__ == "__main__":
    app()
