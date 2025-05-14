from pathlib import Path
import logging

import pandas as pd
import numpy as np
import typer
from tqdm import tqdm

import ml_tools as mlt
import nn_gmm as nng

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)

app = typer.Typer()


@app.command("create-db")
def create_db(
    db_ffp: Path = typer.Argument(..., help="Path to the database file"),
    im_data_dir: Path = typer.Argument(
        ..., help="Path to the Cybershake IM data directory"
    ),
    source_info_dir: Path = typer.Argument(
        ..., help="Path to the Cybershake source info directory"
    ),
    ll_ffp: Path = typer.Argument(..., help="Path to the Cybershake site data file"),
    vs30_ffp: Path = typer.Argument(..., help="Path to the Cybershake vs30 data file"),
    z_ffp: Path = typer.Argument(..., help="Path to the Cybershake Z data file"),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
):
    """
    Create a new database.
    """
    SOURCE_INFO_FIELDS = [
        "magnitude",
        "type",
        "fault_type",
        "tect_type",
        "rake",
        "dip",
        "dtop",
        "dbottom",
        "length",
        "plane_count",
        "dip_dir",
        "shypo",
        "dhypo",
    ]

    SOURCE_TO_DB_COLUMNS_MAPPING = {
        "type": "sim_type",
    }

    logging.getLogger().setLevel(log_level)
    logging.info(f"Logging level set to {log_level}")

    if db_ffp.exists():
        logging.info(f"Database {db_ffp} already exists. Exiting.")
        return

    site_df = pd.read_csv(
        ll_ffp, sep=r"\s+", names=["lon", "lat", "site_id"], index_col="site_id"
    ).sort_index()
    vs30_df = pd.read_csv(
        vs30_ffp, sep=r"\s+", names=["site_id", "vs30"], index_col="site_id"
    ).sort_index()
    z_df = pd.read_csv(
        z_ffp,
        index_col=0,
        header=0,
        names=["site_id", "z1p0", "z2p5", "sigma"],
        usecols=[0, 1, 2],
    ).sort_index()

    site_df.loc[:, vs30_df.columns] = vs30_df.values
    site_df.loc[:, z_df.columns] = z_df.values

    # Add site data
    with nng.imdb.IMDB(db_ffp) as db:
        db.add_site_data(site_df)

    # Add event, realisation and IM data
    im_events = np.sort([cur_dir.stem for cur_dir in im_data_dir.iterdir() if cur_dir.is_dir()])
    source_events = np.sort([
        cur_dir.stem for cur_dir in source_info_dir.iterdir() if cur_dir.is_dir()
    ])
    assert np.all(im_events == source_events), "IM and source events do not match!"
    events = im_events

    for cur_event in tqdm(events, desc="Processing events"):
        logging.debug(f"Processing event: {cur_event}")
        source_dir = source_info_dir / cur_event / "Srf"

        # Read the median data
        median_info = pd.read_csv(source_dir / f"{cur_event}.csv").squeeze()
        median_info = median_info.loc[SOURCE_INFO_FIELDS]
        median_info = median_info.rename(SOURCE_TO_DB_COLUMNS_MAPPING, axis=0)

        # Read the realisation data
        rel_infos = []
        for cur_rel_ffp in source_dir.glob("*REL*.csv"):
            cur_rel_id = f"{cur_event}_{cur_rel_ffp.stem.rsplit('_', maxsplit=1)[-1]}"
            cur_rel_df = pd.read_csv(cur_rel_ffp)
            cur_rel_df.index = [cur_rel_id]

            rel_infos.append(cur_rel_df)

        rel_df = pd.concat(rel_infos, axis=0).rename(
            columns=SOURCE_TO_DB_COLUMNS_MAPPING
        )
        rel_df["event_id"] = cur_event

        # Read the IM data
        im_files = list((im_data_dir / cur_event / "IM").rglob("*REL*.csv"))
        rel_im_dfs = []
        for cur_rel_ffp in im_files:
            cur_rel_id = f"{cur_event}_{cur_rel_ffp.stem.rsplit('_', maxsplit=1)[-1]}"

            if cur_rel_id not in rel_df.index:
                logging.warning(
                    f"Realisation {cur_rel_id} not found in source data, but exists in IM data. Skipping!"
                )
                continue

            cur_im_df = pd.read_csv(cur_rel_ffp, index_col=0)[nng.constants.IMS]
            cur_im_df["site_id"] = cur_im_df.index
            cur_im_df["event_id"] = cur_event
            cur_im_df["rel_id"] = cur_rel_id
            cur_im_df.index = mlt.array_utils.numpy_str_join(
                "_", cur_rel_id, cur_im_df.index.values.astype(str)
            )

            rel_im_dfs.append(cur_im_df)

        im_df = pd.concat(rel_im_dfs, axis=0)

        with nng.imdb.IMDB(db_ffp) as db:
            db.add_event_data(cur_event, median_info)
            db.add_realisation_data(rel_df)
            db.add_record_im_data(im_df)

    logging.info(f"Database {db_ffp} created successfully.")

if __name__ == "__main__":
    app()
