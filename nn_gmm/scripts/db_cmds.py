from pathlib import Path
import logging

import pandas as pd
import numpy as np
import typer
from tqdm import tqdm

import ml_tools as mlt
import nn_gmm as nng
import seismic_hazard_analysis as sha
from qcore import coordinates as coords
from qcore import nhm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

app = typer.Typer()


@app.command("create-imdb")
def create_imdb(
    db_ffp: Path = typer.Argument(..., help="Path to the database file"),
    im_data_dir: Path = typer.Argument(
        ..., help="Path to the Cybershake IM data directory"
    ),
    source_info_dir: Path = typer.Argument(
        ..., help="Path to the Cybershake source info directory"
    ),
    nhm_flt_ffp: Path = typer.Argument(
        ..., help="Path to the NHM 2010 fault data file"
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

    logger = nng.utils.setup_logging(console_level=log_level)
    logger.info(f"Logging level set to {log_level}")

    if db_ffp.exists():
        logger.info(f"Database {db_ffp} already exists. Exiting.")
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

    site_grid_level = nng.utils.get_site_grid_level(site_df.index.values.astype(str))
    site_df["grid_level"] = site_grid_level

    nztm_coords = coords.wgs_depth_to_nztm(site_df[["lat", "lon"]].values)[:, ::-1]
    site_df["nztm_x"], site_df["nztm_y"] = nztm_coords[:, 0], nztm_coords[:, 1]

    # Add site data
    with nng.DuckIMDB(db_ffp) as db:
        db.add_site_data(site_df)

    # Get events (and sanity check)
    im_events = np.sort(
        [cur_dir.stem for cur_dir in im_data_dir.iterdir() if cur_dir.is_dir()]
    )
    source_events = np.sort(
        [cur_dir.stem for cur_dir in source_info_dir.iterdir() if cur_dir.is_dir()]
    )
    assert np.all(im_events == source_events), "IM and source events do not match!"
    events = im_events

    # Compute the site to source distances
    flt_definitions = nhm.load_nhm(nhm_flt_ffp)
    faults = {
        cur_name: sha.nshm_2010.utils.get_fault_objects(cur_fault)
        for cur_name, cur_fault in flt_definitions.items()
        if cur_name in events
    }
    logging.info("Calculating site to source distances...")
    site_event_df = nng.utils.run_site_to_source_calc(faults, site_df)

    logging.info("Adding event and realisation data to the database...")
    event_data, rel_data = [], []
    with nng.DuckIMDB(db_ffp) as db:
        # Add event, realisation and IM data
        for cur_event in tqdm(events, desc="Processing events"):
            source_dir = source_info_dir / cur_event / "Srf"

            # Read the median data
            median_info = pd.read_csv(source_dir / f"{cur_event}.csv").squeeze()
            median_info = median_info.loc[SOURCE_INFO_FIELDS]
            median_info = median_info.rename(SOURCE_TO_DB_COLUMNS_MAPPING, axis=0)

            # Read the realisation data
            rel_infos = []
            for cur_rel_ffp in source_dir.glob("*REL*.csv"):
                cur_rel_id = (
                    f"{cur_event}_{cur_rel_ffp.stem.rsplit('_', maxsplit=1)[-1]}"
                )
                cur_rel_df = pd.read_csv(cur_rel_ffp)
                cur_rel_df.index = [cur_rel_id]

                rel_infos.append(cur_rel_df)

            rel_df = pd.concat(rel_infos, axis=0).rename(
                columns=SOURCE_TO_DB_COLUMNS_MAPPING
            )
            rel_df["event_id"] = cur_event

            # Add hypocentre information
            cur_fault = faults[cur_event]
            s = ((cur_fault.length / 2) - rel_df.shypo.values) / cur_fault.length
            d = rel_df.dhypo.values / cur_fault.width
            hypo_info = np.stack(
                [
                    cur_fault.fault_coordinates_to_wgs_depth_coordinates((s[i], d[i]))
                    for i in range(rel_df.shape[0])
                ],
                axis=0,
            )
            rel_df["hypo_lat"] = hypo_info[:, 0]
            rel_df["hypo_lon"] = hypo_info[:, 1]
            rel_df["hypo_depth"] = hypo_info[:, 2] / 1000

            cur_event_info = median_info.to_dict()
            cur_event_info["event_id"] = cur_event
            event_data.append(cur_event_info)

            rel_data.append(rel_df)

            # db.add_event_data(cur_event, median_info)
            # db.add_realisation_data(rel_df)

        event_df = pd.DataFrame(event_data)
        db.add_event_data(event_df)

        rel_df = pd.concat(rel_data, axis=0)
        db.add_realisation_data(rel_df)

        # Add site to event data
        # Has to be after adding the event data
        # as it uses the event_id -> event_int_id mapping
        db.add_site_event_data(site_event_df)

        # Add IM data
        im_data = []
        logging.info("Adding IM data to the database...")
        for i, cur_event in tqdm(enumerate(events), desc="Processing events"):
            # Read the IM data
            im_files = list((im_data_dir / cur_event / "IM").rglob("*REL*.csv"))
            rel_im_dfs = []
            for cur_rel_ffp in im_files:
                cur_rel_id = (
                    f"{cur_event}_{cur_rel_ffp.stem.rsplit('_', maxsplit=1)[-1]}"
                )

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
            im_data.append(im_df)

            if len(im_data) >= 50 or i == (len(events) - 1):
                db.add_record_im_data(pd.concat(im_data, axis=0))
                im_data = []

    logger.info(f"Database {db_ffp} created successfully.")


@app.command("create-empirical-db")
def create_emp_db(
    db_ffp: Path = typer.Argument(..., help="Path to the database file"),
    imdb_ffp: Path = typer.Argument(..., help="Path to the IMDB file"),
):
    if db_ffp.exists():
        logging.info(f"Database {db_ffp} already exists. Exiting.")
        return

    with nng.DuckEmpiricalDB(db_ffp) as emp_db:
        emp_db.populate(imdb_ffp, nng.constants.GMM_MAPPING)


if __name__ == "__main__":
    app()
