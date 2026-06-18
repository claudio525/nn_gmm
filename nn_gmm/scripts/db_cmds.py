from pathlib import Path
import logging
import re

import pandas as pd
import numpy as np
import typer
from tqdm import tqdm
import xarray as xr

import ml_tools as mlt
import nn_gmm as nng
from qcore import coordinates as coords
from qcore import nhm
import workflow.realisations as wr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

app = typer.Typer()


@app.command("create-imdb")
def create_imdb(
    db_ffp: Path = typer.Argument(..., help="Path to the database file"),
    im_data_loc: Path = typer.Argument(
        ...,
        help="Path to the Cybershake IM data. Either directory of IM event directories, or IM data pickle file.",
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
    cs_100m: bool = typer.Option(
        False, "--cs-100m", help="Whether this is 100m grid Cybershake data"
    ),
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
    import seismic_hazard_analysis as sha

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

    im_data_arrays = None
    if im_data_loc.is_file() and im_data_loc.suffix == ".pkl":
        im_data_arrays = pd.read_pickle(im_data_loc)
        im_events = np.sort(list(im_data_arrays.keys()))
    else:
        im_events = np.sort(
            [cur_dir.stem for cur_dir in im_data_loc.iterdir() if cur_dir.is_dir()]
        )

    # Get source events
    source_events = []
    source_dir_name = "Source" if cs_100m else "Srf"
    for cur_event in im_events:
        if (
            not (source_info_dir / cur_event).exists()
            or (not (source_info_dir / cur_event / source_dir_name).exists())
            or len(list((source_info_dir / cur_event / source_dir_name).iterdir())) == 0
        ):
            logger.warning(
                f"Event {cur_event} exists in IM data but not in source info directory. Skipping!"
            )
        else:
            source_events.append(cur_event)
    source_events = np.array(source_events)

    events = np.intersect1d(im_events, source_events)

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
            source_dir = source_info_dir / cur_event / source_dir_name

            assert (
                source_dir.exists() and len(list(source_dir.iterdir())) > 0
            ), f"Source directory {source_dir} does not exist or is empty!"

            # Read the median data
            median_info = pd.read_csv(source_dir / f"{cur_event}.csv").squeeze()
            median_info = median_info.loc[SOURCE_INFO_FIELDS]
            median_info = median_info.rename(SOURCE_TO_DB_COLUMNS_MAPPING, axis=0)

            # Read the realisation data
            rel_infos = []
            # for cur_rel_ffp in source_dir.glob("*REL*.csv"):
            pattern = re.compile(r".*REL[0-9]+\.csv$")
            rel_info_files = [f for f in source_dir.iterdir() if pattern.match(f.name)]
            for cur_rel_ffp in rel_info_files:
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
        for i, cur_event in enumerate(tqdm(events, desc="Processing events")):
            # Read the IM data
            if im_data_arrays is not None:
                rel_im_dfs = []
                for cur_rel in im_data_arrays[cur_event].realisation.values:
                    cur_rel_id = f"{cur_event}_{cur_rel}"

                    if cur_rel_id not in rel_df.index:
                        logging.warning(
                            f"Realisation {cur_rel_id} not found in source data, but exists in IM data. Skipping!"
                        )
                        continue

                    cur_im_df = (
                        im_data_arrays[cur_event].sel(realisation=cur_rel).to_pandas()
                    )
                    cur_im_df["site_id"] = cur_im_df.index
                    cur_im_df["event_id"] = cur_event
                    cur_im_df["rel_id"] = cur_rel_id

                    rel_im_dfs.append(cur_im_df)
            else:
                im_files = list((im_data_loc / cur_event / "IM").rglob("*REL*.csv"))
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

                    rel_im_dfs.append(cur_im_df)

            im_df = pd.concat(rel_im_dfs, axis=0)
            im_df.index = mlt.array_utils.numpy_str_join(
                "_", im_df.rel_id.values.astype(str), im_df.site_id.values.astype(str)
            )
            im_data.append(im_df)

            if len(im_data) >= 50 or i == (len(events) - 1):
                db.add_record_im_data(pd.concat(im_data, axis=0))
                im_data = []

    logger.info(f"Database {db_ffp} created successfully.")


@app.command("imdb-add-ds-simulations")
def imdb_add_ds_sims(
    db_ffp: Path = typer.Argument(..., help="Path to the IMDB file"),
    ds_im_dir: Path = typer.Argument(..., help="Path to the DS IM data directory"),
    ds_rel_info_dir: Path = typer.Argument(
        ..., help="Path to the DS realisation info directory"
    ),
):
    """
    Add DS point source simulations to an existing IMDB.
    """
    logger = nng.utils.setup_logging()
    im_dirs = {
        cur_dir.stem: cur_dir for cur_dir in ds_im_dir.iterdir() if cur_dir.is_dir()
    }
    events = np.array(list(im_dirs.keys()))

    # Realisation info files
    rel_ffps = {
        event: path
        for event in events
        if (path := ds_rel_info_dir / event / "realisation.json").exists()
    }
    assert len(rel_ffps) == len(events), "Some realisation info files are missing!"

    with nng.DuckIMDB(db_ffp) as db:
        db_event_columns = db.event_table_columns
        db_rel_columns = db.realisation_table_columns

        max_event_int_id = db.max_event_int_id
        assert max_event_int_id is not None, "Database has no events!"
        max_rel_int_id = db.max_rel_int_id
        assert max_rel_int_id is not None, "Database has no realisations!"

        db_site_df = db.get_site_df()
        db_site_ids = db_site_df["site_id"].values.astype(str)

    site_coords = np.concatenate(
        [db_site_df[["lat", "lon"]].values, np.zeros((db_site_df.shape[0], 1))], axis=1
    )

    event_int_id = max_event_int_id + 1
    rel_int_id = max_rel_int_id + 1
    event_data, rel_data = {}, {}
    im_dfs, site_event_dfs = [], []
    for event in events:
        rel_ffp = rel_ffps[event]
        source_config = wr.SourceConfig.read_from_realisation(rel_ffp)

        # Event data
        event_data[event] = {}
        event_data[event]["event_int_id"] = event_int_id
        event_data[event]["event_id"] = event
        event_data[event]["magnitude"] = wr.Magnitudes.read_from_realisation(
            rel_ffp
        ).magnitudes[event]
        event_data[event]["sim_type"] = 1  # Point source
        event_data[event]["fault_type"] = "DS_POINT_SOURCE"
        event_data[event]["tect_type"] = "SUBDUCTION_SLAB"
        event_data[event]["dip"] = source_config.source_geometries[event].dip
        event_data[event]["dtop"] = (
            source_config.source_geometries[event].coordinates[2] / 1000
        )
        event_data[event]["dbottom"] = event_data[event]["dtop"]
        event_data[event]["length"] = source_config.source_geometries[event].length
        event_data[event]["plane_count"] = 0
        event_data[event]["dip_dir"] = source_config.source_geometries[event].dip_dir
        event_int_id += 1

        # Realisation data
        rel_data[event] = {}
        rel_data[event]["rel_int_id"] = rel_int_id
        rel_data[event]["rel_id"] = f"{event}_REL01"
        rel_data[event]["event_int_id"] = event_data[event]["event_int_id"]
        rel_data[event]["magnitude"] = event_data[event]["magnitude"]
        rel_data[event]["rake"] = wr.Rakes.read_from_realisation(rel_ffp).rakes[event]
        rel_data[event]["shypo"], rel_data[event]["dhypo"] = 0.5, 0.5
        rel_data[event]["hypo_lat"] = source_config.source_geometries[
            event
        ].coordinates[0]
        rel_data[event]["hypo_lon"] = source_config.source_geometries[
            event
        ].coordinates[1]
        rel_data[event]["hypo_depth"] = (
            source_config.source_geometries[event].coordinates[2] / 1000
        )
        rel_int_id += 1

        # IM data
        im_ffp = im_dirs[event] / "intensity_measures.h5"
        im_ds = xr.load_dataset(im_ffp)
        site_ids = im_ds.coords["station"].values.astype(str)
        assert np.isin(
            site_ids, db_site_ids
        ).all(), "Some sites in IM data do not exist in the database!"
        im_df = pd.DataFrame(index=site_ids, columns=nng.constants.IMS, data=np.nan)
        im_df["event_int_id"] = event_data[event]["event_int_id"]
        im_df["rel_int_id"] = rel_data[event]["rel_int_id"]
        for im in nng.constants.IMS:
            if im.startswith("pSA"):
                period = nng.utils.get_pSA_period(im)
                im_df[im] = (
                    im_ds["pSA"].sel(component="rotd50", period=period).to_pandas()
                )
            else:
                component = "rotd50" if im in ["PGA", "PGV"] else "geom"
                im_df[im] = im_ds.sel(component=component)[im].to_pandas()

        im_df["site_id"] = im_df.index
        im_dfs.append(im_df)

        # Site-to-source distances
        source_coords = np.array(
            [
                rel_data[event]["hypo_lat"],
                rel_data[event]["hypo_lon"],
                rel_data[event]["hypo_depth"] * 1000,
            ]
        )
        site_mask = np.isin(db_site_ids, site_ids)
        assert site_mask.sum() == im_df.shape[0], "Site count mismatch!"
        rrup = (
            coords.distance_between_wgs_depth_coordinates(
                source_coords[None, :], site_coords[site_mask]
            )
            / 1000
        )
        rjb = (
            coords.distance_between_wgs_depth_coordinates(
                source_coords[None, :2], site_coords[site_mask, :2]
            )
            / 1000
        )
        site_event_df = pd.DataFrame(
            index=db_site_ids[site_mask],
            data={"rrup": rrup, "rjb": rjb},
        )
        site_event_df["rx"], site_event_df["ry"] = 0, 0
        site_event_df["event_id"] = event
        site_event_df["site_id"] = site_event_df.index
        site_event_dfs.append(site_event_df)

    # Combine
    event_df = pd.DataFrame.from_dict(event_data, orient="index")
    assert np.isin(db_event_columns, event_df.columns).all(), "Missing event columns!"
    rel_df = pd.DataFrame.from_dict(rel_data, orient="index").sort_values("rel_id")
    assert np.isin(db_rel_columns, rel_df.columns).all(), "Missing realisation columns!"
    im_df = pd.concat(im_dfs, axis=0, ignore_index=True)
    assert np.isin(
        ["event_int_id", "rel_int_id", "site_id"] + nng.constants.IMS, im_df.columns
    ).all(), "Missing record IM columns!"
    site_event_df = pd.concat(
        site_event_dfs,
        axis=0,
        ignore_index=True,
    )

    # Add to database
    with nng.DuckIMDB(db_ffp) as db:
        db.add_event_data(event_df)
        logger.info(
            f"Added {event_df.shape[0]} DS point source events to the database."
        )
        db.add_realisation_data(rel_df)
        logger.info(
            f"Added {rel_df.shape[0]} DS point source realisations to the database."
        )
        db.add_site_event_data(site_event_df)
        logger.info(
            f"Added site-event distance data for {site_event_df.shape[0]} DS point source event-site pairs to the database."
        )
        db.add_record_im_data(im_df)
        logger.info(
            f"Added IM data for {im_df.shape[0]} DS point source records to the database."
        )


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
