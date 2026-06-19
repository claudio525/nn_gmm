from pathlib import Path
import sys
import logging

from tqdm import tqdm

import pandas as pd
import numpy as np
from source_modelling import sources
from qcore import nhm, point_in_polygon as pip

from . import constants


def get_site_grid_level(site_ids: np.ndarray) -> int:
    """
    Get the grid level of a site based on its ID.

    Parameters
    ----------
    site_id : np.ndarray
        The site IDs.

    Returns
    -------
    np.ndarray
        The grid level of the sites.
    """
    site_grid_level = np.full(site_ids.shape, -1, dtype=int)

    level_0_mask = np.char.startswith(site_ids, "0")
    site_grid_level[level_0_mask] = 0

    level_1_mask = np.char.startswith(site_ids, "1")
    site_grid_level[level_1_mask] = 1

    level_2_mask = np.char.startswith(site_ids, "2")
    site_grid_level[level_2_mask] = 2

    level_3_mask = np.char.startswith(site_ids, "3")
    site_grid_level[level_3_mask] = 3

    level_4_mask = np.char.startswith(site_ids, "4")
    site_grid_level[level_4_mask] = 4

    assert (
        np.count_nonzero(
            ~np.char.isnumeric(site_ids)
            | level_0_mask
            | level_1_mask
            | level_2_mask
            | level_3_mask
            | level_4_mask
        )
        == site_ids.size
    )

    return site_grid_level


def setup_logging(
    log_file: Path = None,
    file_level=logging.DEBUG,
    enable_console: bool = True,
    console_level=logging.INFO,
    file_append: bool = False,
):
    # Create a logger
    logger = logging.getLogger()
    # Set logger to the lowest level of any handler
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers = []

    # Create file handler with its own level
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a" if file_append else "w")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    # Create console handler with its own level
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            # logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    # Suppress numba & matplotlib logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)

    return logger


def run_site_to_source_calc(
    faults: dict[str, sources.Fault], site_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the source to site distances for the given faults and sites

    Parameters
    ----------
    faults: dictionary of Fault objects
        Fault object for each fault id
    site_nztm_coords: array of floats
        The site coordinates in NZTM (X, Y, Depth)
        for each site.
        Shape: (n_sites, 3)
    """
    import seismic_hazard_analysis as sha

    fault_id_mapping = {cur_name: i for i, cur_name in enumerate(faults.keys())}
    fault_id_mapping_reverse = {i: cur_name for i, cur_name in enumerate(faults.keys())}
    plane_nztm_coords = []
    scenario_ids = []
    scenario_section_ids = []
    segment_section_ids = []
    for cur_name, cur_fault in faults.items():
        plane_nztm_coords.append(
            np.stack(
                [cur_plane.bounds[:, [1, 0, 2]] for cur_plane in cur_fault.planes],
                axis=2,
            )
        )
        cur_id = fault_id_mapping[cur_name]
        scenario_ids.append(cur_id)
        # Each scenario only consists of a single fault/section
        scenario_section_ids.append(np.asarray([cur_id]))
        segment_section_ids.append(np.ones(len(cur_fault.planes), dtype=int) * cur_id)

    plane_nztm_coords = np.concatenate(plane_nztm_coords, axis=2)
    scenario_ids = np.asarray(scenario_ids)
    segment_section_ids = np.concatenate(segment_section_ids)

    assert plane_nztm_coords.shape[2] == segment_section_ids.size

    # Change the order of the corners
    plane_nztm_coords = plane_nztm_coords[[0, 3, 1, 2], :, :]

    # Compute rupture scenario distances
    dist_dfs = []
    for cur_site_id, cur_site_row in tqdm(
        site_df.iterrows(), "Processing sites", total=site_df.shape[0]
    ):
        cur_site_nztm_coords = np.array(
            [cur_site_row["nztm_x"], cur_site_row["nztm_y"], 0]
        )
        cur_dist_df = sha.site_source.get_scenario_distances(
            scenario_ids,
            scenario_section_ids,
            plane_nztm_coords,
            segment_section_ids,
            cur_site_nztm_coords,
        )
        cur_dist_df["event_id"] = cur_dist_df.index.map(fault_id_mapping_reverse)
        cur_dist_df["site_id"] = cur_site_id

        dist_dfs.append(cur_dist_df)

    dist_df = pd.concat(dist_dfs, axis=0).reset_index(drop=True)
    dist_df = dist_df.astype({"event_id": "category", "site_id": "category"})

    return dist_df


def get_site_event_int_id(
    site_int_id: np.ndarray, event_int_id: np.ndarray
) -> np.ndarray:
    """
    Generate unique integer IDs for each site-event pair
    based on site and event IDs.

    Parameters
    ----------
    site_int_id : np.ndarray
        Array of site integer IDs.
    event_int_id : np.ndarray
        Array of event integer IDs.

    Returns
    -------
    np.ndarray
        Array of unique integer IDs for each site-event pair.
    """
    # Prime multipliers for good distribution
    p1, p2 = 73856093, 19349663

    return (site_int_id.astype(np.int64) * p1) ^ (
        event_int_id.astype(np.int64) * p2
    ) % 100000000


def get_fault(nhm_flt_ffp: Path, fault_name: str) -> sources.Fault:
    """
    Get a fault object from the NHM fault definitions.

    Parameters
    ----------
    nhm_flt_ffp : Path
        Path to the NHM fault definitions file.
    fault_name : str
        Name of the fault to retrieve.

    Returns
    -------
    sources.Fault
        The fault object corresponding to the given name.
    """
    import seismic_hazard_analysis as sha

    flt_definitions = nhm.load_nhm(nhm_flt_ffp)
    if fault_name not in flt_definitions:
        raise ValueError(f"Fault '{fault_name}' not found in NHM definitions.")

    return sha.nshm_2010.utils.get_fault_objects(flt_definitions[fault_name])


def get_faults(nhm_flt_ffp: Path) -> dict[str, sources.Fault]:
    """
    Get all faults from the NHM fault definitions.

    Parameters
    ----------
    nhm_flt_ffp : Path
        Path to the NHM fault definitions file.

    Returns
    -------
    dict[str, sources.Fault]
        Dictionary mapping fault names to their corresponding Fault objects.
    """
    import seismic_hazard_analysis as sha

    flt_definitions = nhm.load_nhm(nhm_flt_ffp)
    return {
        cur_name: sha.nshm_2010.utils.get_fault_objects(cur_fault)
        for cur_name, cur_fault in flt_definitions.items()
    }


def reverse_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace("p", ".", 1)[::-1]
    return im


def get_im_filename(im: str):
    if im.startswith("pSA"):
        return im.replace(".", "p", 1)
    return im


def get_nice_im_name(im: str, use_latex: bool = False):
    if im.startswith("pSA"):
        return f"pSA({im.split('_')[-1]}s)"

    if use_latex:
        match im.lower():
            case "ds595":
                return "$D_{s595}$"
            case "ds575":
                return "$D_{s575}$"
            case _:
                return im
    return im


def get_pSA_period(im: str):
    if im.startswith("pSA"):
        return float(im.split("_")[-1])
    return None


def add_basin_column(site_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a basin column to the given site dataframe"""
    basin_boundary_files = [
        constants.BASIN_BOUNDARIES_DIR / f
        for f in constants.BASIN_BOUNDARIES_DIR.iterdir()
        if f.name.endswith(".txt")
    ]
    assert (
        len(basin_boundary_files) > 0
    ), f"No basin boundary files found in {constants.BASIN_BOUNDARIES_DIR}"
    basin_boundaries = {
        f.stem.split("_", maxsplit=1)[0]: np.loadtxt(f) for f in basin_boundary_files
    }

    site_df["basin"] = None
    for basin_name, boundary in basin_boundaries.items():
        mask = pip.is_inside_postgis_parallel(site_df[["lon", "lat"]].values, boundary)
        site_df.loc[mask, "basin"] = basin_name

    # Combine Banks Peninsula with Canterbury
    site_df.loc[site_df["basin"] == "BanksPeninsulaVolcanics", "basin"] = "Canterbury"

    # Replace None values with NiB (Not in Basin)
    site_df["basin"] = site_df["basin"].fillna("NiB")

    site_df = site_df.astype({"basin": "category"})
    return site_df


def get_ds_source_data():
    """Load DS source and ERF data."""
    import seismic_hazard_analysis as sha

    background_ffp = constants.HAZARD_RESOURCES_DIR / "NZBCK211_OpenSHA.txt"
    ds_erf_ffp = constants.HAZARD_RESOURCES_DIR / "NZ_DSmodel_2010.txt"

    ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name").sort_index()
    ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp).sort_index()

    # Use numerical index to save memory/disk space
    ds_source_df["rupture_name"] = ds_source_df.index.astype(str)
    ds_erf_df["rupture_name"] = ds_erf_df.index.astype(str)
    ds_source_df.index = np.arange(ds_source_df.shape[0])
    ds_erf_df.index = np.arange(ds_erf_df.shape[0])

    # DS source are point sources
    ds_source_df["is_point_source"] = True

    return ds_source_df, ds_erf_df


def rp_to_poe_string(rp: int) -> str:
    """Convert return period to nice PoE string."""
    poe = rp_to_prob(rp, 50)
    return f"{int(np.round(poe * 100.0))}% in 50 Years"


def rp_to_prob(rp: float, t: float = 1.0):
    """
    Converts return period to exceedance probability
    Based on Poisson distribution

    Parameters
    ----------
    rp: float
        Return period
    t: float
        Time period of interest

    Returns
    -------
    Exceedance probability
    """
    return 1 - np.exp(-t / rp)
