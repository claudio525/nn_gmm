from pathlib import Path
import sys
import logging

from tqdm import tqdm

import pandas as pd
import numpy as np
from source_modelling import sources
import seismic_hazard_analysis as sha


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


def setup_logging(log_file: Path, file_level=logging.DEBUG, console_level=logging.INFO):
    # Create a logger
    logger = logging.getLogger()
    # Set logger to the lowest level of any handler
    logger.setLevel(min(file_level, console_level))

    # Clear any existing handlers
    logger.handlers = []

    # Create file handler with its own level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)  # More detailed logging to file
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Create console handler with its own level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)  # Less verbose on console
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

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

    return (site_int_id * p1) ^ (event_int_id * p2) % 100000000
