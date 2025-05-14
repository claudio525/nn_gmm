import numpy as np


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

    level_0_mask = np.strings.startswith(site_ids, "0")
    site_grid_level[level_0_mask] = 0

    level_1_mask = np.strings.startswith(site_ids, "1")
    site_grid_level[level_1_mask] = 1

    level_2_mask = np.strings.startswith(site_ids, "2")
    site_grid_level[level_2_mask] = 2

    level_3_mask = np.strings.startswith(site_ids, "3")
    site_grid_level[level_3_mask] = 3

    level_4_mask = np.strings.startswith(site_ids, "4")
    site_grid_level[level_4_mask] = 4

    assert (
        np.count_nonzero(
            ~np.strings.isnumeric(site_ids)
            | level_0_mask
            | level_1_mask
            | level_2_mask
            | level_3_mask
            | level_4_mask
        )
        == site_ids.size
    )

    return site_grid_level

