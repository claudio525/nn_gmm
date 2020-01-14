import multiprocessing as mp
from typing import Dict

import numpy as np
import pandas as pd

import seistech_internal as si


def load_site_source_dict(
    site_df: pd.DataFrame, site_source_ffp: str, n_procs: int = 4
):
    """Loads the site-source parameters into a dictionary

    Parameters
    ----------
    site_df: pd.DataFrame
        The site params
    site_source_ffp: str
        File path to the site-source DB
    n_procs: int

    Returns
    -------
    Dictionary
        Keys are the station names
        Values are site-source dataframes
            with the sources as index
    """
    with mp.Pool(processes=n_procs) as p:
        results = p.starmap(
            __load_site_df,
            [(station, site_source_ffp) for station in site_df.index.values],
        )
    return {key: value for key, value in results}


def __load_site_df(cur_site, site_source_ffp):
    """MP helper function"""
    with si.dbs.SiteSourceDB(site_source_ffp) as site_source_db:
        return cur_site, site_source_db.station_data(cur_site)


def load_im_dict(im_db_ffp: str, n_procs: int = 4):
    """Loads the IM values dictionary

    Parameters
    ----------
    im_db_ffp: str
        File path to the IM db
    n_procs: int

    Returns
    -------
    Dictionary
        Keys are the fault names
        Values are the IM dataframes, with the station
            as index and the columns of the format "IM_mean"/"IM_std"
            for each IM
    """
    # Get all the faults in the db
    with pd.HDFStore(im_db_ffp, "r") as store:
        faults = [key[1:] for key in store.keys()]

    with mp.Pool(processes=n_procs) as p:
        results = p.starmap(__load_im_df, [(fault, im_db_ffp) for fault in faults])
    return {key: value for key, value in results}


def __load_im_df(cur_fault: str, im_db_ffp: str):
    """MP helper function"""
    with pd.HDFStore(im_db_ffp, "r") as store:
        return cur_fault, store[cur_fault]


def create_sample_comb(im_dict: Dict):
    """Generates the site-source combinations
    for which there is IM data available

    Parameters
    ----------
    im_dict: Dict
        Dictionary
        Keys are the fault names
        Values are the IM dataframes, with the station
            as index and the columns of the format "IM_mean"/"IM_std"
            for each IM

    Returns
    -------
    numpy array
        with 2 columns, [site, source]
    """
    sample_combs = []
    for cur_source, cur_df in im_dict.items():
        sample_combs.append(
            np.concatenate(
                (
                    cur_df.index.values.astype(str).reshape(-1, 1),
                    np.full(cur_df.index.values.shape[0], cur_source).reshape(-1, 1),
                ),
                axis=1,
            )
        )
    return np.concatenate(sample_combs, axis=0)


def drop_missing_data(
    sample_combs: np.ndarray,
    site_df: pd.DataFrame,
    source_df: pd.DataFrame,
    site_source_dict: Dict[str, pd.DataFrame],
    verbose: bool = True
):
    # Check sites
    unique_sites = np.unique(sample_combs[:, 0])
    missing_sites = unique_sites[~pandas_isin(unique_sites, site_df.index.values)]
    sample_combs = sample_combs[~pandas_isin(sample_combs[:, 0], missing_sites), :]
    if verbose:
        print(f"Site df is missing {missing_sites.size} sites")

    # Check sources
    unique_sources = np.unique(sample_combs[:, 1])
    missing_sources = unique_sources[~pandas_isin(unique_sources, source_df.index.values)]
    sample_combs = sample_combs[~pandas_isin(sample_combs[:, 1], missing_sources), :]
    if verbose:
        print(f"Source df is missing the source: {missing_sources}")

    # Check site-source
    for cur_site in np.unique(sample_combs[:, 0]):
        cur_sources = sample_combs[sample_combs[:, 0] == cur_site, 1]

        cur_missing_site_sources = cur_sources[
            ~np.isin(cur_sources, site_source_dict[cur_site].index.values)
        ]
        if verbose:
            print(f"Site-source entries are missing for site {cur_site} - {cur_missing_site_sources}")


def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """This is the same as a np.isin,
    however is significantly faster for large arrays

    https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0
