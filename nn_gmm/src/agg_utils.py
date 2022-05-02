import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import pandas_isin


def load_distance_df(
    site_df: pd.DataFrame, distance_db_ffps: List[Path], n_procs: int = 4
):
    """Loads the site-source parameters into a dictionary

    Parameters
    ----------
    site_df: pd.DataFrame
        The site params
    distance_db_ffps: list of Path
        File path to the distance site-source DB
    n_procs: int

    Returns
    -------
    Dictionary
        Keys are the station names
        Values are site-source dataframes
            with the sources as index
    """
    if n_procs == 1:
        results = []
        for station in site_df.index.values:
            results.append(__load_site_df(station, distance_db_ffps))
    else:
        with mp.Pool(processes=n_procs) as p:
            results = p.starmap(
                __load_site_df,
                [(station, distance_db_ffps) for station in site_df.index.values],
            )

    return pd.concat(results)


def __load_site_df(cur_site: str, distance_db_ffps: List[Path]):
    """MP helper function"""
    dfs = []
    for cur_distance_db_ffp in distance_db_ffps:
        with pd.HDFStore(cur_distance_db_ffp, "r") as db:
            try:
                cur_df = db[f"/distances/station_{cur_site}"]
            except KeyError:
                continue

            faults = db["faults"].loc[cur_df.fault_id].fault_name.values
            cur_df.index = [f"{cur_fault}_{cur_site}" for cur_fault in faults]
            cur_df["source"] = faults
            cur_df["site"] = cur_site

            dfs.append(cur_df)

    return None if len(dfs) == 0 else pd.concat(dfs)


def _load_directivity_db(site_source_db_ffp: Path):
    dfs = []
    with pd.HDFStore(site_source_db_ffp, "r") as db:
        for cur_fault in db.keys():
            dfs.append(db[cur_fault])
    return dfs


def load_directivity_df(site_source_db_ffps: List[Path], n_procs: int = 1):
    """Creates a site-source dataframe from the specified dbs
    and across all faults in the dbs"""
    if n_procs == 1:
        dfs = []
        for cur_site_source_db_ffp in site_source_db_ffps:
            dfs.extend(_load_directivity_db(cur_site_source_db_ffp))
    else:
        with mp.Pool(processes=n_procs) as pool:
            dfs_lists = pool.map(_load_directivity_db, site_source_db_ffps)
        dfs = [cur_df for cur_dfs_list in dfs_lists for cur_df in cur_dfs_list]

    return pd.concat(dfs)


def load_fault_im_df(cur_fault: str, im_db_ffps: Sequence[Path]):
    """Loads the IM dataframe for the specified fault"""
    for im_db_ffp in im_db_ffps:
        with pd.HDFStore(im_db_ffp, "r") as store:
            try:
                return store[cur_fault]
            except KeyError:
                continue

    return None


def create_sample_comb(im_df: pd.DataFrame):
    """Generates the site-source combinations
    for which there is IM data available

    Parameters
    ----------
    im_df: dataframe

    Returns
    -------
    numpy array
        with 2 columns, [site, source]
    """
    split_ids = np.stack(np.char.split(im_df.index.values.astype(str), "_"), axis=0)

    sample_combs = pd.DataFrame(index=im_df.index)
    sample_combs["source"] = split_ids[:, 0]
    sample_combs["realisation"] = np.where(
        split_ids[:, 0] != split_ids[:, 1],
        np.char.add(split_ids[:, 0], np.char.add("_", split_ids[:, 1])),
        split_ids[:, 0],
    )
    sample_combs["site"] = split_ids[:, 2]

    return sample_combs


def drop_missing_data(
    sample_combs: pd.DataFrame,
    site_df: pd.DataFrame,
    source_df: pd.DataFrame,
    site_source_dict: Dict[str, pd.DataFrame],
    verbose: bool = True,
):
    # Check sites
    unique_sites = np.unique(sample_combs.site)
    missing_sites = unique_sites[~pandas_isin(unique_sites, site_df.index.values)]
    sample_combs = sample_combs.loc[~pandas_isin(sample_combs.site, missing_sites), :]
    if missing_sites.size > 0 and verbose:
        print(f"Site df is missing {missing_sites.size} sites")

    # Check sources
    unique_sources = np.unique(sample_combs.source)
    missing_sources = unique_sources[
        ~pandas_isin(unique_sources, source_df.index.values)
    ]
    sample_combs = sample_combs.loc[
        ~pandas_isin(sample_combs.source, missing_sources), :
    ]
    if missing_sources.size > 0 and verbose:
        print(f"Source df is missing the source: {missing_sources}")

    # Check site-source
    start_time = time.time()

    # Check all sites exist
    missing_sites_source_mask = pd.Series(
        index=sample_combs.index,
        data=~pandas_isin(sample_combs.site, np.asarray(list(site_source_dict.keys()))),
    )
    if np.any(missing_sites_source_mask) and verbose:
        print(
            f"Site-Source dict is missing an entry for site/s: {missing_sites_source_mask.loc[missing_sites_source_mask == True]}"
        )

    for cur_site in np.unique(sample_combs.site):
        cur_site_ids = sample_combs.loc[
            sample_combs.site == cur_site
        ].index.values.astype(str)

        missing_source_ids = cur_site_ids[
            ~np.isin(
                sample_combs.loc[cur_site_ids, "source"],
                site_source_dict[cur_site].index.values.astype(str),
            )
        ]
        if missing_source_ids.size > 0:
            missing_sites_source_mask.loc[missing_source_ids] = True
            if verbose:
                print(
                    f"Site-source entries are missing for site {cur_site} - {missing_source_ids}"
                )

    if np.any(missing_sites_source_mask):
        sample_combs = sample_combs.loc[~missing_sites_source_mask]

    print(f"Site source checking took {time.time() - start_time}")

    return sample_combs
