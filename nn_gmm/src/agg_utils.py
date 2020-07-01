import time
import multiprocessing as mp
from typing import Dict

import numpy as np
import pandas as pd

from .utils import pandas_isin


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
    if n_procs == 1:
        results = []
        for station in site_df.index.values:
            results.append(__load_site_df(station, site_source_ffp))
    else:
        with mp.Pool(processes=n_procs) as p:
            results = p.starmap(
                __load_site_df,
                [(station, site_source_ffp) for station in site_df.index.values],
            )

    return pd.concat(results)


def __load_site_df(cur_site, site_source_ffp):
    """MP helper function"""
    with pd.HDFStore(site_source_ffp, "r") as db:
        try:
            df = db[f"/distances/station_{cur_site}"]
        except KeyError:
            return None

        faults = db["faults"].loc[df.fault_id].fault_name.values
        df.index = [f"{cur_fault}_{cur_site}" for cur_fault in faults]
        df["source"] = faults
        df["site"] = cur_site

    return df


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

    # Check that all the dataframes have the same IMs
    assert np.all(
        [
            np.all(np.isin(cur_df.columns.values, results[0].columns.values))
            for cur_df in results
        ]
    )
    return pd.concat(results, sort=True)


def __load_im_df(cur_fault: str, im_db_ffp: str):
    """MP helper function"""
    with pd.HDFStore(im_db_ffp, "r") as store:
        return store[cur_fault]


def create_sample_comb(im_df: pd.DataFrame):
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
    split_ids = np.stack(np.char.split(im_df.index.values.astype(str), "_"), axis=0)

    sample_combs = pd.DataFrame(index=im_df.index)
    sample_combs["source"] = split_ids[:, 0]
    sample_combs["site"] = split_ids[:, 2]

    return sample_combs

    # ids, sources, sites = [], [], []
    #
    # for cur_source, cur_df in im_dict.items():
    #     cur_ids = cur_df.index.values.astype(str)
    #     split_ids = np.stack(np.char.split(cur_ids, "_"), axis=0)
    #
    #     ids.append(cur_ids)
    #     sources.append(split_ids[:, 0])
    #     sites.append(split_ids[:, 2])
    #
    # ids = np.concatenate(ids)
    # sources, sites = np.concatenate(sources), np.concatenate(sites)
    # sample_combs_df = pd.DataFrame(
    #     index=ids, data=np.stack([sources, sites], axis=1), columns=["source", "site"]
    # )
    # return sample_combs_df


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
