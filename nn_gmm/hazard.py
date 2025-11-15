import logging
import multiprocessing as mp
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import oq_wrapper as oqw

from . import constants
from . import nn_gmm
from .imdb import DuckIMDB

logger = logging.getLogger(__name__)

EMP_GMM_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_SLAB: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMMLogicTree.NSHM2022,
}


def run_sites_hazard(
    model_dir: Path, sites: list[str], device: str, n_procs: int = 1
) -> None:
    """
    Run DS hazard calculations for multiple sites using the specified NN-GMM model.

    Parameters
    ----------
    model_dir : Path
        Path to the directory containing the NN-GMM model and configuration.
    sites : list[str]
        List of site IDs to run the hazard calculations for.
    device : str
        Device to use for model predictions (e.g., 'cpu', 'cuda').
    n_procs : int, optional
        Number of processes to use for parallel computation. Default is 1.
    """
    import seismic_hazard_analysis as sha

    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    # Load the ERF file
    background_ffp = (
        constants.HAZARD_RESOURCES_DIR
        / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt"
    )
    ds_erf_ffp = constants.HAZARD_RESOURCES_DIR / "NZ_DSmodel_2015.txt"

    ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
    ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp)

    (out_dir := model_dir / "ds_hazard/ref_sites").mkdir(parents=True, exist_ok=True)
    if n_procs == 1:
        hazard_results = {}
        for cur_site in sites:
            logger.info(f"Running DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_site_hazard(
                model_dir, site_df.loc[cur_site], ds_source_df, ds_erf_df, device
            )
    else:
        logger.info(
            f"Running DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            hazard_results = p.starmap(
                _run_site_hazard,
                [
                    (model_dir, site_df.loc[cur_site], ds_source_df, ds_erf_df, device)
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    logger.info("Saving hazard results...")
    for cur_site, cur_hazard in hazard_results.items():
        with (out_dir / f"{cur_site}.pkl").open("wb") as f:
            pickle.dump(cur_hazard, f)


def _run_site_hazard(
    model_dir: Path,
    site_series: pd.Series,
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    device: str,
) -> dict[str, dict[str, pd.Series]]:
    """
    Run hazard calculations for a single site.

    Returns
    -------
    dict[str, dict[str, pd.Series]]
        Dictionary with keys
            - hazard type ('total', 'crustal', 'subduction')
            - IM type
        with the final Series containing the hazard:
            index: IM levels
            values: annual rate of exceedance
    """
    import seismic_hazard_analysis as sha

    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")
    rupture_df = ds_source_df.copy()

    # Compute site distances
    rupture_df["rjb"] = (
        np.sqrt(
            (site_series.loc["nztm_x"] - rupture_df["nztm_x"]) ** 2
            + (site_series.loc["nztm_y"] - rupture_df["nztm_y"]) ** 2
        )
        / 1000
    )
    rupture_df["rrup"] = (
        np.sqrt(
            (site_series.loc["nztm_x"] - rupture_df["nztm_x"]) ** 2
            + (site_series.loc["nztm_y"] - rupture_df["nztm_y"]) ** 2
            + (rupture_df["depth"] * 1000) ** 2
        )
        / 1000
    )
    # Use Rjb for rx and ry, in the past we have used zero for this.
    # Using Rjb gives the same result (when using Br13 and ZA06)
    # as using zero, and makes more sense.
    rupture_df["rx"] = rupture_df["rjb"]
    rupture_df["ry"] = rupture_df["rjb"]

    # Apply max-rrup limit
    rupture_df = rupture_df.loc[rupture_df.rrup <= run_config.max_rrup]

    # Add site properties
    rupture_df["vs30"] = site_series.loc["vs30"]
    rupture_df["z1p0"] = site_series.loc["z1p0"]
    rupture_df["z2p5"] = site_series.loc["z2p5"]

    # Rename columns to match expected input names
    rupture_df = rupture_df.rename(
        columns={"mag": "magnitude", "tectonic_type": "tect_type", "dbot": "dbottom"}
    )

    # Convert tectonic type
    rupture_df["tect_type"] = rupture_df["tect_type"].cat.rename_categories(
        {
            "SUBDUCTION_SLAB": "SUBDUCTION_INTERFACE",
        }
    )

    assert np.all(
        np.isin(run_config.source_inputs, rupture_df.columns)
    ), "Source inputs missing"
    assert np.all(
        np.isin(run_config.site_inputs, rupture_df.columns)
    ), "Site inputs missing"
    assert np.all(
        np.isin(run_config.source_to_site_inputs, rupture_df.columns)
    ), "Source-to-site inputs missing"

    pred_df = nn_gmm.run_predictions_dir(model_dir, rupture_df, device)

    total_hazard_results = sha.nshm_2010.compute_gmm_hazard(
        pred_df,
        ds_erf_df.annual_rec_prob,
        run_config.ims,
        mean_col_suffix="_pred",
        std_col_suffix="_pred_std",
    )

    crustal_ids = rupture_df.loc[
        rupture_df.tect_type == "ACTIVE_SHALLOW"
    ].index.values.astype(str)
    crustal_hazard_results = sha.nshm_2010.compute_gmm_hazard(
        pred_df.loc[crustal_ids],
        ds_erf_df.loc[crustal_ids].annual_rec_prob,
        run_config.ims,
        mean_col_suffix="_pred",
        std_col_suffix="_pred_std",
    )

    subduction_ids = rupture_df.loc[
        rupture_df.tect_type == "SUBDUCTION_INTERFACE"
    ].index.values.astype(str)
    subduction_hazard_results = sha.nshm_2010.compute_gmm_hazard(
        pred_df.loc[subduction_ids],
        ds_erf_df.loc[subduction_ids].annual_rec_prob,
        run_config.ims,
        mean_col_suffix="_pred",
        std_col_suffix="_pred_std",
    )

    return {
        "total": total_hazard_results,
        "crustal": crustal_hazard_results,
        "subduction": subduction_hazard_results,
    }


def run_emp_sites_hazard(
    imdb_ffp: Path, sites: list[str], output_dir: Path, n_procs: int = 1
) -> None:
    import seismic_hazard_analysis as sha

    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    # Load the ERF file
    background_ffp = (
        constants.HAZARD_RESOURCES_DIR
        / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt"
    )
    ds_erf_ffp = constants.HAZARD_RESOURCES_DIR / "NZ_DSmodel_2015.txt"

    ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
    ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp)

    if n_procs == 1:
        hazard_results = {}
        for cur_site in sites:
            logger.info(f"Running empirical DS hazard for site: {cur_site}")
            hazard_results[cur_site] = run_emp_site_hazard(
                site_df.loc[cur_site], ds_source_df, ds_erf_df
            )
    else:
        logger.info(
            f"Running empirical DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            hazard_results = p.starmap(
                run_emp_site_hazard,
                [
                    (site_df.loc[cur_site], ds_source_df, ds_erf_df)
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    logger.info("Saving hazard results...")
    for cur_site, cur_hazard in hazard_results.items():
        with (output_dir / f"{cur_site}.pkl").open("wb") as f:
            pickle.dump(cur_hazard, f)


def run_emp_site_hazard(
    site_series: pd.Series, ds_source_df: pd.DataFrame, ds_erf_df: pd.DataFrame
) -> dict[str, dict[str, pd.DataFrame]]:
    import seismic_hazard_analysis as sha

    site_properties = {
        "vs30": site_series.loc["vs30"],
        "vs30measured": True,
        "z1p0": site_series.loc["z1p0"],
        "z2p5": site_series.loc["z2p5"],
        "backarc": False,
    }
    site_nztm_values = site_series[["nztm_x", "nztm_y"]].values

    ds_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
        ds_source_df,
        ds_erf_df,
        site_nztm_values,
        site_properties,
        EMP_GMM_MAPPING,
        constants.PSA_KEYS,
        max_rrup=300.0,
    )

    crustal_ids = ds_source_df.loc[
        ds_source_df.tectonic_type == "ACTIVE_SHALLOW"
    ].index.values.astype(str)
    crustal_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
        ds_source_df.loc[crustal_ids],
        ds_erf_df.loc[crustal_ids],
        site_nztm_values,
        site_properties,
        EMP_GMM_MAPPING,
        constants.PSA_KEYS,
        max_rrup=300.0,
    )

    subduction_ids = ds_source_df.loc[
        (ds_source_df.tectonic_type == "SUBDUCTION_INTERFACE")
        | (ds_source_df.tectonic_type == "SUBDUCTION_SLAB")
    ].index.values.astype(str)
    subduction_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
        ds_source_df.loc[subduction_ids],
        ds_erf_df.loc[subduction_ids],
        site_nztm_values,
        site_properties,
        EMP_GMM_MAPPING,
        constants.PSA_KEYS,
        max_rrup=300.0,
    )

    return {
        "total": ds_hazard,
        "crustal": crustal_hazard,
        "subduction": subduction_hazard,
    }
