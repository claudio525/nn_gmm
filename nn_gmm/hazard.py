import gc
import logging
import multiprocessing as mp
import pickle
from pathlib import Path
from functools import partial

import torch
import pandas as pd
import numpy as np
import xarray as xr

import oq_wrapper as oqw

from . import constants
from . import nn_gmm
from .imdb import DuckIMDB
from . import utils

logger = logging.getLogger(__name__)

EMP_GMM_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_SLAB: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMMLogicTree.NSHM2022,
}

def compute_nn_ds_uhs(ds_results_dir: Path, rps: list[int], sites: list[str] | None = None):
    """Compute NN-GMM DS UHS for all sites in the specified results directory."""
    import seismic_hazard_analysis as sha
    sites = [item.stem for item in ds_results_dir.glob("*.pkl")] if sites is None else sites
    excd_rates = [sha.utils.rp_to_prob(rp) for rp in rps]

    logger.info(f"Computing NN-GMM DS UHS for {len(sites)} sites")
    nn_ds_uhs = xr.DataArray(
        dims=["site", "im", "rp"],
        coords={
            "site": sites,
            "im": constants.PSA_KEYS,
            "rp": rps,
        },
        data=np.full((len(sites), len(constants.PSA_KEYS), len(rps)), np.nan),
    )
    nn_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(ds_results_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in ds_results_dir.glob("*.pkl")
    }
    for site in sites:
        if site not in nn_ds_hazard:
            logger.warning(f"No DS hazard results found for site {site}, skipping UHS computation.")
            continue

        nn_ds_uhs.loc[site, :, :] = sha.uhs.compute_uhs(
            {
                cur_im: nn_ds_hazard[site]["total"][cur_im]
                for cur_im in constants.PSA_KEYS
            },
            excd_rates,
            rps=rps,
        )

    return nn_ds_uhs

def compute_emp_ds_uhs(ds_results_dir: Path, rps: list[int], sites: list[str] | None = None):
    import seismic_hazard_analysis as sha
    sites = [item.stem for item in ds_results_dir.glob("*.pkl")] if sites is None else sites
    excd_rates = [sha.utils.rp_to_prob(rp) for rp in rps]

    logger.info(f"Computing Empirical DS UHS for {len(sites)} sites")
    ds_uhs = xr.DataArray(
        dims=["site", "im", "rp"],
        coords={
            "site": sites,
            "im": constants.PSA_KEYS,
            "rp": rps,
        },
        data=np.full((len(sites), len(constants.PSA_KEYS), len(rps)), np.nan),
    )
    emp_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(ds_results_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in ds_results_dir.glob("*.pkl")
    }
    for site in sites:
        ds_uhs.loc[site, :, :] = sha.uhs.compute_uhs(
            {
                cur_im: emp_ds_hazard[site]["total"][cur_im]
                for cur_im in constants.PSA_KEYS
            },
            excd_rates,
            rps=rps,
        )

    return ds_uhs

def compute_cs_flt_uhs(sites: list[str], rps: list[int]):
    import seismic_hazard_analysis as sha
    excd_rates = [sha.utils.rp_to_prob(rp) for rp in rps]

    logger.info(f"Computing Cybershake fault UHS for {len(sites)} sites")
    cs_flt_hazard = pd.read_pickle(
        constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )
    uhs_da = {}
    for site in sites:
        cur_flt_hazard = {
            cur_im: cs_flt_hazard[cur_im].loc[site] for cur_im in constants.PSA_KEYS
        }
        uhs_da[site] = sha.uhs.compute_uhs(cur_flt_hazard, excd_rates, rps=rps)

    # Convert to DataArray
    uhs_da = xr.DataArray(
        dims=["site", "im", "rp"],
        coords={
            "site": list(uhs_da.keys()),
            "im": constants.PSA_KEYS,
            "rp": rps,
        },
        data=np.stack([uhs_da[site].values for site in uhs_da.keys()]),
    )

    return uhs_da


def run_sites_ds_hazard(
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
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    # Load DS source data
    ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()

    (out_dir := model_dir / "ds_hazard/ref_sites").mkdir(parents=True, exist_ok=True)
    if n_procs == 1:
        hazard_results = {}
        for cur_site in sites:
            logger.info(f"Running DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_site_ds_hazard(
                model_dir, site_df.loc[cur_site], ds_source_df, ds_erf_df, device
            )
    else:
        logger.info(
            f"Running DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            hazard_results = p.starmap(
                _run_site_ds_hazard,
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


def compute_uniform_grid_ds_hazard(
    imdb_ffp: Path, model_dir: Path, device: str, n_procs: int = 1
):
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")

    # Get the sites
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(
            add_nztm=True, min_grid_level=0, max_grid_level=0
        ).set_index("site_id")
    sites = site_df.index.values.astype(str)

    # Load DS source data
    ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()

    if n_procs == 1:
        hazard_results = {}
        for cur_site, cur_row in site_df.iterrows():
            logger.info(f"Running DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_site_ds_hazard(
                model_dir,
                cur_row,
                ds_source_df,
                ds_erf_df,
                device,
                tect_type_hazard=False,
            )
    else:
        logger.info(
        f"Running DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            hazard_results = p.starmap(
                _run_site_ds_hazard,
                [
                    (model_dir, site_df.loc[cur_site], ds_source_df, ds_erf_df, device, False)
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    # Save per IM
    (out_dir := model_dir / "ds_hazard/uniform_grid").mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hazard results in {out_dir}")
    for im in run_config.ims:
        im_hazard_df = pd.concat([hazard_results[cur_site]["total"][im] for cur_site in sites], axis=1)
        im_hazard_df.columns = sites

        im_hazard_df.to_parquet(out_dir / f"{utils.get_im_filename(im)}_ds_hazard.parquet")


def _run_site_ds_hazard(
    model_dir: Path,
    site_series: pd.Series,
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    device: str,
    tect_type_hazard: bool = True,
) -> dict[str, dict[str, pd.Series]]:
    """
    Run DS hazard calculations for a single site using a NN-GMM model.

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

    # Same as what is done for training for DS point sources
    rupture_df["rx"] = 0
    rupture_df["ry"] = 0

    # Apply max-rrup limit
    rupture_df = rupture_df.loc[rupture_df.rrup <= run_config.max_rrup]

    # Add site properties
    rupture_df["vs30"] = site_series.loc["vs30"]
    rupture_df["z1p0"] = site_series.loc["z1p0"]
    rupture_df["z2p5"] = site_series.loc["z2p5"]

    # Rename columns to match expected input names
    rupture_df = rupture_df.rename(
        columns={
            "mag": "magnitude",
            "tectonic_type": "tect_type",
            "dbot": "dbottom",
            "depth": "hypo_depth",
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

    crustal_hazard_results = None
    if tect_type_hazard:
        crustal_ids = rupture_df.loc[
            rupture_df.tect_type == "ACTIVE_SHALLOW"
        ].index.values
        crustal_hazard_results = sha.nshm_2010.compute_gmm_hazard(
            pred_df.loc[crustal_ids],
            ds_erf_df.loc[crustal_ids].annual_rec_prob,
            run_config.ims,
            mean_col_suffix="_pred",
            std_col_suffix="_pred_std",
        )

    subduction_slab_hazard_results = None
    if tect_type_hazard:
        subduction_slab_ids = rupture_df.loc[
            rupture_df.tect_type == "SUBDUCTION_SLAB"
        ].index.values
        subduction_slab_hazard_results = sha.nshm_2010.compute_gmm_hazard(
            pred_df.loc[subduction_slab_ids],
            ds_erf_df.loc[subduction_slab_ids].annual_rec_prob,
            run_config.ims,
            mean_col_suffix="_pred",
            std_col_suffix="_pred_std",
        )

    # Explicit GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    return {
        "total": total_hazard_results,
        "crustal": crustal_hazard_results,
        "subduction_slab": subduction_slab_hazard_results,
    }


def run_emp_sites_ds_hazard(
    imdb_ffp: Path, sites: list[str], output_dir: Path, n_procs: int = 1
) -> None:
    """Compute empirical DS hazard for multiple sites."""
    import seismic_hazard_analysis as sha

    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    # Load the ERF file
    background_ffp = constants.HAZARD_RESOURCES_DIR / "NZBCK211_OpenSHA.txt"
    ds_erf_ffp = constants.HAZARD_RESOURCES_DIR / "NZ_DSmodel_2010.txt"

    ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
    ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp)

    if n_procs == 1:
        hazard_results = {}
        for cur_site in sites:
            logger.info(f"Running empirical DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_emp_site_ds_hazard(
                site_df.loc[cur_site],
                ds_source_df,
                ds_erf_df,
            )
    else:
        logger.info(
            f"Running empirical DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            fn_call = partial(
                _run_emp_site_ds_hazard,
                ds_source_df=ds_source_df,
                ds_erf_df=ds_erf_df,
                tect_type_hazard=False,
            )
            hazard_results = p.starmap(
                fn_call,
                [
                    (
                        site_df.loc[cur_site],
                        ds_source_df,
                        ds_erf_df,
                    )
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    logger.info("Saving hazard results...")
    for cur_site, cur_hazard in hazard_results.items():
        with (output_dir / f"{cur_site}.pkl").open("wb") as f:
            pickle.dump(cur_hazard, f)

def compute_uniform_grid_emp_ds_hazard(
    imdb_ffp: Path,
    output_dir: Path,
    n_procs: int = 1,
):
    # Get the sites
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(
            add_nztm=True, min_grid_level=0, max_grid_level=0
        ).set_index("site_id")
    sites = site_df.index.values.astype(str)

    # Load DS source data
    ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()

    if n_procs == 1:
        hazard_results = {}
        for cur_site, cur_row in site_df.iterrows():
            logger.info(f"Running empirical DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_emp_site_ds_hazard(
                cur_row,
                ds_source_df,
                ds_erf_df,
                tect_type_hazard=False,
            )
    else:
        logger.info(
        f"Running DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            fn_call = partial(
                _run_emp_site_ds_hazard,
                ds_source_df=ds_source_df,
                ds_erf_df=ds_erf_df,
                tect_type_hazard=False,
            )
            hazard_results = p.starmap(
                fn_call,
                [
                    (site_df.loc[cur_site],)
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    # Save per IM
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hazard results in {output_dir}")
    for im in constants.PSA_KEYS:
        im_hazard_df = pd.concat([hazard_results[cur_site]["total"][im] for cur_site in sites], axis=1)
        im_hazard_df.columns = sites

        im_hazard_df.to_parquet(output_dir / f"{utils.get_im_filename(im)}_ds_hazard.parquet")


def _run_emp_site_ds_hazard(
    site_series: pd.Series,
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    max_rrup: float | None = 300.0,
    tect_type_hazard: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute empirical DS hazard for a single site."""
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
        max_rrup=max_rrup,
    )

    crustal_hazard = None
    if tect_type_hazard:
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
            max_rrup=max_rrup,
        )

    subduction_hazard = None
    if tect_type_hazard:
        subduction_ids = ds_source_df.loc[
            ds_source_df.tectonic_type == "SUBDUCTION_SLAB"
        ].index.values.astype(str)
        subduction_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
            ds_source_df.loc[subduction_ids],
            ds_erf_df.loc[subduction_ids],
            site_nztm_values,
            site_properties,
            EMP_GMM_MAPPING,
            constants.PSA_KEYS,
            max_rrup=max_rrup,
        )

    return {
        "total": ds_hazard,
        "crustal": crustal_hazard,
        "subduction": subduction_hazard,
    }


def run_emp_ds_disagg(
    site_ids: np.ndarray,
    imdb_ffp: Path,
    output_dir: Path,
    rps: np.ndarray,
    ims: list[str] | None = None,
    max_rrup: float = 500.0,
    n_procs: int = 1,
    tect_type: str | None = None,
) -> None:
    """Compute empirical DS disaggregation for multiple sites."""
    import seismic_hazard_analysis as sha

    logger.info("Running empirical DS disaggregation...")

    ims = ims or constants.PLOT_IMS

    # Load DS source data
    ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()
    if tect_type is not None:
        logger.info(f"Filtering DS sources for tectonic type: {tect_type}")
        ds_source_df = ds_source_df.loc[ds_source_df.tectonic_type == tect_type]
        ds_erf_df = ds_erf_df.loc[ds_source_df.index]

    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    logger.info("Adding backarc mask to site dataframe...")
    site_df["backarc"] = sha.nshm_2022.get_backarc_mask(site_df[["lon", "lat"]].values)

    if n_procs == 1:
        disagg_contrs = {}
        for site in site_ids:
            logger.info(f"Running disaggregation for site: {site}")
            disagg_contrs[site] = _run_site_emp_ds_disagg(
                site_df.loc[site],
                ds_source_df,
                ds_erf_df,
                rps,
                ims,
                max_rrup,
                output_dir,
            )
    else:
        logger.info(
            f"Running empirical DS disaggregation for {len(site_ids)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            disagg_results = p.starmap(
                _run_site_emp_ds_disagg,
                [
                    (
                        site_df.loc[site],
                        ds_source_df,
                        ds_erf_df,
                        rps,
                        ims,
                        max_rrup,
                        output_dir,
                    )
                    for site in site_ids
                ],
            )
        disagg_contrs = {site: result for site, result in zip(site_ids, disagg_results)}


def _run_site_emp_ds_disagg(
    site_series: pd.Series,
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    rps: np.ndarray,
    ims: list[str],
    max_rrup: float,
    output_dir: Path,
):
    import seismic_hazard_analysis as sha

    if (output_dir / f"{site_series.name}.nc").exists():
        logger.info(
            f"Disaggregation file for site {site_series.name} already exists. Skipping."
        )
        return

    excd_probs = sha.utils.rp_to_prob(rps)

    site_properties = {
        "vs30": site_series.loc["vs30"],
        "vs30measured": True,
        "z1p0": site_series.loc["z1p0"],
        "z2p5": site_series.loc["z2p5"],
        "backarc": site_series.loc["backarc"],
    }
    site_nztm_values = site_series[["nztm_x", "nztm_y"]].values

    # DS Hazard
    oq_rupture_df = sha.nshm_2010.get_oq_ds_rupture_df(
        ds_source_df, site_nztm_values, site_properties
    )
    if max_rrup is not None:
        oq_rupture_df = oq_rupture_df.loc[oq_rupture_df["rrup"] <= max_rrup]

    ds_gm_params_df = sha.nshm_2010.get_emp_gm_params(
        oq_rupture_df, EMP_GMM_MAPPING, ims
    ).sort_index()

    im_levels = {im: sha.utils.get_im_levels(im, n_values=100) for im in ims}
    ds_hazard = sha.nshm_2010.compute_gmm_hazard(
        ds_gm_params_df, ds_erf_df.annual_rec_prob, ims, im_levels=im_levels
    )

    # Get disagg im levels & hazard
    disagg_contr = {}
    for im in ims:
        try:
            disagg_im_levels = sha.utils.exceedance_to_im(
                excd_probs, im_levels[im], ds_hazard[im].values
            )
        except ValueError:
            logger.warning(
                f"IM levels for {im} do not cover all exceedance probabilities at this site. "
                f"Skipping disaggregation for this IM."
            )
            disagg_contr[im] = pd.DataFrame(
                index=ds_gm_params_df.index,
                data=np.full((ds_gm_params_df.shape[0], len(excd_probs)), np.nan),
                columns=excd_probs,
            )
            continue
        disagg_gm_prob = sha.hazard.parametric_gm_excd_prob(
            disagg_im_levels,
            ds_gm_params_df,
            mean_col=f"{im}_mean",
            std_col=f"{im}_std_Total",
        )
        disagg_hz = sha.hazard.hazard_curve(
            disagg_gm_prob, ds_erf_df["annual_rec_prob"]
        )
        im_disagg_contr = (
            sha.disagg.disagg_exceedance_multi(
                disagg_gm_prob, ds_erf_df["annual_rec_prob"], disagg_hz
            )
            .astype(np.float32)
            .sort_index()
        )
        im_disagg_contr.columns = rps
        disagg_contr[im] = im_disagg_contr

    disagg_contr_da = xr.DataArray(
        data=np.stack([disagg_contr[im].values for im in ims], axis=2),
        dims=["rupture_id", "return_period", "im_type"],
        coords={
            "rupture_id": disagg_contr[ims[0]].index.values,
            "return_period": rps,
            "im_type": ims,
        },
    )

    disagg_contr_da.to_netcdf(output_dir / f"{site_series.name}.nc")
