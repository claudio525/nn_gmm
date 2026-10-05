import gc
import logging
import multiprocessing as mp
import pickle
from functools import partial
from pathlib import Path

import ml_tools as mlt
import numpy as np
import oq_wrapper as oqw
import pandas as pd
import torch
import xarray as xr
from qcore import nhm
from scipy import stats
from tqdm import tqdm

from . import constants, nn_gmm, utils
from .imdb import DuckIMDB

logger = logging.getLogger(__name__)

EMP_GMM_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_SLAB: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMMLogicTree.NSHM2022,
}


def compute_cs_parametric_hazard(
    imdb_ffp: Path, output_ffp: Path, site_grid_level: int
):
    import seismic_hazard_analysis as sha

    erf = nhm.load_nhm_df(constants.FLT_ERF_FFP)
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(
            add_nztm=True,
            min_grid_level=site_grid_level,
            max_grid_level=site_grid_level,
        )

        # Target sites
        site_df = site_df.loc[site_df.grid_level == site_grid_level]
        sites = site_df.site_id.values.astype(str)

        # Relevant events
        event_df = imdb.get_event_df()
        event_df = event_df.loc[event_df.sim_type == 4]

        # Relevant records & IM data
        record_info_df = imdb.get_record_info_df(
            sites=sites, events=event_df.event_id.values.astype(str)
        )

        # Compute hazard for each site
        hazard_results = None
        for cur_site in tqdm(sites):
            cur_record_int_ids = record_info_df.loc[
                record_info_df.site_id == cur_site
            ].index.values
            cur_im_data = imdb.get_im_data(
                constants.PSA_KEYS, cur_record_int_ids, log_ims=True
            )
            cur_im_data["event_id"] = record_info_df.loc[
                cur_record_int_ids
            ].event_id.values

            cur_gm_params = cur_im_data.groupby("event_id", observed=True).agg(
                ["mean", "std"]
            )
            cur_gm_params.columns = [
                "_".join(map(str, col)).strip("_") for col in cur_gm_params.columns
            ]

            cur_hazard = sha.nshm_2010.compute_gmm_hazard(
                cur_gm_params,
                1 / erf.recur_int_median,
                constants.PSA_KEYS,
                mean_col_suffix="_mean",
                std_col_suffix="_std",
            )

            if hazard_results is None:
                hazard_results = {
                    cur_im: pd.DataFrame(index=sites, columns=cur_hazard[cur_im].index)
                    for cur_im in cur_hazard
                }

            for cur_im in cur_hazard:
                hazard_results[cur_im].loc[cur_site] = cur_hazard[cur_im]

        mlt.utils.write_pickle(hazard_results, output_ffp)


def compute_fault_nn_hazard(
    cv_results_dir: Path,
    site_grid_level: int,
    output_ffp: Path,
    device: str | None = None,
):
    """
    Computes hazard using the CV surrogate models for the fault sources.
    Each CV model is only used to predict GM parameters for fault and sites
    that were not used in training for that CV model.
    """
    import seismic_hazard_analysis as sha

    run_config = nn_gmm.load_config(cv_results_dir / "run_config.yaml")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(
            add_nztm=True,
            min_grid_level=site_grid_level,
            max_grid_level=site_grid_level,
        )

        # Target sites
        site_df = site_df.loc[site_df.grid_level == site_grid_level]
        sites = site_df.site_id.values.astype(str)

        # Relevant events
        event_df = imdb.get_event_df()
        event_df = event_df.loc[event_df.sim_type == 4]

        # Get rupture distances
        record_info_df = imdb.get_record_info_df(
            sites=sites, events=event_df.event_id.values.astype(str)
        )
        record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
            record_info_df.site_int_id.values, record_info_df.event_int_id.values
        )
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=record_info_df.site_event_int_id.values
        )

    gm_params = []
    cv_model_dirs = [ffp for ffp in cv_results_dir.glob("cv_*") if ffp.is_dir()]

    for cur_cv_dir in tqdm(cv_model_dirs, desc="Computing GM params"):
        if isinstance(run_config, nn_gmm.LocAdjRunConfig):
            cur_cv_train_events = np.load(
                run_config.base_model_dir / cur_cv_dir.name / "train_events.npy"
            )
            cur_cv_train_sites = np.load(
                run_config.base_model_dir / cur_cv_dir.name / "train_sites.npy"
            )
        else:
            cur_cv_train_events = np.load(cur_cv_dir / "train_events.npy")
            cur_cv_train_sites = np.load(cur_cv_dir / "train_sites.npy")

        cur_events = event_df.loc[
            ~event_df.event_id.isin(cur_cv_train_events)
        ].event_id.values
        cur_sites = site_df.loc[
            ~site_df.site_id.isin(cur_cv_train_sites)
        ].site_id.values

        cur_input_df = nn_gmm.get_fault_events_sites_input_df(
            run_config, cur_events, cur_sites
        )

        # Drop any that already have results
        if len(gm_params) > 0:
            cur_input_df = cur_input_df.loc[
                ~cur_input_df.index.isin(
                    np.concatenate([item.index.values for item in gm_params])
                )
            ]

        cur_results_df = nn_gmm.run_predictions_dir(
            cur_cv_dir, cur_input_df, device=device
        )
        gm_params.append(cur_results_df)
    gm_params = pd.concat(gm_params, axis=0)

    # Compute hazard
    erf = nhm.load_nhm_df(constants.FLT_ERF_FFP)
    hazard_results = None
    for cur_site in tqdm(sites, desc="Computing PSHA"):
        cur_site_gm_params = gm_params.loc[gm_params.site_id == cur_site].set_index(
            "event_id"
        )
        assert (
            cur_site_gm_params.shape[0]
            == site_event_df.loc[site_event_df.site_id == cur_site].shape[0]
        ), "Missing GM parameters for some ruptures for site {cur_site}"
        cur_hazard = sha.nshm_2010.compute_gmm_hazard(
            cur_site_gm_params,
            1 / erf.recur_int_median,
            run_config.ims,
            mean_col_suffix="_pred",
            std_col_suffix="_pred_std",
        )

        if hazard_results is None:
            hazard_results = {
                cur_im: pd.DataFrame(index=sites, columns=cur_hazard[cur_im].index)
                for cur_im in cur_hazard
            }

        for cur_im in cur_hazard:
            hazard_results[cur_im].loc[cur_site] = cur_hazard[cur_im]

    mlt.utils.write_pickle(hazard_results, output_ffp)


def compute_fault_nn_hazard_error(
    hazard_results_1: dict, hazard_results_2: dict, rps: list[int]
):
    """
    Compute the error between the two fault hazard results.
    """
    ims = list(hazard_results_1.keys())
    sites = hazard_results_1[ims[0]].index.values.astype(str)

    model_1_uhs = compute_flt_uhs(hazard_results_1, sites, rps)
    model_2_uhs = compute_flt_uhs(hazard_results_2, sites, rps)
    assert model_1_uhs.coords.equals(model_2_uhs.coords)

    residuals = np.log(model_2_uhs.values) - np.log(model_1_uhs.values)
    mean_residual = pd.DataFrame(
        index=ims, data=np.nanmean(residuals, axis=0), columns=rps
    )
    # mean_quantiles = xr.DataArray(
    #     data=np.nanquantile(residuals, [0.05, 0.5, 0.95], axis=0),
    #     dims=["quantile", "im", "rp"],
    #     coords={
    #         "quantile": [0.05, 0.5, 0.95],
    #         "im": nn_uhs.coords["im"],
    #         "rp": rps,
    #     },
    # )

    residual_std = pd.DataFrame(
        index=ims, data=np.nanstd(residuals, axis=0), columns=rps
    )

    return residuals, mean_residual, residual_std


def compute_nn_ds_uhs(
    ds_results_dir: Path, rps: list[int], sites: list[str] | None = None
):
    """Compute NN-GMM DS UHS for all sites in the specified results directory."""
    import seismic_hazard_analysis as sha

    sites = (
        [item.stem for item in ds_results_dir.glob("*.pkl")] if sites is None else sites
    )
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
            logger.warning(
                f"No DS hazard results found for site {site}, skipping UHS computation."
            )
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


def compute_emp_ds_uhs(
    ds_results_dir: Path, rps: list[int], sites: list[str] | None = None
):
    import seismic_hazard_analysis as sha

    sites = (
        [item.stem for item in ds_results_dir.glob("*.pkl")] if sites is None else sites
    )
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
    logger.info(f"Computing Cybershake fault UHS for {len(sites)} sites")
    cs_flt_hazard = pd.read_pickle(
        constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )
    return compute_flt_uhs(cs_flt_hazard, sites, rps)
    # uhs_da = {}
    # for site in sites:
    #     cur_flt_hazard = {
    #         cur_im: cs_flt_hazard[cur_im].loc[site] for cur_im in constants.PSA_KEYS
    #     }
    #     uhs_da[site] = sha.uhs.compute_uhs(cur_flt_hazard, excd_rates, rps=rps)

    # # Convert to DataArray
    # uhs_da = xr.DataArray(
    #     dims=["site", "im", "rp"],
    #     coords={
    #         "site": list(uhs_da.keys()),
    #         "im": constants.PSA_KEYS,
    #         "rp": rps,
    #     },
    #     data=np.stack([uhs_da[site].values for site in uhs_da]),
    # )

    # return uhs_da


def compute_flt_uhs(flt_hazard: pd.DataFrame, sites: list[str], rps: list[int]):
    import seismic_hazard_analysis as sha

    excd_rates = [sha.utils.rp_to_prob(rp) for rp in rps]

    uhs_da = {}
    for site in sites:
        cur_flt_hazard = {
            cur_im: flt_hazard[cur_im].loc[site] for cur_im in constants.PSA_KEYS
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
        data=np.stack([uhs_da[site].values for site in uhs_da]),
    )

    return uhs_da


def run_nn_sites_ds_hazard(
    model_dir: Path,
    sites: list[str],
    device: str,
    n_procs: int = 1,
    sigma_ept_ffp: Path | None = None,
    quantile: float | None = None,
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
    sigma_ept_ffp : Path, optional
        Path to the sigma_ept.csv file (from compute-sigma-ept).
        If given, the hazard is computed at the specified quantile of the
        surrogate epistemic uncertainty and saved in ref_sites_q{quantile}.
    quantile : float, optional
        Hazard quantile to compute, e.g. 0.95, requires sigma_ept_ffp.
    """
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id")

    # Load DS source data
    ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()

    out_dir = model_dir / (
        "ds_hazard/ref_sites" if quantile is None else f"ds_hazard/ref_sites_q{str(quantile).replace('.', 'p')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_procs == 1:
        hazard_results = {}
        for cur_site in sites:
            logger.info(f"Running DS hazard for site: {cur_site}")
            hazard_results[cur_site] = _run_site_ds_hazard(
                model_dir,
                site_df.loc[cur_site],
                ds_source_df,
                ds_erf_df,
                device,
                sigma_ept_ffp=sigma_ept_ffp,
                quantile=quantile,
            )
    else:
        logger.info(
            f"Running DS hazard for {len(sites)} sites using {n_procs} processes..."
        )
        with mp.Pool(n_procs) as p:
            hazard_results = p.starmap(
                _run_site_ds_hazard,
                [
                    (
                        model_dir,
                        site_df.loc[cur_site],
                        ds_source_df,
                        ds_erf_df,
                        device,
                        True,
                        sigma_ept_ffp,
                        quantile,
                    )
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
                    (
                        model_dir,
                        site_df.loc[cur_site],
                        ds_source_df,
                        ds_erf_df,
                        device,
                        False,
                    )
                    for cur_site in sites
                ],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    # Save per IM
    (out_dir := model_dir / "ds_hazard/uniform_grid").mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hazard results in {out_dir}")
    for im in run_config.ims:
        im_hazard_df = pd.concat(
            [hazard_results[cur_site]["total"][im] for cur_site in sites], axis=1
        )
        im_hazard_df.columns = sites

        im_hazard_df.to_parquet(
            out_dir / f"{utils.get_im_filename(im)}_ds_hazard.parquet"
        )


def _run_site_ds_hazard(
    model_dir: Path,
    site_series: pd.Series,
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    device: str,
    tect_type_hazard: bool = True,
    sigma_ept_ffp: Path | None = None,
    quantile: float | None = None,
) -> dict[str, dict[str, pd.Series]]:
    """
    Run DS hazard calculations for a single site using a NN-GMM model.

    If sigma_ept_ffp is given, the predicted median of every rupture is shifted
    by norm.ppf(quantile) * sigma_ept(IM, tect type, magnitude bin)

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
    rupture_df = nn_gmm.get_ds_input_df(run_config, site_series, ds_source_df)

    pred_df = nn_gmm.run_predictions_dir(model_dir, rupture_df, device)

    if sigma_ept_ffp is not None:
        sigma_ept_df = pd.read_csv(sigma_ept_ffp, index_col=[0, 1, 2]).xs(
            "sigma_ept", level="stat"
        )
        mag_bin_edges = np.unique(sigma_ept_df[["mag_bin_min", "mag_bin_max"]])
        sigma_ept_df = (
            sigma_ept_df.reset_index("mag_bin", drop=True)
            .set_index("mag_bin_min", append=True)
            .sort_index()
        )

        # Bins constrained by a single event (i.e. slab M < 6) use the next magnitude bin
        sigma_ept_df.loc[sigma_ept_df.n_events < 2, run_config.ims] = np.nan
        sigma_ept_df = sigma_ept_df.groupby("tect_type")[run_config.ims].bfill()

        rupture_mag_bin_min = pd.cut(
            pred_df.magnitude, mag_bin_edges, right=False, labels=mag_bin_edges[:-1]
        ).astype(float)
        rupture_sigma_ept = sigma_ept_df.reindex(
            pd.MultiIndex.from_arrays(
                [pred_df.tect_type.astype(str), rupture_mag_bin_min]
            )
        ).values
        assert not np.isnan(rupture_sigma_ept).any(), "Missing sigma_ept for some ruptures"

        pred_df[run_config.pred_mean_keys] += stats.norm.ppf(quantile) * rupture_sigma_ept

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


def compute_fault_emp_hazard(imdb_ffp: Path, site_grid_level: int, output_ffp: Path):
    """
    Compute empirical hazard for fault sources using empirical GMMs
    """
    import seismic_hazard_analysis as sha

    erf = nhm.load_nhm_df(constants.FLT_ERF_FFP)
    flt_definitions = nhm.load_nhm(constants.FLT_ERF_FFP)
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(
            add_nztm=True,
            min_grid_level=site_grid_level,
            max_grid_level=site_grid_level,
        ).set_index("site_id")
        site_df["backarc"] = False
        site_df["vs30measured"] = False
        site_df["depth"] = 0

        # Relevant events
        event_df = imdb.get_event_df()
        event_df = event_df.loc[event_df.sim_type == 4]

        # Drop faults that are not in the IMDB to ensure fair comparison
        flt_definitions = {
            flt_id: flt
            for flt_id, flt in flt_definitions.items()
            if flt_id in event_df.event_id.values
        }

    hazard_results = None
    for cur_site in tqdm(site_df.index.values.astype(str), desc="Computing hazard"):
        cur_hazard = sha.nshm_2010.compute_gmm_flt_hazard(
            site_df.loc[cur_site, ["nztm_x", "nztm_y", "depth"]].astype(float).values,
            site_df.loc[
                cur_site, ["vs30", "z1p0", "z2p5", "backarc", "vs30measured"]
            ].to_dict(),
            erf,
            constants.GMM_MAPPING,
            constants.PSA_KEYS,
            flt_definitions=flt_definitions,
        )

        if hazard_results is None:
            hazard_results = {
                cur_im: pd.DataFrame(
                    index=site_df.index, columns=cur_hazard[cur_im].index
                )
                for cur_im in cur_hazard
            }

        for cur_im in cur_hazard:
            hazard_results[cur_im].loc[cur_site] = cur_hazard[cur_im]

    mlt.utils.write_pickle(hazard_results, output_ffp)


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
                [(site_df.loc[cur_site],) for cur_site in sites],
            )
        hazard_results = {site: result for site, result in zip(sites, hazard_results)}

    # Save per IM
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hazard results in {output_dir}")
    for im in constants.PSA_KEYS:
        im_hazard_df = pd.concat(
            [hazard_results[cur_site]["total"][im] for cur_site in sites], axis=1
        )
        im_hazard_df.columns = sites

        im_hazard_df.to_parquet(
            output_dir / f"{utils.get_im_filename(im)}_ds_hazard.parquet"
        )


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


def compute_site_disagg(
    full_model_dir: Path, sites: list[str], im: str, rp: float, device: str
):
    """
    Compute disaggregation for specified sites using the
    NN-GMM model for DS hazard and simulations for fault sources
    """
    import seismic_hazard_analysis as sha

    # Setup
    cs_hazard = pd.read_pickle(constants.CS_FLT_HAZARD_FFP)[im]
    (output_dir := full_model_dir / "disagg").mkdir(parents=True, exist_ok=True)
    run_config = nn_gmm.load_config(full_model_dir / "run_config.yaml")

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True).set_index("site_id").loc[sites]
        assert np.all(~np.isin(sites, np.load(full_model_dir / "train_sites.npy")))

        # Relevant events
        event_df = imdb.get_event_df().set_index("event_id")
        event_df = event_df.loc[event_df.sim_type == 4]

        # Relevant records
        record_info_df = imdb.get_record_info_df(
            sites=sites, events=event_df.index.values.astype(str)
        )
        site_event_df = imdb.get_site_event_df(sites=sites)

        # Load erf
        ds_source_df, ds_erf_df = nn_gmm.utils.get_ds_source_data()
        flt_erf_df = nhm.load_nhm_df(constants.FLT_ERF_FFP)
        annual_rec_prob = pd.concat(
            [ds_erf_df["annual_rec_prob"], 1 / flt_erf_df["recur_int_median"]]
        )

        # Compute disagg
        for cur_site in sites:
            # Compute the im-level
            cur_im_level = sha.utils.exceedance_to_im(
                sha.utils.rp_to_prob(rp),
                cs_hazard.loc[cur_site].index.values.astype(float),
                cs_hazard.loc[cur_site].values.astype(float),
            )

            # Compute P(IM > im_level) for all fault sources
            # using CS simulations
            cur_flt_im_df = imdb.get_im_data(
                ims=[im],
                record_int_ids=record_info_df.loc[
                    record_info_df.site_id == cur_site
                ].index.values,
            )
            cur_flt_im_df["event_id"] = record_info_df.loc[
                cur_flt_im_df.index.values, "event_id"
            ].values
            cur_flt_im_df["rel_id"] = record_info_df.loc[
                cur_flt_im_df.index.values, "rel_id"
            ].values
            cur_flt_im_df = cur_flt_im_df.set_index(["event_id", "rel_id"]).squeeze()

            cur_flt_gm_prob_excd = sha.hazard.non_parametric_gm_excd_prob(
                np.asarray([cur_im_level]), cur_flt_im_df
            )
            cur_flt_events = cur_flt_gm_prob_excd.index.values.astype(str)

            # NN-Model DS Predictions
            cur_ds_nn_input_df = nn_gmm.get_ds_input_df(
                run_config, site_df.loc[cur_site], ds_source_df
            )
            cur_ds_nn_preds = nn_gmm.run_predictions_dir(
                full_model_dir, cur_ds_nn_input_df, device
            )

            # Compute P(IM > im_level) for all DS sources using NN-GMM predictions
            cur_ds_nn_gm_prob_excd = sha.hazard.parametric_gm_excd_prob(
                np.asarray([cur_im_level]),
                cur_ds_nn_preds[[f"{im}_pred", f"{im}_pred_std"]],
                mean_col=f"{im}_pred",
                std_col=f"{im}_pred_std",
            )
            cur_cs_nn_gm_prob_exd = pd.concat(
                [cur_flt_gm_prob_excd, cur_ds_nn_gm_prob_excd], axis=0
            )
            cur_cs_nn_disagg = sha.disagg.disagg_exceedance(
                cur_cs_nn_gm_prob_exd.squeeze(),
                annual_rec_prob.squeeze(),
            ).to_frame(name="contribution")

            # Add magnitude/distance columns
            cur_cs_nn_disagg = cur_cs_nn_disagg.merge(
                cur_ds_nn_input_df[["magnitude", "rrup", "tect_type"]],
                left_index=True,
                right_index=True,
                how="left",
            ).astype({"tect_type": "str"})
            cur_cs_nn_disagg.loc[cur_flt_events, ["magnitude", "tect_type"]] = event_df.loc[
                cur_flt_events, ["magnitude", "tect_type"]
            ].astype({"tect_type": "str"})
            cur_cs_nn_disagg.loc[cur_flt_events, "rrup"] = (
                site_event_df.loc[site_event_df.site_id == cur_site]
                .set_index("event_id")
                .loc[cur_flt_events]
            )

            # Save
            cur_cs_nn_disagg.index = cur_cs_nn_disagg.index.astype(str)
            cur_cs_nn_disagg.to_parquet(
                output_dir
                / f"{cur_site}_{utils.get_im_filename(im)}_RP{int(rp)}_CS_NN_disagg.parquet"
            )

            # Compute P(IM > im_level) for all DS sources using empirical GMMs
            cur_site_properties = {
                "vs30": site_df.loc[cur_site, "vs30"],
                "vs30measured": False,
                "z1p0": site_df.loc[cur_site, "z1p0"],
                "z2p5": site_df.loc[cur_site, "z2p5"],
                "backarc": False
            }
            cur_ds_emp_rupture_df = sha.nshm_2010.get_oq_ds_rupture_df(ds_source_df, site_df.loc[cur_site, ["nztm_x", "nztm_y"]], cur_site_properties)
            cur_ds_emp_gm_params_df = sha.nshm_2010.get_emp_gm_params(
                cur_ds_emp_rupture_df, constants.GMM_MAPPING, [im]
            ).sort_index()
            cur_ds_emp_gm_prob_excd = sha.hazard.parametric_gm_excd_prob(
                np.asarray([cur_im_level]),
                cur_ds_emp_gm_params_df[[f"{im}_mean", f"{im}_std_Total"]],
                mean_col=f"{im}_mean",
                std_col=f"{im}_std_Total",
            )
            cur_cs_emp_gm_prob_excd = pd.concat(
                [cur_flt_gm_prob_excd, cur_ds_emp_gm_prob_excd], axis=0
            )
            cur_cs_emp_disagg = sha.disagg.disagg_exceedance(
                cur_cs_emp_gm_prob_excd.squeeze(),
                annual_rec_prob.squeeze(),
            ).to_frame(name="contribution")

            # Add magnitude/distance columns
            cur_cs_emp_disagg = cur_cs_emp_disagg.merge(
                cur_ds_emp_rupture_df[["mag", "rrup", "tectonic_type"]],
                left_index=True,
                right_index=True,
                how="left",
            ).astype({"tectonic_type": "str"}).rename(columns={"tectonic_type": "tect_type", "mag": "magnitude"})
            cur_cs_emp_disagg.loc[cur_flt_events, ["magnitude", "tect_type"]] = event_df.loc[
                cur_flt_events, ["magnitude", "tect_type"]
            ].astype({"tect_type": "str"})
            cur_cs_emp_disagg.loc[cur_flt_events, "rrup"] = (
                site_event_df.loc[site_event_df.site_id == cur_site]
                .set_index("event_id")
                .loc[cur_flt_events]
            )

            # Save
            cur_cs_emp_disagg.index = cur_cs_emp_disagg.index.astype(str)
            cur_cs_emp_disagg.to_parquet(
                output_dir
                / f"{cur_site}_{utils.get_im_filename(im)}_RP{int(rp)}_CS_EMP_disagg.parquet"
            )


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
