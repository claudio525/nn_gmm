import multiprocessing as mp
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import rpy2.robjects.conversion as cv
from pymer4.models import Lmer

from . import constants
from .empdb import DuckEmpiricalDB
from .imdb import DuckIMDB
from . import nn_gmm

logger = logging.getLogger(__name__)


def get_site_bias_std(res_df: pd.DataFrame, site_df: pd.DataFrame):
    """
    Get the site bias and residual standard deviation
    for the given residuals.
    """
    # Site bias
    site_bias = res_df.groupby("site_int_id").mean()
    site_bias["lon"] = site_df.loc[site_bias.index, "lon"].values
    site_bias["lat"] = site_df.loc[site_bias.index, "lat"].values

    # Site residual standard deviation
    site_res_std = res_df.groupby("site_int_id").std()
    site_res_std["lon"] = site_df.loc[site_res_std.index, "lon"].values
    site_res_std["lat"] = site_df.loc[site_res_std.index, "lat"].values

    return site_bias, site_res_std


def get_nn_sim_residuals(
    model_dir: Path,
    pred_df: pd.DataFrame = None,
    sim_df: pd.DataFrame = None,
    record_info_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Get the residuals of the specified NN model validation results
    """
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")

    if pred_df is None:
        pred_df = pd.read_parquet(model_dir / "val_results.parquet")
    pred_df = pred_df.sort_index()
    val_record_int_ids = pred_df.index.values.astype(int)

    assert record_info_df is None or np.all(
        val_record_int_ids == record_info_df.index.values
    )
    assert sim_df is None or np.all(val_record_int_ids == sim_df.index.values)

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        if sim_df is None:
            sim_df = imdb.get_im_data(run_config.ims, val_record_int_ids)
        if record_info_df is None:
            record_info_df = imdb.get_record_info_df(record_int_ids=val_record_int_ids)

    record_info_df = record_info_df.sort_index()
    sim_df = sim_df.sort_index()
    assert sim_df.index.equals(pred_df.index)

    res_df = pd.DataFrame(
        data=np.log(sim_df[run_config.ims].values)
        - pred_df[run_config.pred_mean_keys].values,
        index=pred_df.index,
        columns=run_config.ims,
    )

    res_df["event_int_id"] = record_info_df.loc[res_df.index, "event_int_id"]
    res_df["site_int_id"] = record_info_df.loc[res_df.index, "site_int_id"]
    res_df["cv_iter"] = pred_df.loc[res_df.index, "cv_iter"]

    return res_df, pred_df, sim_df, record_info_df


def get_emp_sim_residuals(empdb_ffp: Path, sim_df: pd.DataFrame):
    """
    Get the residuals of the empirical GMM with respect to
    the specified simulation results.
    """
    with DuckEmpiricalDB(empdb_ffp, readonly=True) as empdb:
        val_emp_df = empdb.get_gm_params_tmp_table(
            sim_df.index.values.astype(int)
        ).sort_index()
    assert val_emp_df.index.equals(sim_df.index)

    emp_res_df = pd.DataFrame(
        data=np.log(sim_df[constants.PSA_KEYS].values)
        - val_emp_df[constants.GMM_PSA_MEAN_KEYS].values,
        index=val_emp_df.index,
        columns=constants.PSA_KEYS,
    )

    return emp_res_df


def run_nn_mera(
    result_dir: Path,
    site_term: bool = False,
    out_dir: Path = None,
    n_procs: int = 4,
    ims: list[str] = None,
):
    import mera

    run_config = nn_gmm.load_config(result_dir / "run_config.yaml")

    logging.info("Getting NN residuals")
    res_df, _, __, record_info_df = get_nn_sim_residuals(result_dir)

    # with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
    # record_info_df = imdb.get_record_info_df(record_int_ids=res_df.index)
    res_df["rel_id"] = record_info_df.loc[res_df.index, "rel_id"]
    res_df["site_id"] = record_info_df.loc[res_df.index, "site_id"]

    mask = mera.mask_too_few_records(
        res_df[list(run_config.ims) + ["rel_id", "site_id"]],
        "rel_id",
        "site_id",
        min_num_records_per_event=5,
        min_num_records_per_site=5,
    )

    logging.info("Running MERA")
    ims = constants.PLOT_IMS if ims is None else ims
    start = time.time()
    event_mera_results = mera.run_mera(
        res_df,
        ims,
        "rel_id",
        "site_id",
        mask=mask,
        # compute_site_term=False,
        compute_site_term=site_term,
        n_procs=n_procs,
    )
    logging.info(f"Took: {time.time() - start} to run MERA")

    mera_results = None
    # if site_term:
    #     logger.info("Computing site terms")
    #     start_time = time.time()

    #     # Compute remaining residuals
    #     event_rem_res_df = event_mera_results.rem_res_df.copy()
    #     event_rem_res_df["site_id"] = res_df.loc[event_rem_res_df.index, "site_id"]

    #     site_terms = []
    #     rem_residuals = []
    #     bias_std_values = []
    #     if n_procs == 1:
    #         for im in ims:
    #             cur_res_df = event_rem_res_df.loc[mask[im], [im, "site_id"]]

    #             cur_rem_res, cur_site_res, cur_std_values = _run_site_mera(
    #                 cur_res_df, im
    #             )

    #             site_terms.append(cur_site_res)
    #             rem_residuals.append(cur_rem_res)
    #             bias_std_values.append(cur_std_values)
    #     else:
    #         with mp.Pool(n_procs) as pool:
    #             results = pool.starmap(
    #                 _run_site_mera,
    #                 [
    #                     (
    #                         event_rem_res_df.loc[mask[im], [im, "site_id"]],
    #                         im,
    #                     )
    #                     for im in ims
    #                 ],
    #             )
    #         rem_residuals = [res[0] for res in results]
    #         site_terms = [res[1] for res in results]
    #         bias_std_values = [res[2] for res in results]

    #     logger.info(f"Took: {time.time() - start_time} to compute site terms")

    #     rem_res_df = pd.concat(rem_residuals, axis=1)
    #     rem_res_df["site_id"] = event_rem_res_df.loc[rem_res_df.index, "site_id"]
    #     rem_res_df["rel_id"] = record_info_df.loc[rem_res_df.index, "rel_id"]
    #     site_res_df = pd.concat(site_terms, axis=1)
    #     bias_std_df = pd.concat(bias_std_values, axis=1).T

    #     assert bias_std_df.index.equals(event_mera_results.bias_std_df.index)
    #     bias_std_df["tau"] = event_mera_results.bias_std_df["tau"]
    #     bias_std_df["bias_event"] = event_mera_results.bias_std_df["bias"]
    #     bias_std_df["bias"] = bias_std_df["bias_event"] + bias_std_df["bias_site"]
    #     bias_std_df["sigma"] = np.sqrt(
    #         bias_std_df["tau"] ** 2
    #         + bias_std_df["phi_S2S"] ** 2
    #         + bias_std_df["phi_w"] ** 2
    #     )

    #     mera_results = mera.MeraResults(
    #         event_mera_results.event_res_df,
    #         None,
    #         rem_res_df,
    #         bias_std_df,
    #         None,
    #         site_res_df,
    #         None,
    #     )
    
    if mera_results is None:
        mera_results = event_mera_results

    out_dir = (
        result_dir / f"mera{'_site_term' if site_term else ''}"
        if out_dir is None
        else out_dir
    )
    out_dir.mkdir(exist_ok=True)
    mera_results.save_to_parquet(out_dir, save_fit=False)
    logging.info(f"Wrote MERA results to: {out_dir}")


def _run_site_mera(res_df: pd.DataFrame, im: str):
    site_model = Lmer(f"{im} ~ 1 + (1|site_id)", data=res_df)
    site_model.fit(summary=False)

    site_res = site_model.ranef.iloc[:, 0].rename(im)
    rem_res = pd.Series(index=res_df.index, data=site_model.residuals, name=im)
    bias_std_values = pd.Series(
        index=["bias_site","phi_S2S", "phi_w"],
        data=[
            site_model.coefs.iloc[0, 0],
            site_model.ranef_var.loc["site_id", "Std"],
            site_model.ranef_var.loc["Residual", "Std"],
        ],
        name=im,
    )

    return rem_res, site_res, bias_std_values


def get_mag_input_df(
    mag_values: np.ndarray, run_config: nn_gmm.GMMRunConfig | None = None, **kwargs
) -> pd.DataFrame:
    """
    Create a DataFrame with the given magnitude values and additional parameters.

    Parameters
    ----------
    mag_values : np.ndarray
        Array of magnitude values.
    run_config : nn_gmm.RunConfig
        Run configuration of the model
    **kwargs : dict
        Additional parameters to include in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the magnitude values and additional parameters.
    """
    input_df = pd.DataFrame(
        {
            "magnitude": mag_values,
        }
    )

    for k, v in kwargs.items():
        input_df[k] = v

    # Check inputs
    if run_config is not None:
        for key in run_config.site_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required site input: {key}")
        for key in run_config.source_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required source input: {key}")
        for key in run_config.source_to_site_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required event-site input: {key}")

    return input_df


def get_rrup_input_df(
    rrup_values: np.ndarray, run_config: nn_gmm.GMMRunConfig | None = None, **kwargs
) -> pd.DataFrame:
    """
    Create a DataFrame with the given rrup values and additional parameters.

    Parameters
    ----------
    rrup_values : np.ndarray
        Array of rrup values.
    run_config : nn_gmm.RunConfig
        Run configuration of the model
    **kwargs : dict
        Additional parameters to include in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rrup values and additional parameters.
    """
    input_df = pd.DataFrame(
        {
            "rrup": rrup_values,
            "rjb": rrup_values,
            "rx": rrup_values,
            "ry": rrup_values,
        }
    )

    for k, v in kwargs.items():
        input_df[k] = v

    # Check inputs
    if run_config is not None:
        for key in run_config.site_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required site input: {key}")
        for key in run_config.source_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required source input: {key}")
        for key in run_config.source_to_site_inputs:
            if key not in input_df.columns:
                raise ValueError(f"Missing required event-site input: {key}")

    return input_df
