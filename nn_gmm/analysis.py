import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants
from .empdb import DuckEmpiricalDB
from .imdb import DuckIMDB
from . import nn_gmm

logger = logging.getLogger(__name__)


def get_nn_sim_residuals(
    model_dir: Path,
    pred_df: pd.DataFrame = None,
    sim_df: pd.DataFrame = None,
    record_info_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Get the residuals of the specified NN model validation results
    """
    run_config = nn_gmm.GMMRunConfig.from_yaml(model_dir / "run_config.yaml")

    if pred_df is None:
        pred_df = pd.read_parquet(model_dir / "val_results.parquet")
    pred_df = pred_df.sort_index()
    val_record_int_ids = pred_df.index.values.astype(int)

    assert record_info_df is None or np.all(val_record_int_ids == record_info_df.index.values)
    assert sim_df is None or np.all(val_record_int_ids == sim_df.index.values)

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        if sim_df is None:
            sim_df = imdb.get_im_data(
                run_config.ims, val_record_int_ids
            )
        if record_info_df is None:
            record_info_df = imdb.get_record_info_df(
                record_int_ids=val_record_int_ids
            )

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
        val_emp_df = empdb.get_gm_params_tmp_table(sim_df.index.values.astype(int)).sort_index()
    assert val_emp_df.index.equals(sim_df.index)    

    emp_res_df = pd.DataFrame(
        data=np.log(sim_df[constants.PSA_KEYS].values)
        - val_emp_df[constants.GMM_PSA_MEAN_KEYS].values,
        index=val_emp_df.index,
        columns=constants.PSA_KEYS,
    )

    return emp_res_df


def run_nn_mera(
    result_dir: Path, site_term: bool = False, out_dir: Path = None, n_procs: int = 4
):
    import mera
    run_config = nn_gmm.GMMRunConfig.from_yaml(result_dir / "run_config.yaml")

    logging.info("Getting NN residuals")
    res_df, *_ = get_nn_sim_residuals(result_dir)

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(record_int_ids=res_df.index)
    res_df["rel_id"] = record_info_df.loc[res_df.index, "rel_id"]
    res_df["site_id"] = record_info_df.loc[res_df.index, "site_id"]

    mask = mera.mask_too_few_records(
        res_df,
        "rel_id",
        "site_id",
        min_num_records_per_event=5,
        min_num_records_per_site=5,
    )

    logging.info("Running MERA")
    start = time.time()
    mera_results = mera.run_mera(
        res_df,
        # run_config.ims[:4],
        run_config.ims,
        "rel_id",
        "site_id",
        mask=mask,
        compute_site_term=site_term,
        n_procs=n_procs,
    )
    logging.info(f"Took: {time.time() - start} to run MERA")

    out_dir = (
        result_dir / f"mera{'_site_term' if site_term else ''}"
        if out_dir is None
        else out_dir
    )
    out_dir.mkdir(exist_ok=True)
    mera_results.save_to_parquet(out_dir, save_fit=False)
    logging.info(f"Wrote MERA results to: {out_dir}")


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
