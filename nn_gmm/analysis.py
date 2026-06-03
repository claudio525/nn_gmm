import time
import logging
from pathlib import Path

import joblib
import torch
import shap
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from shap.utils._legacy import DenseData
import ml_tools as mlt

from . import constants
from .empdb import DuckEmpiricalDB
from .imdb import DuckIMDB
from . import nn_gmm
from . import nn_gmm_modules as modules

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
    test_results: bool = False,
) -> pd.DataFrame:
    """
    Get the residuals of the specified NN model validation results
    """
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")

    if pred_df is None:
        pred_df = (
            pd.read_parquet(model_dir / "test_results.parquet")
            if test_results
            else pd.read_parquet(model_dir / "val_results.parquet")
        )

    pred_df = pred_df.sort_index()
    record_int_ids = pred_df.index.values.astype(int)
    assert record_info_df is None or np.all(
        record_int_ids == record_info_df.index.values
    )
    assert sim_df is None or np.all(record_int_ids == sim_df.index.values)
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        if sim_df is None:
            sim_df = imdb.get_im_data(run_config.ims, record_int_ids)
        if record_info_df is None:
            record_info_df = imdb.get_record_info_df(record_int_ids=record_int_ids)

    record_info_df = record_info_df.sort_index()
    sim_df = sim_df.sort_index()
    assert sim_df.index.equals(pred_df.index)

    assert np.all(
        sim_df[run_config.ims] > 0
    ), "All simulated IM values must not be in log-space to compute residuals"
    res_df = pd.DataFrame(
        data=np.log(sim_df[run_config.ims].values)
        - pred_df[run_config.pred_mean_keys].values,
        index=pred_df.index,
        columns=run_config.ims,
    )

    res_df["event_int_id"] = record_info_df.loc[res_df.index, "event_int_id"]
    res_df["site_int_id"] = record_info_df.loc[res_df.index, "site_int_id"]
    if not test_results:
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


def run_event_mera(
    result_dir: Path,
    site_term: bool = False,
    out_dir: Path = None,
    n_procs: int = 4,
    ims: list[str] = None,
):
    import mera

    run_config = nn_gmm.load_config(result_dir / "run_config.yaml")

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df()

    if not (event_results_ffp := result_dir / "event_val_results.parquet").exists():
        logger.info("Computing event-level validation results...")
        raise ValueError(
            f"Event-level validation results not found at: {event_results_ffp}"
        )

    event_results_df = pd.read_parquet(result_dir / "event_val_results.parquet")

    # Prepare the residual dataframe
    res_df = event_results_df[run_config.ln_residual_keys].copy()
    # res_df = res_df.reset_index(drop=False)
    res_df["event_id"] = event_df.loc[
        res_df.index.get_level_values("event_int_id"), "event_id"
    ].values
    res_df["site_id"] = site_df.loc[
        res_df.index.get_level_values("site_int_id"), "site_id"
    ].values
    res_df = res_df.rename(
        columns=dict(zip(run_config.ln_residual_keys, run_config.ims))
    ).reset_index(drop=True)

    mask = mera.mask_too_few_records(
        res_df,
        "event_id",
        "site_id",
        min_num_records_per_event=5,
        min_num_records_per_site=5,
    )

    logger.info("Running MERA")
    ims = constants.MERA_IMS if ims is None else ims
    start = time.time()
    mera_results = mera.run_mera(
        res_df,
        ims,
        "event_id",
        "site_id",
        mask=mask,
        compute_site_term=site_term,
        n_procs=n_procs,
    )
    logger.info(f"Took: {time.time() - start} to run MERA")

    out_dir = (
        result_dir / f"event_mera{'_site_term' if site_term else ''}"
        if out_dir is None
        else out_dir
    )
    out_dir.mkdir(exist_ok=True)
    mera_results.save_to_parquet(out_dir, save_fit=False)
    logger.info(f"Wrote event MERA results to: {out_dir}")


def run_nn_mera(
    res_df: pd.DataFrame,
    record_info_df: pd.DataFrame,
    out_dir: Path,
    site_term: bool = False,
    n_procs: int = 4,
    ims: list[str] = None,
):
    """Run MERA on the specified NN model results."""
    import mera

    res_df["rel_id"] = record_info_df.loc[res_df.index, "rel_id"]
    res_df["site_id"] = record_info_df.loc[res_df.index, "site_id"]

    ims = constants.MERA_IMS if ims is None else ims
    mask = mera.mask_too_few_records(
        res_df[ims + ["rel_id", "site_id"]],
        "rel_id",
        "site_id",
        min_num_records_per_event=5,
        min_num_records_per_site=5,
    )

    logger.info("Running MERA")
    start = time.time()
    mera_results = mera.run_mera(
        res_df,
        ims,
        "rel_id",
        "site_id",
        mask=mask,
        compute_site_term=site_term,
        n_procs=n_procs,
    )
    logger.info(f"Took: {time.time() - start} to run MERA")

    out_dir.mkdir(exist_ok=True)
    mera_results.save_to_parquet(out_dir, save_fit=False)
    logger.info(f"Wrote MERA results to: {out_dir}")


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


def _get_pred_fn(
    model: modules.BaseNNModel, run_config: nn_gmm.BaseRunConfig, device: str
):
    """Gets a prediction function that can be used for SHAP value computation."""

    def _pred_fn(X: np.ndarray):
        X = torch.from_numpy(X).to(device)

        model.eval()
        with torch.no_grad():
            if model.uses_loc_inputs:
                raise NotImplementedError()
            else:
                pred_mean, pred_ln_std = model(X).chunk(2, dim=-1)

        pred_std = torch.exp(pred_ln_std).cpu().numpy()
        pred_mean = pred_mean.cpu().numpy()

        return np.concatenate([pred_mean, pred_std], axis=-1)

    return _pred_fn


def _shap_kmeans(X, k, round_values=True):
    """
    Modified version of shap.kmeans, uses MiniBatchKMeans
    (https://github.com/shap/shap/blob/6b74c3b86b2a621fd01f1003712be15fbace973b/shap/utils/_legacy.py#L10)

    Summarize a dataset with k mean samples weighted by the number of data points they
    each represent.

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
        Matrix of data samples to summarize (# samples x # features)

    k : int
        Number of means to use for approximation.

    round_values : bool
        For all i, round the ith dimension of each mean sample to match the nearest value
        from X[:,i]. This ensures discrete features always get a valid value.
    """
    group_names = [str(i) for i in range(X.shape[1])]

    # Specify `n_init` for consistent behaviour between sklearn versions
    kmeans = MiniBatchKMeans(
        init="k-means++", n_clusters=k, random_state=0, n_init=10
    ).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:, j]
                ind = np.argmin(np.abs(xj - kmeans.cluster_centers_[i, j]))
                kmeans.cluster_centers_[i, j] = X[ind, j]

    return (
        DenseData(
            kmeans.cluster_centers_,
            group_names,
            None,
            1.0 * np.bincount(kmeans.labels_),
        ),
        kmeans,
    )


def compute_cv_shape_values(
    results_dir: Path,
    device: str,
    n_procs: int = 8,
    n_clusters: int = 100,
    n_val_samples_per_cluster: int = 10,
):
    """
    Compute SHAP values for the specified CV results.

    Parameters
    ----------
    results_dir : Path
        Directory containing the CV results.
    device : str
        Device to use for computation (e.g., "cpu" or "cuda").
    n_procs : int
        Number of processes to use for parallel computation of SHAP values.
    n_clusters : int
        Number of clusters to use for SHAP background samples
        and validation sample selection.
    """

    run_config = nn_gmm.load_config(results_dir / "run_config.yaml")

    val_df = pd.read_parquet(results_dir / "val_results.parquet")
    input_df = nn_gmm.get_input_dfs(run_config, val_df.index.values.astype(int))

    comb_shap_explanations = []
    for cv_ix in val_df["cv_iter"].unique():
        logger.info(f"Computing SHAP values for CV iteration: {cv_ix}")
        cv_key = f"cv_{cv_ix:02d}"
        cur_result_dir = results_dir / cv_key

        # Loading
        val_record_int_ids = np.load(cur_result_dir / "val_record_ids.npy")
        val_mask = np.isin(input_df.index, val_record_int_ids)
        cur_run_config = nn_gmm.load_config(cur_result_dir / "run_config.yaml")
        cur_model = torch.load(
            cur_result_dir / "model.pt", weights_only=False, map_location=device
        )

        X, X_loc, feature_names, loc_feature_names = nn_gmm.get_input_tensor(
            cur_run_config, input_df, "cpu", return_feature_names=True
        )

        logger.info(
            "Running KMeans to select background samples and validation samples for SHAP"
        )
        X_bg, kmeans = _shap_kmeans(X.numpy(), n_clusters)

        # Create mask for validation records to evaluate
        val_cluster_indices = kmeans.labels_[val_mask]
        val_eval_mask = np.zeros(val_mask.sum(), dtype=bool)
        for cluster_ix in np.unique(val_cluster_indices):
            cluster_mask = val_cluster_indices == cluster_ix

            # Select a subset of validation samples from this cluster
            if np.sum(cluster_mask) > n_val_samples_per_cluster:
                selected_indices = np.random.choice(
                    np.flatnonzero(cluster_mask),
                    n_val_samples_per_cluster,
                    replace=False,
                )
                val_eval_mask[selected_indices] = True
            else:
                val_eval_mask[cluster_mask] = True

        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            _get_pred_fn(cur_model, cur_run_config, device),
            X_bg,
            feature_names=feature_names,
        )

        # Compute SHAP values for selected validation samples
        logger.info(
            f"Computing SHAP values for {val_eval_mask.sum()} validation samples..."
        )
        eval_X_df = pd.DataFrame(
            X[val_mask][val_eval_mask].numpy(),
            columns=feature_names,
            index=input_df.index[val_mask][val_eval_mask],
        )

        def explain_batch(batch: np.ndarray):
            return explainer(batch, silent=True)

        batches = np.array_split(eval_X_df, n_procs)
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(explain_batch)(b) for b in batches
        )

        # Combine results and save
        assert np.all(
            np.concatenate([res.data for res in results], axis=0) == eval_X_df.values
        )
        shap_explanation = shap.Explanation(
            values=np.concatenate([res.values for res in results], axis=0),
            base_values=np.concatenate([res.base_values for res in results], axis=0),
            data=eval_X_df,
            feature_names=feature_names,
        )
        mlt.utils.write_pickle(
            shap_explanation, cur_result_dir / "shap_explanation.pkl", clobber=True
        )

        comb_shap_explanations.append(shap_explanation)

    # Combine SHAP explanations from all CV iterations and save
    comb_shap_explanation = shap.Explanation(
        values=np.concatenate([exp.values for exp in comb_shap_explanations], axis=0),
        base_values=np.concatenate(
            [exp.base_values for exp in comb_shap_explanations], axis=0
        ),
        data=pd.concat([exp.data for exp in comb_shap_explanations], axis=0),
        feature_names=feature_names,
    )

    mlt.utils.write_pickle(
        comb_shap_explanation,
        results_dir / "comb_shap_explanation.pkl",
        clobber=True,
    )
