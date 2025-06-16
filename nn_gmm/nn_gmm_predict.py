import logging
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from . import preprocessing
from . import nn_gmm

logger = logging.getLogger(__name__)

def run_predictions(model_dir: Path, input_df: pd.DataFrame, device: str):
    """
    Run predictions using the given model.

    Parameters
    ----------
    model_dir : Path
        Directory containing the model files.
    input_df : pd.DataFrame
        DataFrame containing the input features for the model.
    device : str
        Device to run the model on, e.g., 'cpu' or 'cuda'.
    """
    run_config = nn_gmm.RunConfig.from_yaml(model_dir / "run_config.yaml")

    # Pre-process the input DataFrame
    pre_site_df = preprocessing.pre_process_site_features(
        input_df, run_config.site_inputs
    )
    input_df["tect_type"] = pd.Categorical(input_df["tect_type"], [str(v) for v in nn_gmm.constants.TECT_TYPES])
    pre_source_df = preprocessing.pre_process_source_features(
        input_df, run_config.source_inputs
    )
    pre_source_site_df = preprocessing.pre_process_event_site_features(
        input_df, run_config.source_to_site_inputs, run_config.max_rrup
    )
    pre_input_df = pd.concat([pre_site_df, pre_source_df, pre_source_site_df], axis=1)
    X = torch.from_numpy(pre_input_df.values).to(dtype=torch.float32, device=device)

    # Load the model
    model = torch.load(model_dir / "model.pt", weights_only=False, map_location=torch.device(device))

    model.eval()
    with torch.no_grad():
        pred_mean, pred_ln_std = model(X).chunk(2, dim=-1)
        pred_std = torch.exp(pred_ln_std)

    pred_mean_df = pd.DataFrame(data=pred_mean.cpu().numpy(), columns=run_config.pred_mean_keys)
    pred_std_df = pd.DataFrame(data=pred_std.cpu().numpy(), columns=run_config.pred_std_keys)
    pred_df = pd.concat([input_df, pred_mean_df, pred_std_df], axis=1)

    return pred_df


def get_mag_input_df(mag_values: np.ndarray, run_config: nn_gmm.RunConfig | None =None, **kwargs) -> pd.DataFrame:
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
    input_df = pd.DataFrame({
        "magnitude": mag_values,
    })

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
    




























