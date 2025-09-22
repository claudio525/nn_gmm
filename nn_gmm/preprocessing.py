import logging
from collections.abc import Sequence

import pandas as pd

from qcore import coordinates as coords

from . import constants

logger = logging.getLogger(__name__)

def preprocess_site_features(site_df: pd.DataFrame, site_feature_keys: Sequence[str]):
    """
    Pre-process the site features in the site DataFrame.
    Does not modify the original DataFrame.

    Parameters
    ----------
    site_df : pd.DataFrame
        DataFrame containing the site features.
    site_feature_keys : Sequence[str]
        List of site feature keys to be pre-processed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the pre-processed site features.
    """
    pre_site_df = site_df.copy()
    pre_site_df = pre_site_df.loc[:, site_feature_keys]

    for cur_feature in site_feature_keys:
        # Standard min-max pre-processing
        if cur_feature in constants.MIN_MAX_PRE_PROCESS_CONFIG:
            cur_min, cur_max = constants.MIN_MAX_PRE_PROCESS_CONFIG[cur_feature]
            pre_site_df[cur_feature] = (
                2 * (pre_site_df[cur_feature] - cur_min) / (cur_max - cur_min) - 1
            )
        else:
            logger.error(f"Feature {cur_feature} not in pre-processing config.")
            raise ValueError(f"Feature {cur_feature} not in pre-processing config.")

    return pre_site_df


def preprocess_source_features(
    source_df: pd.DataFrame, source_feature_keys: Sequence[str]
):
    """
    Pre-process the event features in the event DataFrame.
    Does not modify the original DataFrame.

    Parameters
    ----------
    source : pd.DataFrame
        DataFrame containing the source features.
    source_feature_keys : Sequence[str]
        List of source feature keys to be pre-processed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the pre-processed source features.
    """
    pre_source_df = source_df.copy()
    pre_source_df["tect_type"] = pd.Categorical(pre_source_df["tect_type"], [str(v) for v in constants.TECT_TYPES])
    pre_source_df = pre_source_df.loc[:, source_feature_keys]

    for cur_key in source_feature_keys:
        if cur_key in constants.MIN_MAX_PRE_PROCESS_CONFIG:
            cur_min, cur_max = constants.MIN_MAX_PRE_PROCESS_CONFIG[cur_key]
            pre_source_df[cur_key] = (
                2 * (pre_source_df[cur_key] - cur_min) / (cur_max - cur_min) - 1
            )
        elif cur_key == "tect_type":
            one_hot_df = pd.get_dummies(pre_source_df["tect_type"], prefix="is").astype(float)
            assert one_hot_df.index.equals(pre_source_df.index)
            pre_source_df = pd.concat([pre_source_df, one_hot_df], axis=1)
            pre_source_df = pre_source_df.drop(columns=["tect_type"])
        else:
            logger.error(f"Feature {cur_key} not in pre-processing config.")
            raise ValueError(f"Feature {cur_key} not in pre-processing config.")

    return pre_source_df


def preprocess_event_site_features(
    site_event_df: pd.DataFrame, event_site_feature_keys: Sequence[str], max_rrup: float
):
    pre_site_event_df = site_event_df.copy()
    pre_site_event_df = pre_site_event_df.loc[:, event_site_feature_keys]

    for cur_key in event_site_feature_keys:
        if cur_key in ["rrup", "rjb"]:
            pre_site_event_df[cur_key] = (
                2 * (pre_site_event_df[cur_key] - 0) / (max_rrup - 0) - 1
            )
        elif cur_key in ["rx", "ry"]:
            pre_site_event_df[cur_key] = (
                2 * (pre_site_event_df[cur_key] - (-max_rrup)) / (max_rrup - (-max_rrup)) - 1
            )
        else:
            logger.error(f"Feature {cur_key} not in pre-processing config.")
            raise ValueError(f"Feature {cur_key} not in pre-processing config.")

    return pre_site_event_df