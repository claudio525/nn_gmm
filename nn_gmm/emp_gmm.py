from pathlib import Path

import pandas as pd
import numpy as np

import oq_wrapper as oqw


from . import data
from . import constants

def get_gmm_predictions(input_df: pd.DataFrame, gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM]) -> pd.DataFrame:
    """
    Get GMM predictions for the given input DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing the input features for the GMM.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the GMM predictions.
    """
    rupture_df = input_df.copy()

    OQ_NAME_MAPPING = {
        "magnitude": "mag",
        "z1p0": "z1pt0",
        "z2p5": "z2pt5",
        "dtop": "ztor",
    }
    rupture_df = rupture_df.rename(columns=OQ_NAME_MAPPING)
    rupture_df["vs30measured"] = False

    result = []
    for cur_tect_type in rupture_df["tect_type"].unique():
        cur_oqw_tect_type = oqw.constants.TectType(
            cur_tect_type
        )
        cur_gmm = gmm_mapping[cur_oqw_tect_type]

        cur_result = oqw.run_gmm(
            cur_gmm,
            cur_oqw_tect_type,
            rupture_df[rupture_df["tect_type"] == cur_tect_type],
            "pSA",
            periods=constants.PSA_PERIODS,
        )
        result.append(cur_result)

    result_df = pd.concat(result, axis=0).sort_index()
    result_df = pd.concat([input_df, result_df], axis=1)

    return result_df
    
    