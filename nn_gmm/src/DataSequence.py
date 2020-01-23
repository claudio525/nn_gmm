from typing import Dict

import keras
import pandas as pd


class DataSequence(keras.utils.Sequence):
    def __init__(
        self,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame,
        site_source_dict: Dict[str, pd.DataFrame],
        im_dict: Dict[str, pd.DataFrame],
        batch_size: int,
    ):
        self.im_dict = im_dict
        self.site_source_dict = site_source_dict
        self.source_df = source_df
        self.site_df = site_df

        self.batch_size = batch_size


        # Compute number of samples



