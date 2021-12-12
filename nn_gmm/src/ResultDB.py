from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import h5py

class ResultDB:
    """
    Class for writing and accessing model predictions
    from hdf5 database
    """

    def __init__(self, db_ffp: Path):
        self.db_ffp = db_ffp

        with h5py.File(self.db_ffp, "r") as db:
            self._columns = db["data"].attrs["columns"].astype(str)

    @property
    def columns(self):
        return self._columns

    def get_data(self, columns: Sequence[str]):
        with h5py.File(self.db_ffp, "r") as db:
            column_ind = np.flatnonzero(np.isin(self.columns, columns))
            data = db["data"][:, column_ind]
            ids = db["ids"][:]

            df = pd.DataFrame(data=data, index=ids, columns=self.columns[column_ind])

            if "site" in columns:
                site_values = db["site"][:]
                df["site"] = site_values

        return df

    @classmethod
    def get_data_static(cls, db_ffp: Path, columns: Sequence[str]):
        """Wrapper class for retrieving data without
        having to create an instance (in the Caller)"""
        result_db = cls(db_ffp)
        return result_db.get_data(columns)


    @staticmethod
    def write_data(result_df: pd.DataFrame, db_ffp: Path):
        ids = result_df.index.values.astype(str)
        data_columns = np.asarray([col for col in result_df.columns.values if col not in ["site"]])
        data = result_df.loc[:, data_columns].values

        with h5py.File(db_ffp, "w") as db:
            id_ds = db.create_dataset("ids", data=ids.astype(h5py.string_dtype()))
            if "site" in result_df.columns:
                site_ds = db.create_dataset("site", data=result_df.site.values.astype(h5py.string_dtype()))
            data_ds = db.create_dataset("data", data=data)
            data_ds.attrs["columns"] = data_columns.astype(str).astype(h5py.string_dtype())