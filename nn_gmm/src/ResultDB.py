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
            ids = db["ids"][:].astype(str)

            df = pd.DataFrame(data=data, index=ids, columns=self.columns[column_ind])

            if "fault" in columns:
                fault_names = db["fault_names"][:].astype(str)
                fault_ids = db["fault_ids"][:]
                df["fault"] = fault_names[fault_ids]

            if "rupture" in columns:
                rupture_names = db["rupture_names"][:].astype(str)
                rupture_ids = db["rupture_ids"][:]
                df["rupture"] = rupture_names[rupture_ids]

            if "site" in columns:
                site_names = db["site_names"][:].astype(str)
                site_ids = db["site_ids"][:]
                df["site"] = site_names[site_ids]

        return df.sort_index()

    @classmethod
    def get_data_static(cls, db_ffp: Path, columns: Sequence[str]):
        """Wrapper class for retrieving data without
        having to create an instance (in the Caller)"""
        result_db = cls(db_ffp)
        return result_db.get_data(columns)


    @staticmethod
    def write_data(result_df: pd.DataFrame, db_ffp: Path):
        ids = result_df.index.values.astype(str)
        data_columns = np.asarray([col for col in result_df.columns.values if col not in ["site", "fault", "rupture"]])
        data = result_df.loc[:, data_columns].values

        with h5py.File(db_ffp, "w") as db:
            id_ds = db.create_dataset("ids", data=ids.astype(h5py.string_dtype()))

            if "site" in result_df.columns:
                unique_sites, site_ids = np.unique(result_df["site"], return_inverse=True)
                site_ids_ds = db.create_dataset("site_ids", data=site_ids, dtype=np.int32)
                site_names_ds = db.create_dataset("site_names", data=unique_sites.astype(h5py.string_dtype()))

            if "fault" in result_df.columns:
                unique_faults, fault_ids = np.unique(result_df["fault"], return_inverse=True)
                fault_ids_ds = db.create_dataset("fault_ids", data=fault_ids, dtype=np.int32)
                fault_names_ds = db.create_dataset("fault_names", data=unique_faults.astype(h5py.string_dtype()))

            if "rupture" in result_df.columns:
                unique_ruptures, rupture_ids = np.unique(result_df["rupture"], return_inverse=True)
                rupture_ids_ds = db.create_dataset("rupture_ids", data=rupture_ids, dtype=np.int32)
                rupture_names_ds = db.create_dataset("rupture_names", data=unique_ruptures.astype(h5py.string_dtype()))

            data_ds = db.create_dataset("data", data=data)
            data_ds.attrs["columns"] = data_columns.astype(str).astype(h5py.string_dtype())

