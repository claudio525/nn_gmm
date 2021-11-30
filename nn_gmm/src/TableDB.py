import multiprocessing as mp
from pathlib import Path
from typing import Sequence, List

import h5py
import numpy as np
import pandas as pd

from .console import console

def _load_column(db_ffp: Path, column: str):
    with h5py.File(db_ffp, "r") as db:
        return db[column][:]


class TableDB:

    def __init__(self, db_ffp: Path):
        self._db_ffp = db_ffp

    @property
    def length(self):
        raise NotImplementedError()

    @property
    def columns(self):
        raise NotImplementedError()

    @staticmethod
    def get_df(db_ffp: Path, index_col: str, columns: List[str] = None):
        data_dict = {}
        with h5py.File(db_ffp, "r") as db:
            columns = list(db.keys()) if columns is None else columns
            for cur_col in columns:
                data_dict[cur_col] = db[cur_col][:]

        return pd.DataFrame.from_dict(data_dict).set_index(index_col).sort_index()

    # @staticmethod
    # def get_df_mp(db_ffp: Path, index_col: str, columns: List[str] = None, n_procs: int = 4):
    #     if columns is None:
    #         with h5py.File(db_ffp, "r") as db:
    #             columns = list(db.keys())
    #
    #     with mp.Pool(processes=n_procs) as p:
    #         data = p.starmap(_load_column, [(db_ffp, cur_col) for cur_col in columns])
    #
    #     return pd.DataFrame(data=np.column_stack(data), columns=columns).set_index(index_col).sort_index()

    @classmethod
    def from_csv(
        cls,
        db_ffp: Path,
        csv_ffps: Sequence[Path],
        use_float32: bool = True,
        use_int32: bool = True,
    ):
        if db_ffp.exists():
            console.print(f"[red]File {db_ffp} already exists, quitting![/]")
            return

        column_types = None
        with h5py.File(db_ffp, "w") as db:
            for ix, cur_csv_ffp in enumerate(csv_ffps):
                console.print(f"Processing {ix+1}/{len(csv_ffps)}")

                if not cur_csv_ffp.exists():
                    console.print(f"[orange]CSV file {cur_csv_ffp} does not exist, skipping[/]")
                    continue
                cur_df = pd.read_csv(cur_csv_ffp)

                if ix == 0:
                    column_types = cur_df.dtypes.to_dict()
                    for cur_col, cur_type in column_types.items():
                        if use_float32 and cur_type is np.dtype(np.float64):
                            column_types[cur_col] = np.float32
                        elif use_int32 and cur_type is np.dtype(np.int64):
                            column_types[cur_col] = np.int32
                        # elif cur_type is np.dtype(object):
                        #     column_types[cur_col] = str

                    for cur_col, cur_type in column_types.items():
                        cur_dset = db.create_dataset(
                            cur_col,
                            data=cur_df[cur_col].values.astype(column_types[cur_col]),
                            dtype=h5py.string_dtype() if cur_type is np.dtype(object) else None,
                            shape=(cur_df.shape[0],),
                            maxshape=(None,),
                            chunks=True,
                        )
                    continue


                # Sanity check
                assert np.all(cur_df.columns == list(column_types.keys()))

                for cur_col in cur_df.columns:
                    cur_dset = db[cur_col]
                    cur_dset.resize((cur_dset.shape[0] + cur_df.shape[0], ))

                    cur_dset[-cur_df.shape[0]:] = cur_df[cur_col].values

        return cls(db_ffp)

