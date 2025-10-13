import time
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
import duckdb

import oq_wrapper as oqw

from . import constants
from . import utils
from .imdb import DuckIMDB

logger = logging.getLogger(__name__)


class EmpiricalDB:

    def __init__(
        self,
        db_ffp: Path,
        readonly: bool = False,
        memory_map_size: int = 2,
        cache_size: int = 1000,
    ):
        self.db_ffp = db_ffp
        self.readonly = readonly
        self._memory_map_size = memory_map_size
        self._cache_size = cache_size

        self._conn = None
        self._cursor = None

        self._gm_params_table_columns = None

    def open(self) -> None:
        """
        Opens a connection to the database.
        """
        if self.readonly:
            self._conn = sqlite3.connect(
                f"file:{self.db_ffp}?mode=ro&immutable=1", uri=True
            )
            # Performance optimizations for read mode
            self._cursor = self._conn.cursor()
            self._cursor.execute("PRAGMA journal_mode = OFF")
            self._cursor.execute("PRAGMA synchronous = OFF")
            self._cursor.execute("PRAGMA automatic_index = OFF")
            self._cursor.execute(f"PRAGMA cache_size = -{self._cache_size * 1024}")
            self._cursor.execute("PRAGMA temp_store = MEMORY")
            self._cursor.execute(
                f"PRAGMA mmap_size = {self._memory_map_size * 1024**3}"
            )
        else:
            self._conn = sqlite3.connect(self.db_ffp)
            # Performance optimizations for write mode
            self._cursor = self._conn.cursor()
            self._cursor.execute("PRAGMA page_size = 32768")
            self._cursor.execute("PRAGMA journal_mode = MEMORY")
            self._cursor.execute("PRAGMA synchronous = OFF")
            self._cursor.execute(f"PRAGMA cache_size = -{self._cache_size * 1024}")
            self._create_tables()

    def close(self, commit: bool = True) -> None:
        """
        Closes the connection to the database.

        Parameters
        ----------
        commit : bool, optional
            Whether to commit changes before closing, by default True.
            Ignored in readonly mode.
        """
        if self._conn:
            if not self.readonly and commit:
                self._cursor.execute("PRAGMA optimize")
                self._conn.commit()
            self._conn.close()
            self._conn = None
            self._cursor = None

    def __enter__(self) -> "EmpiricalDB":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self._conn:
            if exc_type is None:
                self.close(commit=not self.readonly)
            else:
                if not self.readonly:
                    self._conn.rollback()
                self.close(commit=False)

    @property
    def gm_params_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the record_ims table.

        Returns
        -------
        np.ndarray
            Column names of the gm_params table.
        """
        if self._gm_params_table_columns is None:
            self._cursor.execute("PRAGMA table_info(gm_params)")
            self._gm_params_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._gm_params_table_columns

    def _create_tables(self) -> None:
        # Empirical GM params
        im_columns = ", ".join(
            [
                f"{cur_col} REAL"
                for cur_col in constants.DB_GMM_PSA_MEAN_KEYS
                + constants.DB_GMM_PSA_TOTAL_STD_KEYS
            ]
        )
        self._cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS gm_params (
            record_int_id INTEGER PRIMARY KEY,
            event_int_id INTEGER,
            rel_int_id INTEGER,
            site_int_id INTEGER,
            {im_columns})
            """
        )

    def _add_gm_params(self, gm_params_df: pd.DataFrame) -> None:
        """
        Adds GMM parameters to the database.
        """
        gm_params_df = gm_params_df.rename(
            columns=dict(
                zip(constants.GMM_PSA_MEAN_KEYS, constants.DB_GMM_PSA_MEAN_KEYS)
            )
        )
        gm_params_df = gm_params_df.rename(
            columns=dict(
                zip(
                    [f"{cur_key}_std_Total" for cur_key in constants.PSA_KEYS],
                    constants.DB_GMM_PSA_TOTAL_STD_KEYS,
                )
            )
        )
        assert np.all(
            np.isin(
                self.gm_params_table_columns[1:],
                gm_params_df.columns.values.astype(str),
            )
        )

        # Instert the GMM parameters into the database
        # Note: If the record already exists, it will be skipped
        columns = ", ".join(self.gm_params_table_columns)
        placeholders = ", ".join(["?"] * len(self.gm_params_table_columns))
        query = f"""
        INSERT OR IGNORE INTO gm_params ({columns})
        VALUES ({placeholders})
        """
        values = list(
            gm_params_df[self.gm_params_table_columns[1:]].itertuples(
                index=True, name=None
            )
        )
        self._cursor.executemany(query, values)

        logger.debug(f"Inserted GMM parameters for {len(gm_params_df)} records.")

    def get_gm_params_tmp_table(
        self,
        record_int_ids: np.ndarray,
        incl_std: bool = True,
    ):
        """
        Get GMM parameters for the specified records and IMs,
        using a temporary table for the record IDs.

        Parameters
        ----------
        record_int_ids : np.ndarray
            Record int ids to retrieve GMM parameters for.
        incl_std : bool, optional
            Whether to include standard deviation columns, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the GMM parameters for the given records IDs.
        """
        logger.info(f"Getting GMM parameters for {len(record_int_ids)} records.")

        start = time.time()
        self._cursor.execute(
            "CREATE TEMP TABLE IF NOT EXISTS temp_record_ids (record_int_id INTEGER PRIMARY KEY)"
        )
        self._cursor.executemany(
            "INSERT OR IGNORE INTO temp_record_ids VALUES (?)",
            [(int(cur_id),) for cur_id in record_int_ids],
        )
        logger.info(
            f"Took: {time.time() - start:.3f}s to create temp table with {len(record_int_ids)} records."
        )

        columns = (
            constants.DB_GMM_PSA_MEAN_KEYS + constants.DB_GMM_PSA_TOTAL_STD_KEYS
            if incl_std
            else constants.DB_GMM_PSA_MEAN_KEYS
        )

        # Query using a join instead of the IN clause
        start = time.time()
        im_df = pd.read_sql(
            f"""
            SELECT g.record_int_id, {", ".join(columns)}
            FROM gm_params g
            JOIN temp_record_ids t ON g.record_int_id = t.record_int_id
            """,
            self._conn,
            index_col="record_int_id",
            dtype={cur_key: "float32" for cur_key in columns},
        ).rename(columns=constants.DB_PSA_KEYS_TO_PSA)
        logger.info(
            f"Took: {time.time() - start:.3f}s to get IM data for {len(record_int_ids)} records."
        )

        # Clean up
        self._cursor.execute("DROP TABLE IF EXISTS temp_record_ids")

        # Convert column names
        im_df = im_df.rename(
            columns=dict(
                zip(constants.DB_GMM_PSA_MEAN_KEYS, constants.GMM_PSA_MEAN_KEYS)
            )
        ).rename(
            columns=dict(
                zip(
                    constants.DB_GMM_PSA_TOTAL_STD_KEYS,
                    constants.GMM_PSA_TOTAL_STD_KEYS,
                )
            )
        )

        return im_df

    def populate(
        self,
        imdb_ffp: Path,
        gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM],
        batch_size: int = 100_000,
    ):
        logger.info(
            f"Populating empirical database {self.db_ffp} with data from {imdb_ffp}."
        )
        with DuckIMDB(imdb_ffp, readonly=True) as imdb:
            record_info_df = imdb.get_record_info_df()
            event_df = imdb.get_event_df()
            rel_df = imdb.get_rel_df()
            site_df = imdb.get_site_df()
            site_event_df = imdb.get_site_event_df()

        n_records = record_info_df.shape[0]
        n_batches = int(np.ceil((n_records / batch_size)))
        logger.info(
            f"Populating empirical database with {n_records} records, "
            f"in {n_batches} batches of size {batch_size}."
        )
        for i in tqdm(range(n_batches)):
            rupture_df = record_info_df.iloc[
                i * batch_size : (i + 1) * batch_size
            ].copy()
            rupture_df = rupture_df.drop(columns=["event_id", "site_id", "rel_id"])
            rupture_df["site_event_int_id"] = utils.get_site_event_int_id(
                rupture_df["site_int_id"].values, rupture_df["event_int_id"].values
            )
            rupture_df["vs30"] = site_df.loc[rupture_df["site_int_id"], "vs30"].values
            rupture_df["vs30measured"] = False
            rupture_df["z1pt0"] = site_df.loc[rupture_df["site_int_id"], "z1p0"].values
            rupture_df["z2pt5"] = site_df.loc[rupture_df["site_int_id"], "z2p5"].values
            rupture_df["rrup"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rrup"
            ].values
            rupture_df["rjb"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rjb"
            ].values
            rupture_df["rx"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rx"
            ].values
            rupture_df["dip"] = event_df.loc[rupture_df["event_int_id"], "dip"].values
            rupture_df["tect_type"] = event_df.loc[
                rupture_df["event_int_id"], "tect_type"
            ].values
            rupture_df["ztor"] = event_df.loc[rupture_df["event_int_id"], "dtop"].values
            rupture_df["mag"] = rel_df.loc[rupture_df["rel_int_id"], "magnitude"].values
            rupture_df["rake"] = rel_df.loc[rupture_df["rel_int_id"], "rake"].values
            rupture_df["hypo_depth"] = rel_df.loc[
                rupture_df["rel_int_id"], "hypo_depth"
            ].values

            for cur_tect_type in rupture_df["tect_type"].unique():
                cur_oqw_tect_type = oqw.constants.TectType(
                    constants.TECTONIC_TYPE_MAPPING[cur_tect_type]
                )
                cur_gmm = gmm_mapping[cur_oqw_tect_type]

                cur_result = oqw.run_gmm(
                    cur_gmm,
                    cur_oqw_tect_type,
                    rupture_df[rupture_df["tect_type"] == cur_tect_type],
                    "pSA",
                    periods=constants.PSA_PERIODS,
                )

                # Add event_int, site_int_id, rel_int_id to the results
                cur_result["event_int_id"] = record_info_df.loc[
                    cur_result.index, "event_int_id"
                ].values
                cur_result["site_int_id"] = record_info_df.loc[
                    cur_result.index, "site_int_id"
                ].values
                cur_result["rel_int_id"] = record_info_df.loc[
                    cur_result.index, "rel_int_id"
                ].values

                # Add to the database
                self._add_gm_params(cur_result)


class DuckEmpiricalDB:
    """DuckDB-backed empirical database for storing ground motion parameters."""

    def __init__(self, db_ffp: Path, readonly: bool = False):
        self.db_ffp = db_ffp
        self.readonly = readonly
        self._conn = None

        self._gm_params_table_columns = None

    def open(self) -> None:
        """
        Opens a connection to the database.
        """
        if self.readonly:
            self._conn = duckdb.connect(database=self.db_ffp, read_only=True)
            # Set memory limit to 8GB
            self._conn.execute("SET memory_limit='8GB'")
            self._conn.execute("SET enable_progress_bar = false")
        else:
            self._conn = duckdb.connect(database=self.db_ffp)

            # Create tables if they don't exist
            self._create_tables()

    def close(self) -> None:
        """
        Closes the connection to the database.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckEmpiricalDB":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self._conn:
            if exc_type is not None and not self.readonly:
                try:
                    self._conn.rollback()
                except Exception as e:
                    logger.error(f"Error occurred during rollback: {e}")
            self.close()

    @property
    def gm_params_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the record_ims table.

        Returns
        -------
        np.ndarray
            Column names of the gm_params table.
        """
        if self._gm_params_table_columns is None:
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE "
                "table_name = 'gm_params' ORDER BY ordinal_position"
            ).fetchall()
            self._gm_params_table_columns = np.array([row[0] for row in result])
        return self._gm_params_table_columns

    def _create_tables(self) -> None:
        # Empirical GM params
        im_columns = ", ".join(
            [
                f"{cur_col} REAL"
                for cur_col in constants.DB_GMM_PSA_MEAN_KEYS
                + constants.DB_GMM_PSA_TOTAL_STD_KEYS
            ]
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS gm_params (
            record_int_id BIGINT PRIMARY KEY,
            event_int_id INTEGER,
            rel_int_id INTEGER,
            site_int_id INTEGER,
            {im_columns})
            """
        )

    def _add_gm_params(self, gm_params_df: pd.DataFrame) -> None:
        """
        Adds GMM parameters to the database.
        """
        gm_params_df = gm_params_df.rename(
            columns=dict(
                zip(constants.GMM_PSA_MEAN_KEYS, constants.DB_GMM_PSA_MEAN_KEYS)
            )
        )
        gm_params_df = gm_params_df.rename(
            columns=dict(
                zip(
                    [f"{cur_key}_std_Total" for cur_key in constants.PSA_KEYS],
                    constants.DB_GMM_PSA_TOTAL_STD_KEYS,
                )
            )
        )
        assert np.all(
            np.isin(
                self.gm_params_table_columns[1:],
                gm_params_df.columns.values.astype(str),
            )
        )

        gm_params_df["record_int_id"] = gm_params_df.index.astype(np.int64)
        gm_params_df = gm_params_df[self.gm_params_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO gm_params 
            SELECT * FROM gm_params_df
            """
        )

        logger.debug(f"Inserted GMM parameters for {len(gm_params_df)} records.")

    def get_gm_params_tmp_table(
        self,
        record_int_ids: np.ndarray,
        incl_std: bool = True,
    ):
        """
        Get GMM parameters for the specified records and IMs,
        using a temporary table for the record IDs.

        Parameters
        ----------
        record_int_ids : np.ndarray
            Record int ids to retrieve GMM parameters for.
        incl_std : bool, optional
            Whether to include standard deviation columns, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the GMM parameters for the given records IDs.
        """
        logger.info(f"Getting GMM parameters for {len(record_int_ids)} records.")

        record_int_ids_df = pd.DataFrame({"record_int_id": record_int_ids})
        start = time.time()
        self._conn.register("temp_record_ids", record_int_ids_df)
        logger.info(
            f"Took: {time.time() - start:.3f}s to create temp table with {len(record_int_ids)} records."
        )

        columns = (
            constants.DB_GMM_PSA_MEAN_KEYS + constants.DB_GMM_PSA_TOTAL_STD_KEYS
            if incl_std
            else constants.DB_GMM_PSA_MEAN_KEYS
        )

        # Query using a join instead of the IN clause
        start = time.time()
        im_df = self._conn.execute(
            f"""
            SELECT g.record_int_id, {", ".join(columns)}
            FROM gm_params g
            JOIN temp_record_ids t ON g.record_int_id = t.record_int_id
            """,
        ).df().rename(columns=constants.DB_PSA_KEYS_TO_PSA).set_index("record_int_id")
        logger.info(
            f"Took: {time.time() - start:.3f}s to get IM data for {len(record_int_ids)} records."
        )

        # Clean up
        self._conn.unregister("temp_record_ids")

        # Convert column names
        im_df = im_df.rename(
            columns=dict(
                zip(constants.DB_GMM_PSA_MEAN_KEYS, constants.GMM_PSA_MEAN_KEYS)
            )
        ).rename(
            columns=dict(
                zip(
                    constants.DB_GMM_PSA_TOTAL_STD_KEYS,
                    constants.GMM_PSA_TOTAL_STD_KEYS,
                )
            )
        )

        return im_df

    def populate(
        self,
        imdb_ffp: Path,
        gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM],
        batch_size: int = 1_000_000,
    ):
        logger.info(
            f"Populating empirical database {self.db_ffp} with data from {imdb_ffp}."
        )
        with DuckIMDB(imdb_ffp, readonly=True) as imdb:
            record_info_df = imdb.get_record_info_df()
            event_df = imdb.get_event_df()
            rel_df = imdb.get_rel_df()
            site_df = imdb.get_site_df()
            site_event_df = imdb.get_site_event_df()

        n_records = record_info_df.shape[0]
        n_batches = int(np.ceil((n_records / batch_size)))
        logger.info(
            f"Populating empirical database with {n_records} records, "
            f"in {n_batches} batches of size {batch_size}."
        )
        for i in tqdm(range(n_batches)):
            rupture_df = record_info_df.iloc[
                i * batch_size : (i + 1) * batch_size
            ].copy()
            rupture_df = rupture_df.drop(columns=["event_id", "site_id", "rel_id"])
            rupture_df["site_event_int_id"] = utils.get_site_event_int_id(
                rupture_df["site_int_id"].values, rupture_df["event_int_id"].values
            )
            rupture_df["vs30"] = site_df.loc[rupture_df["site_int_id"], "vs30"].values
            rupture_df["vs30measured"] = False
            rupture_df["z1pt0"] = site_df.loc[rupture_df["site_int_id"], "z1p0"].values
            rupture_df["z2pt5"] = site_df.loc[rupture_df["site_int_id"], "z2p5"].values
            rupture_df["rrup"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rrup"
            ].values
            rupture_df["rjb"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rjb"
            ].values
            rupture_df["rx"] = site_event_df.loc[
                rupture_df["site_event_int_id"], "rx"
            ].values
            rupture_df["dip"] = event_df.loc[rupture_df["event_int_id"], "dip"].values
            rupture_df["tect_type"] = event_df.loc[
                rupture_df["event_int_id"], "tect_type"
            ].values
            rupture_df["ztor"] = event_df.loc[rupture_df["event_int_id"], "dtop"].values
            rupture_df["mag"] = rel_df.loc[rupture_df["rel_int_id"], "magnitude"].values
            rupture_df["rake"] = rel_df.loc[rupture_df["rel_int_id"], "rake"].values
            rupture_df["hypo_depth"] = rel_df.loc[
                rupture_df["rel_int_id"], "hypo_depth"
            ].values

            for cur_tect_type in rupture_df["tect_type"].unique():
                cur_oqw_tect_type = oqw.constants.TectType(
                    constants.TECTONIC_TYPE_MAPPING[cur_tect_type]
                )
                cur_gmm = gmm_mapping[cur_oqw_tect_type]

                cur_result = oqw.run_gmm(
                    cur_gmm,
                    cur_oqw_tect_type,
                    rupture_df[rupture_df["tect_type"] == cur_tect_type],
                    "pSA",
                    periods=constants.PSA_PERIODS,
                )

                # Add event_int, site_int_id, rel_int_id to the results
                cur_result["event_int_id"] = record_info_df.loc[
                    cur_result.index, "event_int_id"
                ].values
                cur_result["site_int_id"] = record_info_df.loc[
                    cur_result.index, "site_int_id"
                ].values
                cur_result["rel_int_id"] = record_info_df.loc[
                    cur_result.index, "rel_int_id"
                ].values

                # Add to the database
                self._add_gm_params(cur_result)