from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
import sqlite3

from . import constants
from . import utils

logger = logging.getLogger(__name__)


class IMDB:

    def __init__(
        self,
        db_ffp: Path,
        readonly: bool = False,
        memory_map_size: int = 2,
        cache_size: int = 1000,
    ):
        """
        Initializes the IMDB class.

        Parameters
        ----------
        db_ffp : Path
            File path to the SQLite database.
        readonly : bool, optional
            Whether to open the database in read-only mode, by default False.
        memory_map_size : int, optional
            Size of the memory map in GB, by default 2.
        cache_size : int, optional
            Size of the cache in MB, by default 1000.
        """
        self.db_ffp = db_ffp
        self.readonly = readonly
        self._conn = None
        self._cursor = None

        self._event_table_columns = None
        self._realisation_table_columns = None
        self._site_table_columns = None
        self._site_event_table_columns = None
        self._record_im_table_columns = None

        self._event_to_int_id_mapping = None
        self._event_int_to_id_mapping = None
        self._rel_to_int_id_mapping = None
        self._rel_int_to_id_mapping = None
        self._site_to_int_id_mapping = None
        self._site_int_to_id_mapping = None

        self._n_records = None
        self._memory_map_size = memory_map_size
        self._cache_size = cache_size

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

    def __enter__(self) -> "IMDB":
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
    def event_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the events table.

        Returns
        -------
        np.ndarray
            Column names of the events table.
        """
        if self._event_table_columns is None:
            self._cursor.execute("PRAGMA table_info(events)")
            self._event_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._event_table_columns

    @property
    def realisation_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the realisations table.

        Returns
        -------
        np.ndarray
            Column names of the realisations table.
        """
        if self._realisation_table_columns is None:
            self._cursor.execute("PRAGMA table_info(realisations)")
            self._realisation_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._realisation_table_columns

    @property
    def site_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the sites table.

        Returns
        -------
        np.ndarray
            Column names of the sites table.
        """
        if self._site_table_columns is None:
            self._cursor.execute("PRAGMA table_info(sites)")
            self._site_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._site_table_columns

    @property
    def site_event_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the site_to_event table.

        Returns
        -------
        np.ndarray
            Column names of the site_to_event table.
        """
        if self._site_event_table_columns is None:
            self._cursor.execute("PRAGMA table_info(site_event)")
            self._site_event_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._site_event_table_columns

    @property
    def record_im_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the record_ims table.

        Returns
        -------
        np.ndarray
            Column names of the record_ims table.
        """
        if self._record_im_table_columns is None:
            self._cursor.execute("PRAGMA table_info(record_ims)")
            self._record_im_table_columns = np.array(
                [column_info[1] for column_info in self._cursor.fetchall()]
            )
        return self._record_im_table_columns

    @property
    def event_to_int_id_mapping(self) -> pd.Series:
        """
        Returns a mapping of event_ids to event_int_ids.

        Returns
        -------
        dict
            Mapping of event_ids to event_int_ids.
        """
        if self._event_to_int_id_mapping is None or not self.readonly:
            self._event_to_int_id_mapping = pd.read_sql(
                "SELECT event_id, event_int_id FROM events",
                self._conn,
                index_col="event_id",
            ).squeeze(axis=1)
        return self._event_to_int_id_mapping

    @property
    def event_int_to_id_mapping(self) -> dict:
        """
        Returns a mapping of event_int_ids to event_ids.

        Returns
        -------
        dict
            Mapping of event_int_ids to event_ids.
        """
        if self._event_int_to_id_mapping is None or not self.readonly:
            self._event_int_to_id_mapping = (
                (
                    pd.read_sql(
                        "SELECT event_int_id, event_id FROM events",
                        self._conn,
                        index_col="event_int_id",
                    )
                )
                .squeeze(axis=1)
                .astype("category")
            )
        return self._event_int_to_id_mapping

    @property
    def rel_to_int_id_mapping(self) -> dict:
        """
        Returns a mapping of rel_ids to rel_int_ids.

        Returns
        -------
        dict
            Mapping of rel_ids to rel_int_ids.
        """
        if self._rel_to_int_id_mapping is None or not self.readonly:
            self._rel_to_int_id_mapping = (
                pd.read_sql(
                    "SELECT rel_id, rel_int_id FROM realisations",
                    self._conn,
                    index_col="rel_id",
                )
            ).squeeze(axis=1)
        return self._rel_to_int_id_mapping

    @property
    def rel_int_to_id_mapping(self) -> dict:
        """
        Returns a mapping of rel_int_ids to rel_ids.

        Returns
        -------
        dict
            Mapping of rel_int_ids to rel_ids.
        """
        if self._rel_int_to_id_mapping is None:
            self._rel_int_to_id_mapping = (
                (
                    pd.read_sql(
                        "SELECT rel_int_id, rel_id FROM realisations",
                        self._conn,
                        index_col="rel_int_id",
                    )
                )
                .squeeze(axis=1)
                .astype("category")
            )
        return self._rel_int_to_id_mapping

    @property
    def site_to_int_id_mapping(self) -> dict:
        """
        Returns a mapping of site_ids to site_int_ids.

        Returns
        -------
        dict
            Mapping of site_ids to site_int_ids.
        """
        if self._site_to_int_id_mapping is None:
            self._site_to_int_id_mapping = (
                pd.read_sql(
                    "SELECT site_id, site_int_id FROM sites",
                    self._conn,
                    index_col="site_id",
                )
            ).squeeze(axis=1)
        return self._site_to_int_id_mapping

    @property
    def site_int_to_id_mapping(self) -> dict:
        """
        Returns a mapping of site_int_ids to site_ids.

        Returns
        -------
        dict
            Mapping of site_int_ids to site_ids.
        """
        if self._site_int_to_id_mapping is None:
            self._site_int_to_id_mapping = (
                (
                    pd.read_sql(
                        "SELECT site_int_id, site_id FROM sites",
                        self._conn,
                        index_col="site_int_id",
                    )
                )
                .squeeze(axis=1)
                .astype("category")
            )
        return self._site_int_to_id_mapping

    @property
    def n_records(self) -> int:
        """
        Returns the number of records in the database.

        Returns
        -------
        int
            Number of records in the database.
        """
        if self._n_records is None or not self.readonly:
            self._cursor.execute("SELECT COUNT(*) FROM record_ims")
            self._n_records = self._cursor.fetchone()[0]

        return self._n_records

    def _get_record_int_id(
        self, event_int_id: np.ndarray, rel_int_id: np.ndarray, site_int_id: np.ndarray
    ) -> np.ndarray:
        """
        Generate unique integer IDs for each record
        based on event, realisation, and site IDs.

        Parameters
        ----------
        event_int_id : np.ndarray
            Array of event integer IDs.
        rel_int_id : np.ndarray
            Array of realisation integer IDs.
        site_int_id : np.ndarray
            Array of site integer IDs.

        Returns
        -------
        np.ndarray
            Array of unique integer IDs for each record.
        """
        # Prime multipliers for good distribution
        p1, p2, p3 = 73856093, 19349663, 83492791

        return (event_int_id * p1) ^ (rel_int_id * p2) ^ (site_int_id * p3) % 100000000

    def get_event_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all event data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all event data.
        """
        event_df = pd.read_sql(
            "SELECT * FROM events", self._conn, index_col="event_int_id"
        )
        event_df["tect_type"] = event_df["tect_type"].astype("category")

        return event_df

    def get_site_df(
        self, max_grid_level: int | None = None, min_grid_level: int | None = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing all site data.

        Parameters
        ----------
        max_grid_level : int, optional
            Maximum grid level to filter the sites.
            If None, all sites are returned.
        min_grid_level : int, optional
            Minimum grid level to filter the sites.
            If None, all sites are returned.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all site data.
        """
        site_df = pd.read_sql(
            "SELECT * FROM sites", self._conn, index_col="site_int_id"
        )

        if max_grid_level is not None:
            site_df = site_df[site_df["grid_level"] <= max_grid_level]
        if min_grid_level is not None:
            site_df = site_df[site_df["grid_level"] >= min_grid_level]

        return site_df

    def get_rel_df(self, events: np.ndarray | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame containing all realisation data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all realisation data.
        """
        logger.info("Getting realisation data from the database.")
        rel_df = pd.read_sql(
            "SELECT * FROM realisations", self._conn, index_col="rel_int_id"
        )
        rel_df["event_id"] = self.event_int_to_id_mapping.loc[
            rel_df.event_int_id.values
        ].values

        if events is not None:
            rel_df = rel_df[rel_df["event_id"].isin(events)]

        return rel_df

    def get_site_event_df(
        self,
        events: np.ndarray | None = None,
        sites: np.ndarray | None = None,
        site_event_int_ids: np.ndarray | None = None,
        max_rrup: float | None = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing all site-event data.

        Parameters
        ----------
        sites : np.ndarray, optional
            Array of site IDs to filter the site-event data.
            If None, all site-event data is returned.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all site-event data.
        """
        logger.info("Getting site-event data from the database.")

        if events is not None and sites is not None:
            raise NotImplementedError()
        elif site_event_int_ids is not None:
            if site_event_int_ids.size > 1000:
                logger.info("Using tmp table for site-event data retrieval.")
                start = time.time()
                self._cursor.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS temp_site_event_ids (site_event_int_id INTEGER PRIMARY KEY)"
                )
                self._cursor.executemany(
                    "INSERT OR IGNORE INTO temp_site_event_ids VALUES (?)",
                    [(int(site_event_int_id),) for site_event_int_id in site_event_int_ids]
                )
                logger.info(
                    f"Took: {time.time() - start:.3f} to create temp table with {len(site_event_int_ids)} site-event IDs."
                )

                # Query using a join instead of IN clause
                start = time.time()
                site_event_df = pd.read_sql(
                    """
                    SELECT se.*
                    FROM site_event se
                    INNER JOIN temp_site_event_ids tei
                    ON tei.site_event_int_id = se.site_event_int_id
                    """,
                    self._conn,
                    index_col="site_event_int_id",
                )
                logger.info(
                    f"Took: {time.time() - start:.3f} to get site-event data for {len(site_event_int_ids)} site-event IDs."
                )

                # Clean up
                self._cursor.execute("DROP TABLE IF EXISTS temp_site_event_ids")
            else:
                start = time.time()
                site_event_df = pd.read_sql(
                    f"""
                    SELECT * FROM site_event WHERE site_event_int_id IN ({", ".join("?" * len(site_event_int_ids))})
                    """,
                    self._conn,
                    params=site_event_int_ids.tolist(),
                    index_col="site_event_int_id",
                )
                logger.info(
                    f"Took: {time.time() - start:.3f} to get site-event data for {len(site_event_int_ids)} site-event IDs."
                )
        elif events is not None:
            start = time.time()
            event_int_ids = self.event_to_int_id_mapping.loc[events].values.tolist()
            site_event_df = pd.read_sql(
                f"""
                SELECT se.*
                FROM site_event se
                INNER JOIN (SELECT event_int_id FROM events WHERE event_int_id IN ({", ".join("?" * len(event_int_ids))})) ef
                ON ef.event_int_id = se.event_int_id
                """,
                self._conn,
                params=event_int_ids.tolist(),
                index_col="site_event_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f} to get site-event data for {len(event_int_ids)} events."
            )
        elif sites is not None:
            start = time.time()
            site_int_ids = self.site_to_int_id_mapping.loc[sites].values.tolist()
            site_event_df = pd.read_sql(
                f"""
                SELECT se.*
                FROM site_event se
                INNER JOIN (SELECT site_int_id FROM sites WHERE site_int_id IN ({", ".join("?" * len(site_int_ids))})) sf
                ON sf.site_int_id = se.site_int_id
                {"WHERE se.rrup <= {}".format(max_rrup) if max_rrup is not None else ""}
                """,
                self._conn,
                params=site_int_ids,
                index_col="site_event_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f} to get site-event data for {len(site_int_ids)} sites with max_rrup = {max_rrup}."
            )
        else:
            logger.warning(
                "No filters applied, returning all site-event data. This is slow."
            )
            start = time.time()
            site_event_df = pd.read_sql(
                "SELECT * FROM site_event", self._conn, index_col="site_event_int_id"
            )
            logger.info(f"Took: {time.time() - start:.3f} to get all site-event data.")

        site_event_df["site_id"] = self.site_int_to_id_mapping.loc[
            site_event_df.site_int_id.values
        ].values
        site_event_df["event_id"] = self.event_int_to_id_mapping.loc[
            site_event_df.event_int_id.values
        ].values

        return site_event_df

    def get_record_info_df(
        self, events: np.ndarray | None = None, sites: np.ndarray | None = None, record_int_ids: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing
        the record_id, event_id, rel_id, and site_id.
        """
        logger.info("Getting record info from the database.")

        if record_int_ids is not None:
            if record_int_ids.size > 1000:
                logger.info("Using tmp table for record info retrieval.")
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

                # Query using a join instead of IN clause
                start = time.time()
                record_info_df = pd.read_sql(
                    """
                    SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id
                    FROM record_ims r
                    JOIN temp_record_ids t ON r.record_int_id = t.record_int_id
                    """,
                    self._conn,
                    index_col="record_int_id",
                )
                logger.info(
                    f"Took: {time.time() - start:.3f}s to get record info for {len(record_int_ids)} records."
                )

                # Clean up
                self._cursor.execute("DROP TABLE IF EXISTS temp_record_ids")
            else:
                start = time.time()
                record_info_df = pd.read_sql(
                    f"""
                    SELECT record_int_id, event_int_id, site_int_id, rel_int_id
                    FROM record_ims WHERE record_int_id IN ({", ".join("?" * len(record_int_ids))})
                    """,
                    self._conn,
                    params=record_int_ids.tolist(),
                    index_col="record_int_id",
                )
                logger.info(
                    f"Took: {time.time() - start:.3f}s to get record info for {len(record_int_ids)} records."
                )
        elif events is not None and sites is not None:
            site_int_ids = self.site_to_int_id_mapping.loc[sites].values.tolist()
            event_int_ids = self.event_to_int_id_mapping.loc[events].values.tolist()
            start = time.time()
            record_info_df = pd.read_sql(
                f"""
                SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id
                FROM record_ims r
                INNER JOIN (SELECT site_int_id FROM sites WHERE site_int_id IN ({", ".join("?" * len(site_int_ids))})) sf
                ON sf.site_int_id = r.site_int_id
                INNER JOIN (SELECT event_int_id FROM events WHERE event_int_id IN ({", ".join("?" * len(event_int_ids))})) ef
                ON ef.event_int_id = r.event_int_id
                """,
                self._conn,
                params=site_int_ids + event_int_ids,
                index_col="record_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for {len(site_int_ids)} sites and {len(event_int_ids)} events."
            )
        elif sites is not None:
            site_int_ids = self.site_to_int_id_mapping.loc[sites].values.tolist()
            start = time.time()
            record_info_df = pd.read_sql(
                f"""
                SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id
                FROM record_ims r
                INNER JOIN (SELECT site_int_id FROM sites WHERE site_int_id IN ({", ".join("?" * len(site_int_ids))})) sf
                ON sf.site_int_id = r.site_int_id
                """,
                self._conn,
                params=site_int_ids,
                index_col="record_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for {len(site_int_ids)} sites."
            )
        elif events is not None:
            raise NotImplementedError()
        else:
            # If no filters are applied, return all records
            logger.warning("No filters applied, returning all records. This is slow!")
            start = time.time()
            record_info_df = pd.read_sql(
                "SELECT record_int_id, event_int_id, site_int_id, rel_int_id FROM record_ims",
                self._conn,
                index_col="record_int_id",
            )
            logger.info(f"Took: {time.time() - start:.3f}s to get all record info.")

        # Map the integer IDs back to their original values
        record_info_df["event_id"] = self.event_int_to_id_mapping.loc[
            record_info_df.event_int_id.values
        ].values
        record_info_df["site_id"] = self.site_int_to_id_mapping.loc[
            record_info_df.site_int_id.values
        ].values
        record_info_df["rel_id"] = self.rel_int_to_id_mapping.loc[
            record_info_df.rel_int_id.values
        ].values

        return record_info_df

    def get_im_data_tmp_table(
        self, ims: np.ndarray, record_int_ids: np.ndarray
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing the IM data for the given record IDs.
        This method uses a temporary table to speed up the query.

        Parameters
        ----------
        ims: np.ndarray
            IMs to retrieve.
        record_int_ids: np.ndarray
            Record int ids to retrieve IM data for.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the IM data for the given record IDs.
        """
        logger.info("Getting IM data from the database.")

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

        # Query using a join instead of IN clause
        start = time.time()
        im_df = pd.read_sql(
            f"""
            SELECT r.record_int_id, {", ".join(constants.PSA_KEYS_TO_DB_SERIES.loc[ims].values.astype(str).tolist())}
            FROM record_ims r
            JOIN temp_record_ids t ON r.record_int_id = t.record_int_id
            """,
            self._conn,
            index_col="record_int_id",
        ).rename(columns=constants.DB_PSA_KEYS_TO_PSA)
        logger.info(
            f"Took: {time.time() - start:.3f}s to get IM data for {len(record_int_ids)} records."
        )

        # Clean up
        self._cursor.execute("DROP TABLE IF EXISTS temp_record_ids")
        return im_df

    def get_im_data(
        self,
        ims: np.ndarray,
        record_int_ids: np.ndarray = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing the IM data for the given record IDs.

        Parameters
        ----------
        ims : np.ndarray
            IMs to retrieve.
        record_ids : np.ndarray
            Record IDs to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the IM data for the given record IDs.
        """
        logger.debug("Getting IM data from the database.")

        if record_int_ids is not None:
            start = time.time()
            im_df = pd.read_sql(
                f"""
                SELECT record_int_id, {", ".join(constants.PSA_KEYS_TO_DB_SERIES.loc[ims].values.astype(str).tolist())}
                FROM record_ims
                WHERE record_int_id IN ({", ".join("?" * len(record_int_ids))})
                """,
                self._conn,
                params=record_int_ids.tolist(),
                index_col="record_int_id",
            ).rename(columns=constants.DB_PSA_KEYS_TO_PSA)
            logger.debug(
                f"Took: {time.time() - start:.3f}s to get IM data for {len(record_int_ids)} records."
            )
        elif limit is not None and offset is not None:
            start = time.time()
            im_df = pd.read_sql(
                f"""
                SELECT record_int_id, {", ".join(constants.PSA_KEYS_TO_DB_SERIES.loc[ims].values.astype(str).tolist())}
                FROM record_ims
                LIMIT {limit} OFFSET {offset}
                """,
                self._conn,
                index_col="record_int_id",
            ).rename(columns=constants.DB_PSA_KEYS_TO_PSA)
            logger.debug(
                f"Took: {time.time() - start:.3f}s to get IM data with limit={limit} and offset={offset}."
            )
        else:
            raise ValueError(
                "Either record_int_ids or both limit and offset must be provided."
            )

        return im_df

    def add_event_data(self, event_id: str, median_info: pd.Series) -> None:
        """
        Add event data to the 'events' table.

        Parameters
        ----------
        event_id : str
            Unique identifier for the event.
        median_info : pd.Series
            Series containing event metadata.
        """
        assert np.all(np.isin(self.event_table_columns[2:], median_info.index.values))

        # Check if event_id already exists in the database
        self._cursor.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,))
        if self._cursor.fetchone() is not None:
            logger.warning(f"Event ID {event_id} already exists. Skipping insert.")
            return

        # Insert the event data into the database
        columns = ", ".join(self.event_table_columns[1:])
        placeholders = ", ".join(["?"] * len(self.event_table_columns[1:]))
        query = f"""
        INSERT INTO events (
            {columns}
        ) VALUES ({placeholders})"""
        values = [event_id] + [
            (
                int(median_info[col])
                if col in ["sim_type", "plane_count"]
                else median_info[col]
            )
            for col in self.event_table_columns[2:]
        ]
        self._cursor.execute(query, values)
        logger.debug(f"Inserted event data for event {event_id}.")

    def add_realisation_data(self, rel_df: pd.DataFrame) -> None:
        """
        Add realisation data to the 'realisations' table.
        Note: All realisation data must belong to the same event.

        Parameters
        ----------
        rel_df : pd.DataFrame
            DataFrame containing realisation data. All rows
            must belong to the same event.
        """
        assert np.all(np.isin(self.realisation_table_columns[3:], rel_df.columns))
        assert np.all(rel_df.event_id == rel_df.event_id.iloc[0])
        rel_df = rel_df.sort_index()

        # Get event_int_id for the realisations
        rel_df["event_int_id"] = self.event_to_int_id_mapping.loc[
            rel_df.event_id
        ].values

        # Check if realisation for the event already exists in the database
        self._cursor.execute(
            "SELECT 1 FROM realisations WHERE event_int_id = ?",
            (rel_df.event_int_id.iloc[0],),
        )
        if self._cursor.fetchone() is not None:
            logger.warning(
                f"Realisations data for event {rel_df.event_id.iloc[0]}"
                f"already exists. Skipping insert."
            )
            return

        # Insert the realisation data into the database
        columns = ", ".join(self.realisation_table_columns[1:])
        placeholders = ", ".join(["?"] * len(self.realisation_table_columns[1:]))
        query = f"""
        INSERT INTO realisations (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(
            rel_df[self.realisation_table_columns[2:]].itertuples(index=True, name=None)
        )
        self._cursor.executemany(query, values)

        logger.debug(f"Inserted realisation data for event {rel_df.event_id.iloc[0]}")

    def add_site_data(self, site_df: pd.DataFrame) -> None:
        """
        Add site data to the 'sites' table.

        Parameters
        ----------
        site_df : pd.DataFrame
            DataFrame containing the site data to add
        """
        assert np.all(np.isin(self.site_table_columns[2:], site_df.columns))
        site_df = site_df[self.site_table_columns[2:]]
        site_df = site_df.sort_index()

        # Insert the site data into the database
        # Note: If the site already exists, it will be skipped
        columns = ", ".join(self.site_table_columns[1:])
        placeholders = ", ".join(["?"] * len(self.site_table_columns[1:]))
        query = f"""
        INSERT OR IGNORE INTO sites (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(site_df.itertuples(index=True, name=None))
        self._cursor.executemany(query, values)
        logger.debug(f"Inserted site data for {len(site_df)} sites.")

    def add_site_event_data(self, site_event_df: pd.DataFrame) -> None:
        """
        Add site-event data to the 'site_event' table.

        Parameters
        ----------
        site_event_df : pd.DataFrame
            DataFrame containing the site-event data to add
        """
        assert np.all(np.isin(self.site_event_table_columns[3:], site_event_df.columns))

        site_event_df["site_int_id"] = self.site_to_int_id_mapping.loc[
            site_event_df.site_id
        ].values.astype(int)
        site_event_df["event_int_id"] = self.event_to_int_id_mapping.loc[
            site_event_df.event_id
        ].values.astype(int)
        site_event_df["site_event_int_id"] = utils.get_site_event_int_id(
            site_event_df.site_int_id.values, site_event_df.event_int_id.values
        )

        # Insert the site-event data into the database
        # Note: If the site-event pair already exists, it will be skipped
        columns = ", ".join(self.site_event_table_columns)
        placeholders = ", ".join(["?"] * len(self.site_event_table_columns))
        query = f"""
        INSERT OR IGNORE INTO site_event (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(
            site_event_df[self.site_event_table_columns].itertuples(
                index=False, name=None
            )
        )
        self._cursor.executemany(query, values)
        logger.debug(
            f"Inserted site-event data for {len(site_event_df)} site-event pairs."
        )

    def add_record_im_data(self, im_df: pd.DataFrame) -> None:
        """
        Add IM data to the 'record_ims' table.

        Parameters
        ----------
        im_df : pd.DataFrame
            DataFrame containing the IM data to add
        """
        im_df = im_df.rename(
            columns=dict(zip(constants.PSA_KEYS, constants.PSA_KEYS_TO_DB))
        )
        assert np.all(np.isin(self.record_im_table_columns[4:], im_df.columns))

        im_df["event_int_id"] = self.event_to_int_id_mapping.loc[im_df.event_id].values
        im_df["site_int_id"] = self.site_to_int_id_mapping.loc[im_df.site_id].values
        im_df["rel_int_id"] = self.rel_to_int_id_mapping.loc[im_df.rel_id].values
        im_df["record_int_id"] = self._get_record_int_id(
            im_df.event_int_id.values, im_df.rel_int_id.values, im_df.site_int_id.values
        )

        # Insert the IM data into the database
        # Note: If the record already exists, it will be skipped
        columns = ", ".join(self.record_im_table_columns)
        placeholders = ", ".join(["?"] * len(self.record_im_table_columns))
        query = f"""
        INSERT OR IGNORE INTO record_ims (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(
            im_df[self.record_im_table_columns].itertuples(index=False, name=None)
        )
        self._cursor.executemany(query, values)

        logger.debug(f"Inserted IM data for {len(im_df)} records.")

    def _create_tables(self) -> None:
        """
        Creates tables in the database if they don't exist.
        Only called in write mode.
        """
        if self.readonly:
            logger.warning("Attempting to create tables in readonly mode, ignoring.")
            return

        # Event table
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
            event_int_id INTEGER PRIMARY KEY,
            event_id TEXT UNIQUE,
            magnitude REAL,
            sim_type INTEGER,
            fault_type REAL,
            tect_type REAL,
            dip REAL,
            dtop REAL,
            dbottom REAL,
            length REAL,
            plane_count INTEGER,
            dip_dir REAL)"""
        )
        # Create index for fast access using event_id
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_events_event_id 
               ON events (event_id)"""
        )

        # Realisation table
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realisations (
            rel_int_id INTEGER PRIMARY KEY,
            rel_id TEXT UNIQUE,
            event_int_id INT,
            magnitude REAL,
            rake REAL,
            shypo REAL,
            dhypo REAL,
            hypo_lat REAL,
            hypo_lon REAL,
            hypo_depth REAL,
            FOREIGN KEY (event_int_id) REFERENCES events (event_int_id))"""
        )
        # Create index for fast access using rel_id
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_realisations_rel_id 
               ON realisations (rel_id)"""
        )

        # Site table
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sites (
            site_int_id INTEGER PRIMARY KEY,
            site_id TEXT UNIQUE,
            lat REAL,
            lon REAL,
            vs30 REAL,
            z1p0 REAL,
            z2p5 REAL,
            grid_level INTEGER)"""
        )
        # Create index for fast access using site_id
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_sites_site_id 
               ON sites (site_id)"""
        )

        # Site-Event table
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS site_event (
            site_event_int_id INTEGER PRIMARY KEY,
            site_int_id INTEGER,
            event_int_id INTEGER,
            rrup REAL,
            rjb REAL,
            rx REAL,
            ry REAL,
            FOREIGN KEY (site_int_id) REFERENCES sites (site_int_id),
            FOREIGN KEY (event_int_id) REFERENCES events (event_int_id)
            )
        """
        )
        # Create index for fast access using site_int_id
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_site_event_site_id 
                ON site_event (site_int_id)"""
        )
        # Create index for fast access using event_int_id
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_site_event_event_id 
                ON site_event (event_int_id)"""
        )

        # Simulation records
        im_columns = ", ".join(
            [
                f"{cur_col} REAL"
                for cur_col in constants.DB_PSA_KEYS + constants.NON_PSA_IMS
            ]
        )
        self._cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS record_ims (
            record_int_id INTEGER PRIMARY KEY,
            event_int_id INTEGER,
            rel_int_id INTEGER,
            site_int_id INTEGER,
            {im_columns},
            FOREIGN KEY (event_int_id) REFERENCES events (event_int_id),
            FOREIGN KEY (rel_int_id) REFERENCES realisations (rel_int_id),
            FOREIGN KEY (site_int_id) REFERENCES sites (site_int_id)
            )
        """
        )

        # Create indexes for faster querying
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_event_id 
               ON record_ims (event_int_id)"""
        )
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_rel_id 
               ON record_ims (rel_int_id)"""
        )
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_site_id 
               ON record_ims (site_int_id)"""
        )
        self._cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_site_event_rel
                ON record_ims (site_int_id, event_int_id, rel_int_id)"""
        )
