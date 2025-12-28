import logging
import time

import duckdb
import numpy as np
import pandas as pd

from qcore import coordinates as coords

from . import constants
from . import utils

logger = logging.getLogger(__name__)


class DuckIMDB:
    """
    IMDB implementation using DuckDB.
    """

    def __init__(self, db_ffp: str, readonly: bool = False) -> None:
        self.db_ffp = db_ffp
        self.readonly = readonly
        self._conn = None

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

    def __enter__(self) -> "DuckIMDB":
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
    def event_table_columns(self) -> np.ndarray:
        """
        Returns the column names of the events table.

        Returns
        -------
        np.ndarray
            Column names of the events table.
        """
        if self._event_table_columns is None:
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE "
                "table_name = 'events' ORDER BY ordinal_position"
            ).fetchall()
            self._event_table_columns = np.array([row[0] for row in result])
        return self._event_table_columns
    
    @property
    def max_event_int_id(self) -> int:
        """
        Returns the maximum event_int_id in the events table.

        Returns
        -------
        int
            Maximum event_int_id.
        """
        result = self._conn.execute(
            "SELECT MAX(event_int_id) FROM events"
        ).fetchone()
        return result[0] if result[0] is not None else None

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
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'realisations' ORDER BY ordinal_position"
            ).fetchall()
            self._realisation_table_columns = np.array([row[0] for row in result])
        return self._realisation_table_columns
    
    @property
    def max_rel_int_id(self) -> int:
        """
        Returns the maximum rel_int_id in the realisations table.

        Returns
        -------
        int
            Maximum rel_int_id.
        """
        result = self._conn.execute(
            "SELECT MAX(rel_int_id) FROM realisations"
        ).fetchone()
        return result[0] if result[0] is not None else None

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
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'sites' ORDER BY ordinal_position"
            ).fetchall()
            self._site_table_columns = np.array([row[0] for row in result])
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
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'site_event' ORDER BY ordinal_position"
            ).fetchall()
            self._site_event_table_columns = np.array([row[0] for row in result])
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
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'record_ims' ORDER BY ordinal_position"
            ).fetchall()
            self._record_im_table_columns = np.array([row[0] for row in result])

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
            self._event_to_int_id_mapping = (
                self._conn.execute("SELECT event_id, event_int_id FROM events")
                .df()
                .set_index("event_id")
                .squeeze(axis=1)
            )
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
                self._conn.execute("SELECT event_int_id, event_id FROM events")
                .df()
                .set_index("event_int_id")
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
                self._conn.execute("SELECT rel_id, rel_int_id FROM realisations")
                .df()
                .set_index("rel_id")
                .squeeze(axis=1)
            )
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
        if self._rel_int_to_id_mapping is None or not self.readonly:
            self._rel_int_to_id_mapping = (
                self._conn.execute("SELECT rel_int_id, rel_id FROM realisations")
                .df()
                .set_index("rel_int_id")
                .squeeze()
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
        if self._site_to_int_id_mapping is None or not self.readonly:
            self._site_to_int_id_mapping = (
                self._conn.execute("SELECT site_id, site_int_id FROM sites")
                .df()
                .set_index("site_id")
                .squeeze()
            )

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
        if self._site_int_to_id_mapping is None or not self.readonly:
            self._site_int_to_id_mapping = (
                self._conn.execute("SELECT site_int_id, site_id FROM sites")
                .df()
                .set_index("site_int_id")
                .squeeze(axis=1)
                .astype("category")
            )
        return self._site_int_to_id_mapping

    def get_event_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all event data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all event data.
        """
        event_df = (
            self._conn.execute("SELECT * FROM events").df().set_index("event_int_id")
        )

        return event_df

    def get_site_df(
        self,
        max_grid_level: int | None = None,
        min_grid_level: int | None = None,
        add_nztm: bool = False,
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
        logger.info("Getting site data from the database.")
        site_df = (
            self._conn.execute("SELECT * FROM sites").df().set_index("site_int_id")
        )

        if max_grid_level is not None:
            site_df = site_df[site_df["grid_level"] <= max_grid_level]
        if min_grid_level is not None:
            site_df = site_df[site_df["grid_level"] >= min_grid_level]

        if add_nztm:
            nztm_coords = coords.wgs_depth_to_nztm(site_df[["lat", "lon"]].values)
            site_df.loc[:, "nztm_y"], site_df.loc[:, "nztm_x"] = (
                nztm_coords[:, 0],
                nztm_coords[:, 1],
            )

        return site_df

    def get_rel_df(
        self, events: np.ndarray | None = None, rel_int_ids: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing all realisation data.

        Parameters
        ----------
        events : np.ndarray, optional
            Array of event IDs to filter the realisation data.
            If None, all realisation data is returned.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all realisation data.
        """
        logger.info("Getting realisation data from the database.")
        rel_df = (
            self._conn.execute("SELECT * FROM realisations")
            .df()
            .set_index("rel_int_id")
        )

        if events is not None:
            rel_df["event_id"] = self.event_int_to_id_mapping.loc[
                rel_df.event_int_id.values
            ].values
            rel_df = rel_df[rel_df["event_id"].isin(events)]
        if rel_int_ids is not None:
            rel_df = rel_df[rel_df.index.isin(rel_int_ids)]

        return rel_df

    def get_site_event_df(
        self,
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

        if site_event_int_ids is not None:
            site_event_int_ids = np.unique(site_event_int_ids)

            start = time.time()
            # DuckDB handles large datasets efficiently with temporary views
            site_event_ids_df = pd.DataFrame({"site_event_int_id": site_event_int_ids})

            # Register the DataFrame as a temporary view in DuckDB
            self._conn.register("temp_site_event_ids", site_event_ids_df)

            logger.info(
                f"Took: {time.time() - start:.3f} to register "
                f"temp view with {len(site_event_int_ids)} site-event IDs."
            )

            # Query using a join with the temporary view
            start = time.time()
            site_event_df = (
                self._conn.execute(
                    """
                SELECT se.*
                FROM site_event se
                INNER JOIN temp_site_event_ids t
                ON t.site_event_int_id = se.site_event_int_id
                """
                )
                .df()
                .set_index("site_event_int_id")
            )

            logger.info(
                f"Took: {time.time() - start:.3f} to get site-event"
                f" data for {len(site_event_int_ids)} site-event IDs."
            )

            self._conn.unregister("temp_site_event_ids")
        elif sites is not None:
            site_int_ids_df = self.site_to_int_id_mapping.loc[sites].to_frame()

            start = time.time()
            self._conn.register("temp_site_ids", site_int_ids_df)

            site_event_df = (
                self._conn.execute(
                    f"""
                SELECT se.*
                FROM site_event se
                INNER JOIN temp_site_ids sf
                ON sf.site_int_id = se.site_int_id
                {"WHERE se.rrup <= {}".format(max_rrup) if max_rrup is not None else ""}
                """
                )
                .df()
                .set_index("site_event_int_id")
            )
            logger.info(
                f"Took: {time.time() - start:.3f} to get site-event "
                f"data for {len(sites)} sites with max_rrup = {max_rrup}."
            )

            self._conn.unregister("temp_site_ids")
        else:
            logger.warning(
                "No filters applied, returning all site-event data. This is slow."
            )
            start = time.time()
            site_event_df = (
                self._conn.execute("SELECT * FROM site_event")
                .df()
                .set_index("site_event_int_id")
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
        self,
        events: np.ndarray | None = None,
        sites: np.ndarray | None = None,
        record_int_ids: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing
        the record_id, event_id, rel_id, and site_id.
        """
        logger.info("Getting record info from the database.")

        if record_int_ids is not None:
            logger.info("Using tmp table for record info retrieval.")
            start = time.time()
            record_int_ids_df = pd.DataFrame({"record_int_id": record_int_ids})
            self._conn.register("temp_record_ids", record_int_ids_df)
            logger.info(
                f"Took: {time.time() - start:.3f}s to create temp table with {len(record_int_ids)} records."
            )

            start = time.time()
            record_info_df = (
                self._conn.execute(
                    """
                SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id
                FROM record_ims r
                JOIN temp_record_ids t ON r.record_int_id = t.record_int_id
                """
                )
                .df()
                .set_index("record_int_id")
            )
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for {len(record_int_ids)} records."
            )

            self._conn.unregister("temp_record_ids")
        elif events is not None or sites is not None:
            query = "SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id FROM record_ims r "

            if sites is not None:
                site_int_ids_df = (
                    self.site_to_int_id_mapping.loc[sites]
                    .to_frame()
                    .reset_index(drop=True)
                )
                self._conn.register("temp_site_ids", site_int_ids_df)
                query += (
                    "INNER JOIN temp_site_ids ts ON ts.site_int_id = r.site_int_id "
                )
            if events is not None:
                event_int_ids_df = (
                    self.event_to_int_id_mapping.loc[events]
                    .to_frame()
                    .reset_index(drop=True)
                )
                self._conn.register("temp_event_ids", event_int_ids_df)
                query += (
                    "INNER JOIN temp_event_ids te ON te.event_int_id = r.event_int_id "
                )

            start = time.time()
            record_info_df = self._conn.execute(query).df().set_index("record_int_id")
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for "
                f"{len(sites) if sites is not None else 'all'} sites and "
                f"{len(events) if events is not None else 'all'} events."
            )
        else:
            # If no filters are applied, return all records
            logger.warning("No filters applied, returning all records. This is slow!")
            start = time.time()
            record_info_df = (
                self._conn.execute(
                    "SELECT record_int_id, event_int_id, site_int_id, rel_int_id FROM record_ims"
                )
                .df()
                .set_index("record_int_id")
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

    def get_im_data(self, ims: np.ndarray, record_int_ids: np.ndarray) -> pd.DataFrame:
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

        record_int_ids_df = pd.DataFrame({"record_int_id": record_int_ids})
        start = time.time()
        self._conn.register("temp_record_ids", record_int_ids_df)
        logger.info(
            f"Took: {time.time() - start:.3f}s to create temp table with {len(record_int_ids)} records."
        )

        start = time.time()
        im_df = (
            self._conn.execute(
                f"""
            SELECT r.record_int_id, {", ".join(constants.IMS_TO_DB_IMS_SERIES.loc[ims].values.astype(str).tolist())}
            FROM record_ims r
            JOIN temp_record_ids t ON r.record_int_id = t.record_int_id
            """
            )
            .df()
            .set_index("record_int_id")
            .rename(columns=constants.DB_PSA_KEYS_TO_PSA)
        )
        logger.info(
            f"Took: {time.time() - start:.3f}s to get IM data for {len(record_int_ids)} records."
        )

        self._conn.unregister("temp_record_ids")
        return im_df

    def add_site_data(self, site_df: pd.DataFrame) -> None:
        """
        Add site data to the 'sites' table.

        Parameters
        ----------
        site_df : pd.DataFrame
            DataFrame containing the site data to add
        """
        site_df = site_df.sort_index().reset_index()
        site_df["site_int_id"] = np.arange(1, len(site_df) + 1, dtype=int)
        site_df = site_df[self.site_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO sites 
            SELECT * FROM site_df
            """
        )
        logger.info(f"Inserted site data for {len(site_df)} sites.")

    def add_site_event_data(self, site_event_df: pd.DataFrame) -> None:
        """
        Add site-event data to the 'site_event' table.

        Parameters
        ----------
        site_event_df : pd.DataFrame
            DataFrame containing the site-event data to add
        """
        site_event_df["site_int_id"] = self.site_to_int_id_mapping.loc[
            site_event_df.site_id
        ].values.astype(int)
        site_event_df["event_int_id"] = self.event_to_int_id_mapping.loc[
            site_event_df.event_id
        ].values.astype(int)
        site_event_df["site_event_int_id"] = utils.get_site_event_int_id(
            site_event_df.site_int_id.values, site_event_df.event_int_id.values
        )

        site_event_df = site_event_df[self.site_event_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO site_event
            SELECT * FROM site_event_df
            """
        )

        logger.info(
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
        im_df = im_df.rename(columns=constants.PSA_KEYS_TO_DB)
        if "event_int_id" not in im_df.columns:
            im_df["event_int_id"] = self.event_to_int_id_mapping.loc[im_df.event_id].values
        if "site_int_id" not in im_df.columns:
            im_df["site_int_id"] = self.site_to_int_id_mapping.loc[im_df.site_id].values
        if "rel_int_id" not in im_df.columns:
            im_df["rel_int_id"] = self.rel_to_int_id_mapping.loc[im_df.rel_id].values

        im_df["record_int_id"] = self._get_record_int_id(
            im_df.event_int_id.values, im_df.rel_int_id.values, im_df.site_int_id.values
        )
        im_df = im_df[self.record_im_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO record_ims 
            SELECT * FROM im_df
            """
        )

        logger.debug(f"Inserted IM data for {len(im_df)} records.")

    def add_event_data(self, event_df: pd.DataFrame) -> None:
        """
        Add event data to the 'events' table.

        Parameters
        ----------
        event_id : str
            Unique identifier for the event.
        median_info : pd.Series
            Series containing event metadata.
        """
        if "event_int_id" not in event_df.columns:
            assert self.max_event_int_id is None
            event_df["event_int_id"] = np.arange(1, len(event_df) + 1, dtype=int)
        event_df = event_df[self.event_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO events 
            SELECT * FROM event_df
            """
        )
        logger.info(f"Inserted event data for {len(event_df)} events.")

    def add_realisation_data(self, rel_df: pd.DataFrame) -> None:
        """
        Add realisation data to the 'realisations' table.

        Parameters
        ----------
        rel_df : pd.DataFrame
            DataFrame containing realisation data. 
            Index needs to be rel_id.
        """
        # Get event_int_id for the realisations
        if "event_int_id" not in rel_df.columns:
            assert "event_id" in rel_df.columns, "Must have event_id column in rel_df"
            rel_df["event_int_id"] = self.event_to_int_id_mapping.loc[
                rel_df.event_id
            ].values

        # Use index as rel_id if not already present
        # (Assumes multi-index??)
        if "rel_id" not in rel_df.columns:
            assert self.max_rel_int_id is None
            rel_df = rel_df.sort_index().reset_index().rename(columns={"index": "rel_id"})

        # Assign rel_int_id if not already present
        if "rel_int_id" not in rel_df.columns:
            assert self.max_rel_int_id is None
            rel_df["rel_int_id"] = np.arange(1, len(rel_df) + 1, dtype=int)

        rel_df = rel_df[self.realisation_table_columns]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO realisations 
            SELECT * FROM rel_df
            """
        )

        logger.info(f"Inserted realisation data for {len(rel_df)} realisations.")

    
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

        return (
            (event_int_id.astype(np.int64) * p1)
            ^ (rel_int_id.astype(np.int64) * p2)
            ^ (site_int_id.astype(np.int64) * p3) % 100000000
        )

    def _create_tables(self):
        """
        Creates tables in the database if they don't exist.
        Only called in write mode.
        """
        if self.readonly:
            logger.warning("Attempting to create tables in readonly mode, ignoring.")
            return

        # Event table
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
            event_int_id INTEGER PRIMARY KEY,
            event_id TEXT UNIQUE,
            magnitude REAL,
            sim_type INTEGER,
            fault_type ENUM ('OTHER_CRUSTAL_FAULTING', 'NORMAL_FAULTING', 'PLATE_BOUNDARY', 'INTERFACE_FAULTING', 'DS_POINT_SOURCE'),
            tect_type ENUM ('ACTIVE_SHALLOW', 'VOLCANIC', 'SUBDUCTION_INTERFACE', 'SUBDUCTION_SLAB'),
            dip REAL,
            dtop REAL,
            dbottom REAL,
            length REAL,
            plane_count INTEGER,
            dip_dir REAL)"""
        )

        # Realisation table
        self._conn.execute(
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

        # Site table
        self._conn.execute(
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

        # Site-Event table
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS site_event (
            site_event_int_id BIGINT PRIMARY KEY,
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

        # Simulation records
        im_columns = ", ".join(
            [
                f"{cur_col} REAL"
                for cur_col in constants.DB_PSA_KEYS + constants.NON_PSA_IMS
            ]
        )
        self._conn.execute(
            f"""    
            CREATE TABLE IF NOT EXISTS record_ims (
            record_int_id BIGINT PRIMARY KEY,
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
