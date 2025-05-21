from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
import sqlite3

from . import constants

logger = logging.getLogger(__name__)


class IMDB:

    def __init__(self, db_ffp: Path):
        self.db_ffp = db_ffp
        self.conn = None
        self.cursor = None

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

    def __enter__(self) -> "IMDB":
        self.conn = sqlite3.connect(self.db_ffp)
        self.cursor = self.conn.cursor()

        # Performance optimizations
        self.cursor.execute("PRAGMA journal_mode = MEMORY")
        self.cursor.execute("PRAGMA synchronous = OFF")
        self.cursor.execute("PRAGMA cache_size = -500000")  # ~100MB cache

        self._create_tables()

        return self

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

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
            self.cursor.execute("PRAGMA table_info(events)")
            self._event_table_columns = np.array(
                [column_info[1] for column_info in self.cursor.fetchall()]
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
            self.cursor.execute("PRAGMA table_info(realisations)")
            self._realisation_table_columns = np.array(
                [column_info[1] for column_info in self.cursor.fetchall()]
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
            self.cursor.execute("PRAGMA table_info(sites)")
            self._site_table_columns = np.array(
                [column_info[1] for column_info in self.cursor.fetchall()]
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
            self.cursor.execute("PRAGMA table_info(site_event)")
            self._site_event_table_columns = np.array(
                [column_info[1] for column_info in self.cursor.fetchall()]
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
            self.cursor.execute("PRAGMA table_info(record_ims)")
            self._record_im_table_columns = np.array(
                [column_info[1] for column_info in self.cursor.fetchall()]
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
        if self._event_to_int_id_mapping is None:
            self._event_to_int_id_mapping = (
                pd.read_sql("SELECT event_id, event_int_id FROM events", self.conn)
                .set_index("event_id")
                .squeeze()
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
        if self._event_int_to_id_mapping is None:
            self._event_int_to_id_mapping = (
                pd.read_sql("SELECT event_int_id, event_id FROM events", self.conn)
                .set_index("event_int_id")
                .squeeze()
            ).astype("category")
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
        if self._rel_to_int_id_mapping is None:
            self._rel_to_int_id_mapping = (
                pd.read_sql("SELECT rel_id, rel_int_id FROM realisations", self.conn)
                .set_index("rel_id")
                .squeeze()
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
        if self._rel_int_to_id_mapping is None:
            self._rel_int_to_id_mapping = (
                pd.read_sql("SELECT rel_int_id, rel_id FROM realisations", self.conn)
                .set_index("rel_int_id")
                .squeeze()
            ).astype("category")
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
                pd.read_sql("SELECT site_id, site_int_id FROM sites", self.conn)
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
        if self._site_int_to_id_mapping is None:
            self._site_int_to_id_mapping = (
                pd.read_sql("SELECT site_int_id, site_id FROM sites", self.conn)
                .set_index("site_int_id")
                .squeeze()
            ).astype("category")
        return self._site_int_to_id_mapping

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

    def _get_site_event_int_id(
        self, site_int_id: np.ndarray, event_int_id: np.ndarray
    ) -> np.ndarray:
        """
        Generate unique integer IDs for each site-event pair
        based on site and event IDs.

        Parameters
        ----------
        site_int_id : np.ndarray
            Array of site integer IDs.
        event_int_id : np.ndarray
            Array of event integer IDs.

        Returns
        -------
        np.ndarray
            Array of unique integer IDs for each site-event pair.
        """
        # Prime multipliers for good distribution
        p1, p2 = 73856093, 19349663

        return (site_int_id * p1) ^ (event_int_id * p2) % 100000000

    def get_event_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all event data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all event data.
        """
        event_df = pd.read_sql("SELECT * FROM events", self.conn, index_col="event_int_id")
        event_df["tect_type"] = event_df["tect_type"].astype("category")

        return event_df

    def get_site_df(self, max_grid_level: int = None) -> pd.DataFrame:
        """
        Returns a DataFrame containing all site data.

        Parameters
        ----------
        max_grid_level : int, optional
            Maximum grid level to filter the sites.
            If None, all sites are returned.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all site data.
        """
        site_df = pd.read_sql("SELECT * FROM sites", self.conn, index_col="site_int_id")

        if max_grid_level is not None:
            site_df = site_df[site_df["grid_level"] <= max_grid_level]

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
        rel_df = pd.read_sql("SELECT * FROM realisations", self.conn, index_col="rel_int_id")
        rel_df["event_id"] = self.event_int_to_id_mapping[
            rel_df.event_int_id.values
        ].values

        if events is not None:
            rel_df = rel_df[rel_df["event_id"].isin(events)]

        return rel_df

    def get_site_event_df(self, sites: np.ndarray, max_rrup: float | None = None) -> pd.DataFrame:
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

        start = time.time()
        site_int_ids = self.site_to_int_id_mapping[sites].values.tolist()
        site_event_df = pd.read_sql(
            f"""
            SELECT se.*
            FROM site_event se
            INNER JOIN (SELECT site_int_id FROM sites WHERE site_int_id IN ({", ".join("?" * len(site_int_ids))})) sf
            ON sf.site_int_id = se.site_int_id
            {"WHERE se.rrup <= {}".format(max_rrup) if max_rrup is not None else ""}
            """,
            self.conn,
            params=site_int_ids,
            index_col="site_event_int_id",
        )
        print(f"Took: {time.time() - start:.3f} to get site-event data for {len(site_int_ids)} sites.")

        site_event_df["site_id"] = self.site_int_to_id_mapping[
            site_event_df.site_int_id.values
        ].values
        site_event_df["event_id"] = self.event_int_to_id_mapping[
            site_event_df.event_int_id.values
        ].values

        return site_event_df

    def get_record_info_df(
        self, events: np.ndarray | None = None, sites: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing
        the record_id, event_id, rel_id, and site_id.
        """
        logger.info("Getting record info from the database.")

        if events is not None and sites is not None:
            site_int_ids = self.site_to_int_id_mapping[sites].values.tolist()
            event_int_ids = self.event_to_int_id_mapping[events].values.tolist()
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
                self.conn,
                params=site_int_ids + event_int_ids,
                index_col="record_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for {len(site_int_ids)} sites and {len(event_int_ids)} events."
            )
        elif sites is not None:
            site_int_ids = self.site_to_int_id_mapping[sites].values.tolist()
            start = time.time()
            record_info_df = pd.read_sql(
                f"""
                SELECT r.record_int_id, r.event_int_id, r.site_int_id, r.rel_int_id
                FROM record_ims r
                INNER JOIN (SELECT site_int_id FROM sites WHERE site_int_id IN ({", ".join("?" * len(site_int_ids))})) sf
                ON sf.site_int_id = r.site_int_id
                """,
                self.conn,
                params=site_int_ids,
                index_col="record_int_id",
            )
            logger.info(
                f"Took: {time.time() - start:.3f}s to get record info for {len(site_int_ids)} sites."
            )
        elif events is not None:
            raise NotImplementedError()

        else:
            logger.warning(
                "No filters applied, return all records. Currently this is not supported."
            )
            return None

        # Map the integer IDs back to their original values
        record_info_df["event_id"] = self.event_int_to_id_mapping[
            record_info_df.event_int_id.values
        ].values
        record_info_df["site_id"] = self.site_int_to_id_mapping[
            record_info_df.site_int_id.values
        ].values
        record_info_df["rel_id"] = self.rel_int_to_id_mapping[
            record_info_df.rel_int_id.values
        ].values

        return record_info_df

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
        self.cursor.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,))
        if self.cursor.fetchone() is not None:
            logger.warning(f"Event ID {event_id} already exists. Skipping insert.")
            return

        # Insert the event data into the database
        columns = ", ".join(self.event_table_columns[1:])
        placeholders = ", ".join(["?"] * len(self.event_table_columns[1:]))
        query = f"""
        INSERT INTO events (
            {columns}
        ) VALUES ({placeholders})"""
        values = [event_id] + [median_info[col] for col in self.event_table_columns[2:]]
        self.cursor.execute(query, values)
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
        rel_df["event_int_id"] = self.event_to_int_id_mapping[rel_df.event_id.iloc[0]]

        # Check if realisation for the event already exists in the database
        self.cursor.execute(
            "SELECT 1 FROM realisations WHERE event_int_id = ?",
            (rel_df.event_int_id.iloc[0],),
        )
        if self.cursor.fetchone() is not None:
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
        self.cursor.executemany(query, values)

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
        self.cursor.executemany(query, values)
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

        site_event_df["site_int_id"] = site_event_df.site_id.map(
            self.site_to_int_id_mapping
        ).astype(int)
        site_event_df["event_int_id"] = site_event_df.event_id.map(
            self.event_to_int_id_mapping
        ).astype(int)
        site_event_df["site_event_int_id"] = self._get_site_event_int_id(
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
        self.cursor.executemany(query, values)
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
        im_df = im_df.rename(columns=constants.PSA_KEYS_TO_DB)
        assert np.all(np.isin(self.record_im_table_columns[4:], im_df.columns))

        im_df["event_int_id"] = im_df.event_id.map(self.event_to_int_id_mapping)
        im_df["rel_int_id"] = im_df.rel_id.map(self.rel_to_int_id_mapping)
        im_df["site_int_id"] = im_df.site_id.map(self.site_to_int_id_mapping)
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
        self.cursor.executemany(query, values)

        logger.debug(f"Inserted IM data for {len(im_df)} records.")

    def _create_tables(self) -> None:
        # Event table
        self.cursor.execute(
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
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_events_event_id 
               ON events (event_id)"""
        )

        # Realisation table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realisations (
            rel_int_id INTEGER PRIMARY KEY,
            rel_id TEXT UNIQUE,
            event_int_id INT,
            magnitude REAL,
            rake REAL,
            shypo REAL,
            dhypo REAL,
            FOREIGN KEY (event_int_id) REFERENCES events (event_int_id))"""
        )
        # Create index for fast access using rel_id
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_realisations_rel_id 
               ON realisations (rel_id)"""
        )

        # Site table
        self.cursor.execute(
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
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_sites_site_id 
               ON sites (site_id)"""
        )

        # Site-Event table
        self.cursor.execute(
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
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_site_event_site_id 
                ON site_event (site_int_id)"""
        )
        # Create index for fast access using event_int_id
        self.cursor.execute(
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
        self.cursor.execute(
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
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_event_id 
               ON record_ims (event_int_id)"""
        )
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_rel_id 
               ON record_ims (rel_int_id)"""
        )
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_site_id 
               ON record_ims (site_int_id)"""
        )
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_record_ims_site_event_rel
                ON record_ims (site_int_id, event_int_id, rel_int_id)"""
        )
