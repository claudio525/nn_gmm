from pathlib import Path
import logging

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
    def event_table_columns(self) -> list:
        """
        Returns the column names of the events table.

        Returns
        -------
        list
            Column names of the events table.
        """
        self.cursor.execute("PRAGMA table_info(events)")
        return [column_info[1] for column_info in self.cursor.fetchall()]

    @property
    def realisation_table_columns(self) -> list:
        """
        Returns the column names of the realisations table.

        Returns
        -------
        list
            Column names of the realisations table.
        """
        self.cursor.execute("PRAGMA table_info(realisations)")
        return [column_info[1] for column_info in self.cursor.fetchall()]

    @property
    def site_table_columns(self) -> list:
        """
        Returns the column names of the sites table.

        Returns
        -------
        list
            Column names of the sites table.
        """
        self.cursor.execute("PRAGMA table_info(sites)")
        return [column_info[1] for column_info in self.cursor.fetchall()]

    @property
    def record_im_table_columns(self) -> list:
        """
        Returns the column names of the record_ims table.

        Returns
        -------
        list
            Column names of the record_ims table.
        """
        self.cursor.execute("PRAGMA table_info(record_ims)")
        return [column_info[1] for column_info in self.cursor.fetchall()]

    @property
    def event_to_int_id_mapping(self) -> dict:
        """
        Returns a mapping of event_ids to event_int_ids.

        Returns
        -------
        dict
            Mapping of event_ids to event_int_ids.
        """
        self.cursor.execute("SELECT event_id, event_int_id FROM events")
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    @property
    def rel_to_int_id_mapping(self) -> dict:
        """
        Returns a mapping of rel_ids to rel_int_ids.

        Returns
        -------
        dict
            Mapping of rel_ids to rel_int_ids.
        """
        self.cursor.execute("SELECT rel_id, rel_int_id FROM realisations")
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    @property
    def site_to_int_id_mapping(self) -> dict:
        """
        Returns a mapping of site_ids to site_int_ids.

        Returns
        -------
        dict
            Mapping of site_ids to site_int_ids.
        """
        self.cursor.execute("SELECT site_id, site_int_id FROM sites")
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def get_record_int_id(
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
        query = """
        INSERT INTO events (
            event_id, magnitude, sim_type, fault_type, tect_type, rake, dip, dtop, 
            dbottom, length, plane_count, dip_dir, shypo, dhypo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
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
        im_df["record_int_id"] = self.get_record_int_id(
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
        values = list(im_df[self.record_im_table_columns].itertuples(index=False, name=None))
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
            rake REAL,
            dip REAL,
            dtop REAL,
            dbottom REAL,
            length REAL,
            plane_count INTEGER,
            dip_dir REAL,
            shypo REAL,
            dhypo REAL)"""
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
            z2p5 REAL)"""
        )
        # Create index for fast access using site_id
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_sites_site_id 
               ON sites (site_id)"""
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
