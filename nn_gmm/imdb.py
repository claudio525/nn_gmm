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
        assert np.all(np.isin(self.event_table_columns[1:], median_info.index.values))

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
        values = [event_id] + [median_info[col] for col in self.event_table_columns[1:]]
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
        assert np.all(np.isin(self.realisation_table_columns[1:], rel_df.columns))
        assert np.all(rel_df.event_id == rel_df.event_id.iloc[0])
        rel_df = rel_df.sort_index()[self.realisation_table_columns[1:]]

        # Check if realisation for the event already exists in the database
        self.cursor.execute(
            "SELECT 1 FROM realisations WHERE event_id = ?", (rel_df.event_id.iloc[0],)
        )
        if self.cursor.fetchone() is not None:
            logger.warning(
                f"Realisations data for event {rel_df.event_id.iloc[0]}"
                f"already exists. Skipping insert."
            )
            return

        # Insert the realisation data into the database
        columns = ", ".join(self.realisation_table_columns)
        placeholders = ", ".join(["?"] * len(self.realisation_table_columns))
        query = f"""
        INSERT INTO realisations (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(rel_df.itertuples(index=True, name=None))
        self.cursor.executemany(query, values)

        logger.debug(f"Inserted realisation data for event {rel_df.event_id.iloc[0]}")

    def add_record_im_data(self, im_df: pd.DataFrame) -> None:
        """
        Add IM data to the 'record_ims' table.

        Parameters
        ----------
        im_df : pd.DataFrame
            DataFrame containing the IM data to add
        """
        im_df = im_df.rename(columns=constants.PSA_KEYS_TO_DB)
        assert np.all(np.isin(self.record_im_table_columns[1:], im_df.columns))
        im_df = im_df[self.record_im_table_columns[1:]]

        # Insert the IM data into the database
        # Note: If the record already exists, it will be skipped
        columns = ", ".join(self.record_im_table_columns)
        placeholders = ", ".join(["?"] * len(self.record_im_table_columns))
        query = f"""
        INSERT OR IGNORE INTO record_ims (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(im_df.itertuples(index=True, name=None))
        self.cursor.executemany(query, values)

        logger.debug(f"Inserted IM data for {len(im_df)} records.")

    def add_site_data(self, site_df: pd.DataFrame) -> None:
        """
        Add site data to the 'sites' table.

        Parameters
        ----------
        site_df : pd.DataFrame
            DataFrame containing the site data to add
        """
        assert np.all(np.isin(self.site_table_columns[1:], site_df.columns))
        site_df = site_df[self.site_table_columns[1:]]
        site_df = site_df.sort_index()

        # Insert the site data into the database
        # Note: If the site already exists, it will be skipped
        columns = ", ".join(self.site_table_columns)
        placeholders = ", ".join(["?"] * len(self.site_table_columns))
        query = f"""
        INSERT OR IGNORE INTO sites (
            {columns}
        ) VALUES ({placeholders})"""
        values = list(site_df.itertuples(index=True, name=None))
        self.cursor.executemany(query, values)
        logger.debug(f"Inserted site data for {len(site_df)} sites.")

    def _create_tables(self) -> None:
        # Event table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
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

        # Realisation table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realisations (
            rel_id TEXT PRIMARY KEY,
            event_id TEXT,
            magnitude REAL,
            rake REAL,
            dip REAL,
            dip_dir REAL,
            shypo REAL,
            dhypo REAL,
            FOREIGN KEY (event_id) REFERENCES events (event_id))"""
        )

        # Site table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sites (
                site_id TEXT PRIMARY KEY,
                lat REAL,
                lon REAL,
                vs30 REAL,
                z1p0 REAL,
                z2p5 REAL)"""
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
            record_id TEXT PRIMARY KEY,
            event_id TEXT,
            rel_id TEXT,
            site_id TEXT,
            {im_columns},
            FOREIGN KEY (event_id) REFERENCES events (event_id),
            FOREIGN KEY (rel_id) REFERENCES realisations (rel_id),
            FOREIGN KEY (site_id) REFERENCES sites (site_id)
            )
        """
        )
