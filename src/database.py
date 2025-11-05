"""
Database operations module

Handles all PostgreSQL database interactions
"""

import psycopg2
import pandas as pd
import logging
from typing import List, Dict, Any
from .config import DatabaseConfig


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager

        Args:
            config: Database configuration
        """
        self.config = config

    def _get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password
        )

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as DataFrame

        Args:
            sql: SQL query string

        Returns:
            DataFrame with query results
        """
        conn = None
        try:
            conn = self._get_connection()
            data = pd.read_sql_query(sql, conn)
            conn.commit()
            return data
        finally:
            if conn:
                conn.close()

    def get_active_webcams(self) -> pd.DataFrame:
        """
        Get all active webcam URLs from database

        Returns:
            DataFrame with webcam information
        """
        return self.query("SELECT * FROM webcam_urls WHERE debug = false")

    def insert_or_update(self, table: str, data: List[Dict[str, Any]],
                        conflict_columns: List[str] = None,
                        dtypes: Dict[str, str] = None):
        """
        Insert or update records in database

        Args:
            table: Table name
            data: List of dictionaries containing row data
            conflict_columns: Columns to check for conflicts (for UPSERT)
            dtypes: Optional type casting dictionary
        """
        if conflict_columns is None:
            conflict_columns = []
        if dtypes is None:
            dtypes = {}

        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            for row in data:
                self._insert_or_update_row(cur, table, row, conflict_columns, dtypes)

            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as e:
            logging.exception("Error executing PostgreSQL")
            raise
        finally:
            if conn:
                conn.close()

    def _insert_or_update_row(self, cursor, table: str, data: Dict[str, Any],
                             conflict_columns: List[str],
                             dtypes: Dict[str, str]):
        """
        Insert or update a single row

        Args:
            cursor: Database cursor
            table: Table name
            data: Row data dictionary
            conflict_columns: Columns to check for conflicts
            dtypes: Type casting dictionary
        """
        # Escape single quotes in string values
        data = {
            k: v.replace("'", "''") if isinstance(v, str) else v
            for k, v in data.items()
        }

        keys = ", ".join(data.keys())

        # Format values
        def format_value(key, value):
            if isinstance(value, (int, float)):
                return str(value)
            elif value is None:
                dtype = f"::{dtypes[key]}" if key in dtypes else ""
                return f"null{dtype}"
            else:
                dtype = f"::{dtypes[key]}" if key in dtypes else ""
                return f"'{value}'{dtype}"

        values = ", ".join([format_value(k, v) for k, v in data.items()])

        # Build query
        if not conflict_columns:
            # Simple INSERT
            query = f"INSERT INTO {table} ({keys}) VALUES ({values});"
        else:
            # UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
            conflict_keys = ", ".join(conflict_columns)
            non_conflict_items = {k: v for k, v in data.items() if k not in conflict_columns}

            update_parts = [
                f"{k}={format_value(k, v)}"
                for k, v in non_conflict_items.items()
            ]
            update_clause = ", ".join(update_parts)

            query = f"""
                INSERT INTO {table} ({keys})
                VALUES ({values})
                ON CONFLICT ({conflict_keys})
                DO UPDATE SET {update_clause};
            """

        cursor.execute(query)

    def store_detections(self, detections: pd.DataFrame):
        """
        Store detection results in database

        Args:
            detections: DataFrame with columns: webcam, date, detections
        """
        # Localize timestamps to UTC
        if 'date' in detections.columns:
            detections.index = detections['date']
            detections.index = detections.index.tz_localize('Europe/Berlin')
            detections.index = detections.index.tz_convert('UTC')
            detections['date'] = detections.index

        # Store in database
        self.insert_or_update(
            table='webcam_detections',
            data=detections.to_dict(orient='records'),
            conflict_columns=['webcam', 'date']
        )
