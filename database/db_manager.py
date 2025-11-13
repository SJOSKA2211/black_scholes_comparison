"""
Database Manager Module
Manages SQLite database connections and operations for storing option parameters,
method results, and market data. Implemented as a singleton.
"""
import sqlite3
import sqlitecloud # Import sqlitecloud
from typing import List, Tuple, Dict, Any
import os

class DBManager:
    _instance = None
    DATABASE_NAME = "option_pricing_data.db"
    SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")

    def __new__(cls, db_name: str = DATABASE_NAME):
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_name: str = DATABASE_NAME):
        if self._initialized:
            return
        self.db_name = db_name
        self.conn = None
        self._connect()
        self._create_tables_from_schema()
        self._initialized = True

    def _connect(self):
        """Establishes a connection to the SQLite database (local or SQLiteCloud)."""
        if self.conn is not None:
            self.conn.close()
        
        if self.db_name.startswith("sqlitecloud://"):
            # Connect to SQLiteCloud
            try:
                self.conn = sqlitecloud.connect(self.db_name)
                # SQLiteCloud client might not have row_factory directly,
                # or it might behave differently. We'll assume it's compatible for now.
                # If issues arise, we might need to fetch data differently or wrap cursor.
            except Exception as e:
                raise ConnectionError(f"Failed to connect to SQLiteCloud: {e}")
        else:
            # Connect to local SQLite file
            self.conn = sqlite3.connect(self.db_name)
            self.conn.row_factory = sqlite3.Row # Allows accessing columns by name

    def _create_tables_from_schema(self):
        """Creates tables in the database by executing the schema.sql script."""
        if not os.path.exists(self.SCHEMA_FILE):
            raise FileNotFoundError(f"Schema file not found: {self.SCHEMA_FILE}")
        
        with open(self.SCHEMA_FILE, 'r') as f:
            schema_sql = f.read()
        
        cursor = self.conn.cursor()
        # SQLiteCloud might not support executescript directly, or might require different handling
        # For now, we'll try executescript. If it fails, we might need to parse and execute statements individually.
        try:
            cursor.executescript(schema_sql)
        except AttributeError: # Fallback for sqlitecloud if executescript is not available
            for statement in schema_sql.split(';'):
                if statement.strip():
                    cursor.execute(statement)
        self.conn.commit()

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def insert_option(self, option_params: Dict[str, Any]) -> int:
        """
        Inserts option parameters into the 'options' table.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO options (S, K, T, r, sigma, option_type, dividend_yield)
            VALUES (:S, :K, :T, :r, :sigma, :option_type, :dividend_yield)
        """,
        option_params)
        self.conn.commit()
        return cursor.lastrowid

    def insert_method(self, method_params: Dict[str, Any]) -> int:
        """
        Inserts a pricing method into the 'methods' table if it doesn't exist,
        otherwise returns the existing method_id.
        """
        cursor = self.conn.cursor()
        
        # Check if method already exists
        cursor.execute("SELECT id FROM methods WHERE name = :name", {"name": method_params["name"]})
        existing_method = cursor.fetchone()
        if existing_method:
            return existing_method[0] # Return existing ID
        
        # If not, insert new method
        cursor.execute("""
            INSERT INTO methods (name, description)
            VALUES (:name, :description)
        """,
        method_params)
        self.conn.commit()
        return cursor.lastrowid

    def insert_experiment(self, experiment_params: Dict[str, Any]) -> int:
        """
        Inserts experiment details into the 'experiments' table.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (name, description, parameters, start_time, end_time)
            VALUES (:name, :description, :parameters, :start_time, :end_time)
        """,
        experiment_params)
        self.conn.commit()
        return cursor.lastrowid

    def insert_result(self, result_data: Dict[str, Any]) -> int:
        """
        Inserts pricing method results into the 'results' table.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO results (option_id, method_id, experiment_id, price, delta, gamma, vega, theta, rho, computation_time)
            VALUES (:option_id, :method_id, :experiment_id, :price, :delta, :gamma, :vega, :theta, :rho, :computation_time)
        """,
        result_data)
        self.conn.commit()
        return cursor.lastrowid

    def insert_market_data(self, market_data_params: Dict[str, Any]) -> int:
        """
        Inserts real market data into the 'market_data' table.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_data (ticker, trade_date, expiration_date, strike_price, option_type,
                                     bid_price, ask_price, last_price, implied_volatility, underlying_price)
            VALUES (:ticker, :trade_date, :expiration_date, :strike_price, :option_type,
                    :bid_price, :ask_price, :last_price, :implied_volatility, :underlying_price)
        """,
        market_data_params)
        self.conn.commit()
        return cursor.lastrowid

    def clear_market_data(self, ticker: str = None):
        """
        Clears market data from the 'market_data' table.
        If ticker is provided, clears data only for that ticker.
        Otherwise, clears all market data.
        """
        cursor = self.conn.cursor()
        if ticker:
            cursor.execute("DELETE FROM market_data WHERE ticker = ?", (ticker,))
        else:
            cursor.execute("DELETE FROM market_data")
        self.conn.commit()

    def get_all_methods(self) -> List[sqlite3.Row]:
        """Retrieves all methods from the 'methods' table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM methods")
        return cursor.fetchall()

    def get_all_experiments(self) -> List[sqlite3.Row]:
        """Retrieves all experiments from the 'experiments' table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments")
        return cursor.fetchall()

    def get_all_options(self) -> List[sqlite3.Row]:
        """Retrieves all options from the 'options' table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM options")
        return cursor.fetchall()

    def get_results_for_option(self, option_id: int) -> List[sqlite3.Row]:
        """Retrieves all results for a given option_id from the 'results' table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM results WHERE option_id = ?", (option_id,))
        return cursor.fetchall()

    def get_market_data_by_ticker(self, ticker: str) -> List[sqlite3.Row]:
        """Retrieves market data for a given ticker."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM market_data WHERE ticker = ?", (ticker,))
        return cursor.fetchall()

    # Add more query methods as needed