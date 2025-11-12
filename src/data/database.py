"""
Database module for storing option parameters, method results, and market data.
Uses SQLite for simplicity.
"""
import sqlite3
from typing import List, Tuple, Dict, Any

DATABASE_NAME = "option_pricing_data.db"

def connect_db(db_name: str = DATABASE_NAME) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.

    Parameters:
    -----------
    db_name : str
        The name of the database file.

    Returns:
    --------
    sqlite3.Connection
        A connection object to the database.
    """
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def create_tables(conn: sqlite3.Connection):
    """
    Creates the necessary tables in the database if they don't already exist.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    """
    cursor = conn.cursor()

    # Table for storing option parameters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            S REAL NOT NULL,
            K REAL NOT NULL,
            T REAL NOT NULL,
            r REAL NOT NULL,
            sigma REAL NOT NULL,
            option_type TEXT NOT NULL,
            dividend_yield REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Table for storing results from different pricing methods
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            option_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            price REAL,
            delta REAL,
            gamma REAL,
            vega REAL,
            theta REAL,
            rho REAL,
            computation_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (option_id) REFERENCES options (id)
        )
    """
    )

    # Table for storing real market data (simplified for now)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            trade_date DATE NOT NULL,
            expiration_date DATE NOT NULL,
            strike_price REAL NOT NULL,
            option_type TEXT NOT NULL,
            bid_price REAL,
            ask_price REAL,
            last_price REAL,
            implied_volatility REAL,
            underlying_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()

def insert_option(conn: sqlite3.Connection, option_params: Dict[str, Any]) -> int:
    """
    Inserts option parameters into the 'options' table.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option_params : Dict[str, Any]
        A dictionary containing option parameters (S, K, T, r, sigma, option_type, dividend_yield).

    Returns:
    --------
    int
        The ID of the newly inserted option.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO options (S, K, T, r, sigma, option_type, dividend_yield)
        VALUES (:S, :K, :T, :r, :sigma, :option_type, :dividend_yield)
    """,
    option_params)
    conn.commit()
    return cursor.lastrowid

def insert_result(conn: sqlite3.Connection, result_data: Dict[str, Any]) -> int:
    """
    Inserts pricing method results into the 'results' table.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    result_data : Dict[str, Any]
        A dictionary containing result data (option_id, method, price, delta, gamma, vega, theta, rho, computation_time).

    Returns:
    --------
    int
        The ID of the newly inserted result.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results (option_id, method, price, delta, gamma, vega, theta, rho, computation_time)
        VALUES (:option_id, :method, :price, :delta, :gamma, :vega, :theta, :rho, :computation_time)
    """,
    result_data)
    conn.commit()
    return cursor.lastrowid

def insert_market_data(conn: sqlite3.Connection, market_data_params: Dict[str, Any]) -> int:
    """
    Inserts real market data into the 'market_data' table.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    market_data_params : Dict[str, Any]
        A dictionary containing market data (ticker, trade_date, expiration_date, strike_price,
        option_type, bid_price, ask_price, last_price, implied_volatility, underlying_price).

    Returns:
    --------
    int
        The ID of the newly inserted market data entry.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO market_data (ticker, trade_date, expiration_date, strike_price, option_type,
                                 bid_price, ask_price, last_price, implied_volatility, underlying_price)
        VALUES (:ticker, :trade_date, :expiration_date, :strike_price, :option_type,
                :bid_price, :ask_price, :last_price, :implied_volatility, :underlying_price)
    """,
    market_data_params)
    conn.commit()
    return cursor.lastrowid

def get_all_options(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Retrieves all options from the 'options' table.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.

    Returns:
    --------
    List[sqlite3.Row]
        A list of rows, where each row is an option.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM options")
    return cursor.fetchall()

def get_results_for_option(conn: sqlite3.Connection, option_id: int) -> List[sqlite3.Row]:
    """
    Retrieves all results for a given option_id from the 'results' table.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option_id : int
        The ID of the option.

    Returns:
    --------
    List[sqlite3.Row]
        A list of rows, where each row is a result.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results WHERE option_id = ?", (option_id,))
    return cursor.fetchall()

def get_market_data_by_ticker(conn: sqlite3.Connection, ticker: str) -> List[sqlite3.Row]:
    """
    Retrieves market data for a given ticker.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    ticker : str
        The stock ticker symbol.

    Returns:
    --------
    List[sqlite3.Row]
        A list of rows, where each row is a market data entry.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM market_data WHERE ticker = ?", (ticker,))
    return cursor.fetchall()

if __name__ == "__main__":
    # Example usage:
    conn = connect_db()
    create_tables(conn)
    print("Database and tables created successfully.")

    # Insert an example option
    option_params = {
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2,
        "option_type": "call", "dividend_yield": 0.0
    }
    option_id = insert_option(conn, option_params)
    print(f"Inserted option with ID: {option_id}")

    # Insert an example result for the option
    result_data = {
        "option_id": option_id, "method": "BlackScholes", "price": 10.45,
        "delta": 0.5, "gamma": 0.02, "vega": 0.2, "theta": -0.01, "rho": 0.1,
        "computation_time": 0.001
    }
    result_id = insert_result(conn, result_data)
    print(f"Inserted result with ID: {result_id}")

    # Insert example market data
    market_data_params = {
        "ticker": "SPX", "trade_date": "2025-11-12", "expiration_date": "2026-01-17",
        "strike_price": 4500.0, "option_type": "call", "bid_price": 10.0,
        "ask_price": 11.0, "last_price": 10.5, "implied_volatility": 0.25,
        "underlying_price": 4500.0
    }
    market_data_id = insert_market_data(conn, market_data_params)
    print(f"Inserted market data with ID: {market_data_id}")

    # Retrieve and print data
    print("\nAll options:")
    for opt in get_all_options(conn):
        print(dict(opt))

    print(f"\nResults for option ID {option_id}:")
    for res in get_results_for_option(conn, option_id):
        print(dict(res))

    print(f"\nMarket data for SPX:")
    for md in get_market_data_by_ticker(conn, "SPX"):
        print(dict(md))

    conn.close()
    print("Database connection closed.")
