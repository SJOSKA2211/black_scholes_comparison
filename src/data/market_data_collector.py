import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

from src.data import database

def fetch_and_store_options_data(conn: sqlite3.Connection, ticker: str, expiration_dates: List[str] = None):
    """
    Fetches options data for a given ticker and stores it in the database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    ticker : str
        The stock ticker symbol (e.g., "SPY").
    expiration_dates : List[str], optional
        A list of expiration dates in 'YYYY-MM-DD' format. If None, fetches all available.
    """
    try:
        tk = yf.Ticker(ticker)
        
        if expiration_dates is None:
            expiration_dates = tk.options # Get all available expiration dates
        
        print(f"Fetching options data for {ticker} for expirations: {expiration_dates}")

        for exp_date in expiration_dates:
            try:
                opt = tk.option_chain(exp_date)
                
                # Process calls
                calls = opt.calls
                for index, row in calls.iterrows():
                    market_data_params = {
                        "ticker": ticker,
                        "trade_date": datetime.now().strftime('%Y-%m-%d'), # Current date as trade date
                        "expiration_date": exp_date,
                        "strike_price": row['strike'],
                        "option_type": "call",
                        "bid_price": row['bid'],
                        "ask_price": row['ask'],
                        "last_price": row['lastPrice'],
                        "implied_volatility": row['impliedVolatility'],
                        "underlying_price": tk.info['regularMarketPrice'] # Get current underlying price
                    }
                    database.insert_market_data(conn, market_data_params)
                
                # Process puts
                puts = opt.puts
                for index, row in puts.iterrows():
                    market_data_params = {
                        "ticker": ticker,
                        "trade_date": datetime.now().strftime('%Y-%m-%d'),
                        "expiration_date": exp_date,
                        "strike_price": row['strike'],
                        "option_type": "put",
                        "bid_price": row['bid'],
                        "ask_price": row['ask'],
                        "last_price": row['lastPrice'],
                        "implied_volatility": row['impliedVolatility'],
                        "underlying_price": tk.info['regularMarketPrice']
                    }
                    database.insert_market_data(conn, market_data_params)
                
                print(f"Successfully stored options for {ticker} expiring on {exp_date}")

            except Exception as e:
                print(f"Error fetching/storing options for {ticker} expiring on {exp_date}: {e}")

    except Exception as e:
        print(f"Error fetching ticker data for {ticker}: {e}")

if __name__ == "__main__":
    conn = database.connect_db()
    database.create_tables(conn) # Ensure tables exist

    # Example usage: Fetch SPY options for a few upcoming expiration dates
    # You can get available expiration dates by running: yf.Ticker("SPY").options
    
    # For demonstration, let's pick a couple of recent/upcoming expiration dates
    # Replace with actual dates you want to fetch
    # Example: expiration_dates = ["2025-11-14", "2025-11-21", "2025-12-20"]
    
    # Fetch all available expiration dates for SPY
    spy_ticker = yf.Ticker("SPY")
    all_spy_expirations = spy_ticker.options
    
    # Fetch for the first 3 available expiration dates to avoid too much data
    selected_expirations = all_spy_expirations[:3] 
    
    fetch_and_store_options_data(conn, "SPY", expiration_dates=selected_expirations)
    
    conn.close()
    print("Market data collection complete.")