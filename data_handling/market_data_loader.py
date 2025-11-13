import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

from database.db_manager import DBManager

def download_historical_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data for a given ticker and date range.

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., "SPY").
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        A Pandas DataFrame containing the historical stock data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No historical data found for {ticker} between {start_date} and {end_date}")
        return data
    except Exception as e:
        print(f"Error downloading historical stock data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_and_store_options_data(db_manager: DBManager, ticker: str, expiration_dates: List[str] = None):
    """
    Fetches options data for a given ticker and stores it in the database.

    Parameters:
    -----------
    db_manager : DBManager
        The database manager object.
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
                    db_manager.insert_market_data(market_data_params)
                
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
                    db_manager.insert_market_data(market_data_params)
                
                print(f"Successfully stored options for {ticker} expiring on {exp_date}")

            except Exception as e:
                print(f"Error fetching/storing options for {ticker} expiring on {exp_date}: {e}")

    except Exception as e:
        print(f"Error fetching ticker data for {ticker}: {e}")

if __name__ == "__main__":
    db_manager = DBManager() # Instantiate the singleton DBManager

    # Example usage: Download historical stock data
    print("\nDownloading historical stock data for AAPL...")
    aapl_data = download_historical_stock_data("AAPL", "2023-01-01", "2023-01-31")
    if not aapl_data.empty:
        print("AAPL Historical Data (first 5 rows):")
        print(aapl_data.head())
    else:
        print("Failed to download AAPL historical data.")

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
    
    fetch_and_store_options_data(db_manager, "SPY", expiration_dates=selected_expirations)
    
    db_manager.close_connection()
    print("Market data collection complete.")