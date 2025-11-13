"""
Test cases for Market Data Collector Module
"""
import pytest
import pandas as pd
import os
from datetime import datetime, timedelta
import yfinance as yf # Added import
from src.data.market_data_collector import download_historical_stock_data, fetch_and_store_options_data
from src.data.database import connect_db, create_tables

# Fixture for a temporary SQLite database
@pytest.fixture
def temp_db():
    db_name = "test_option_pricing_data.db"
    conn = connect_db(db_name)
    create_tables(conn)
    yield conn
    conn.close()
    os.remove(db_name)

def test_download_historical_stock_data_valid_ticker():
    ticker = "MSFT"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    df = download_historical_stock_data(ticker, start_date, end_date)
    
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    assert "Open" in df.columns
    assert "High" in df.columns
    assert "Low" in df.columns
    assert "Close" in df.columns
    assert "Volume" in df.columns

def test_download_historical_stock_data_invalid_ticker():
    ticker = "NONEXISTENTTICKER123"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    df = download_historical_stock_data(ticker, start_date, end_date)
    
    assert df.empty

# Note: Testing fetch_and_store_options_data directly can be slow and dependent on live data.
# For a more robust test, one would typically mock the yfinance API calls.
# For now, we'll do a basic test that it runs without immediate errors.
def test_fetch_and_store_options_data_basic(temp_db):
    conn = temp_db
    ticker = "SPY"
    # Fetch for a very near expiration date to limit data volume
    today = datetime.now()
    # Try to find an expiration date in the near future
    yf_ticker = yf.Ticker(ticker)
    options_exp_dates = yf_ticker.options
    
    selected_exp_date = None
    for exp_date_str in options_exp_dates:
        exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
        if exp_date > today:
            selected_exp_date = exp_date_str
            break
    
    if selected_exp_date:
        fetch_and_store_options_data(conn, ticker, expiration_dates=[selected_exp_date])
        # Verify some data was inserted
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data WHERE ticker = ?", (ticker,))
        count = cursor.fetchone()[0]
        assert count > 0
    else:
        pytest.skip(f"No suitable expiration date found for {ticker} to test options data fetching.")
