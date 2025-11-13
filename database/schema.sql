-- SQL script to create tables for the option pricing database

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
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    parameters TEXT, -- Store experiment-specific parameters as JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    option_id INTEGER NOT NULL,
    method_id INTEGER NOT NULL,
    experiment_id INTEGER,
    price REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    rho REAL,
    computation_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (option_id) REFERENCES options (id),
    FOREIGN KEY (method_id) REFERENCES methods (id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

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
);
