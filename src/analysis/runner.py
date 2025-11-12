import sqlite3
import time
from typing import Dict, Any

from src.options.base import Option
from src.analytical.black_scholes import BlackScholes
from src.numerical.trees import BinomialTree
from src.data import database

def run_analytical_pricing(conn: sqlite3.Connection, option: Option) -> Dict[str, Any]:
    """
    Runs the Black-Scholes analytical pricing model and saves results to the database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option : Option
        The option to price.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the option_id and result_id.
    """
    # Insert option into the database
    option_params = {
        "S": option.S, "K": option.K, "T": option.T, "r": option.r,
        "sigma": option.sigma, "option_type": option.option_type,
        "dividend_yield": option.dividend_yield
    }
    option_id = database.insert_option(conn, option_params)

    start_time = time.time()
    bs_results = BlackScholes.price(option)
    computation_time = time.time() - start_time

    # Insert results into the database
    result_data = {
        "option_id": option_id,
        "method": "BlackScholes",
        "price": bs_results["price"],
        "delta": bs_results["delta"],
        "gamma": bs_results["gamma"],
        "vega": bs_results["vega"],
        "theta": bs_results["theta"],
        "rho": bs_results["rho"],
        "computation_time": computation_time
    }
    result_id = database.insert_result(conn, result_data)

    return {"option_id": option_id, "result_id": result_id}

def run_binomial_tree_pricing(conn: sqlite3.Connection, option: Option, n_steps: int = 100, american: bool = False) -> Dict[str, Any]:
    """
    Runs the Binomial Tree pricing model and saves results to the database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option : Option
        The option to price.
    n_steps : int
        Number of steps for the binomial tree.
    american : bool
        Whether to price an American option.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the option_id and result_id.
    """
    # Insert option into the database
    option_params = {
        "S": option.S, "K": option.K, "T": option.T, "r": option.r,
        "sigma": option.sigma, "option_type": option.option_type,
        "dividend_yield": option.dividend_yield
    }
    option_id = database.insert_option(conn, option_params)

    bt = BinomialTree(option, n_steps)
    if american:
        price, computation_time = bt.american()
        method_name = f"BinomialTree_American_{n_steps}_steps"
    else:
        price, computation_time = bt.european()
        method_name = f"BinomialTree_European_{n_steps}_steps"

    # Binomial tree does not directly calculate Greeks, so they are None
    result_data = {
        "option_id": option_id,
        "method": method_name,
        "price": price,
        "delta": None,
        "gamma": None,
        "vega": None,
        "theta": None,
        "rho": None,
        "computation_time": computation_time
    }
    result_id = database.insert_result(conn, result_data)

    return {"option_id": option_id, "result_id": result_id}


if __name__ == "__main__":
    # Example usage
    conn = database.connect_db()
    database.create_tables(conn)

    # Define an option
    test_option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

    # Run analytical pricing and save to DB
    analytical_result_info = run_analytical_pricing(conn, test_option)
    print(f"Analytical pricing results saved: {analytical_result_info}")

    # Run binomial tree pricing (European) and save to DB
    binomial_european_result_info = run_binomial_tree_pricing(conn, test_option, n_steps=500, american=False)
    print(f"Binomial European pricing results saved: {binomial_european_result_info}")

    # Run binomial tree pricing (American) and save to DB
    binomial_american_result_info = run_binomial_tree_pricing(conn, test_option, n_steps=500, american=True)
    print(f"Binomial American pricing results saved: {binomial_american_result_info}")

    conn.close()
    print("Database connection closed.")