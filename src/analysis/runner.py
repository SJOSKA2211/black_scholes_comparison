import sqlite3
import time
from typing import Dict, Any

from src.options.base import Option
from src.analytical.black_scholes import BlackScholes
from src.numerical.trees import BinomialTree
from src.numerical.monte_carlo import MonteCarlo
from src.numerical.finite_difference import FiniteDifference
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

def run_monte_carlo_pricing(conn: sqlite3.Connection, option: Option, n_paths: int = 10000,
                            n_steps: int = 100, seed: int = None, method_type: str = "standard",
                            control_price: float = None) -> Dict[str, Any]:
    """
    Runs the Monte Carlo pricing model and saves results to the database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option : Option
        The option to price.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int
        Random seed for reproducibility.
    method_type : str
        Type of Monte Carlo method ("standard", "antithetic_variates", "control_variates").
    control_price : float, optional
        Analytical price of control variate, required for "control_variates" method.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the option_id and result_id.
    """
    if method_type == "control_variates" and control_price is None:
        raise ValueError("control_price must be provided for 'control_variates' method type.")

    # Insert option into the database
    option_params = {
        "S": option.S, "K": option.K, "T": option.T, "r": option.r,
        "sigma": option.sigma, "option_type": option.option_type,
        "dividend_yield": option.dividend_yield
    }
    option_id = database.insert_option(conn, option_params)

    mc = MonteCarlo(option, n_paths=n_paths, n_steps=n_steps, seed=seed)
    mc_results = {}

    # start_time = time.time() # Moved inside the if/elif block for more accurate timing of specific MC method
    if method_type == "standard":
        mc_results = mc.standard()
    elif method_type == "antithetic_variates":
        mc_results = mc.antithetic_variates()
    elif method_type == "control_variates":
        # Note: For control variates, we use analytical Black-Scholes price as the control.
        # This means we would need to calculate it here or pass it in.
        # For simplicity, we are passing it in as `control_price`.
        mc_results = mc.control_variates(control_price)
    else:
        raise ValueError(f"Unknown Monte Carlo method type: {method_type}")
    # computation_time = time.time() - start_time # Moved inside the if/elif block

    # Insert results into the database
    result_data = {
        "option_id": option_id,
        "method": f"MonteCarlo_{method_type}_{n_paths}_paths",
        "price": mc_results["price"],
        "delta": None,  # Monte Carlo doesn't directly give Greeks
        "gamma": None,
        "vega": None,
        "theta": None,
        "rho": None,
        "computation_time": mc_results["computation_time"]
    }
    result_id = database.insert_result(conn, result_data)

    return {"option_id": option_id, "result_id": result_id}

def run_finite_difference_pricing(conn: sqlite3.Connection, option: Option, M: int = 100, N: int = 100,
                                  S_max: float = None, method_type: str = "explicit") -> Dict[str, Any]:
    """
    Runs the Finite Difference pricing model and saves results to the database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option : Option
        The option to price.
    M : int
        Number of spatial grid points.
    N : int
        Number of time steps.
    S_max : float, optional
        Maximum stock price for grid.
    method_type : str
        Type of Finite Difference method ("explicit", "implicit", "crank_nicolson").

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

    fd = FiniteDifference(option, M=M, N=N, S_max=S_max)
    fd_results = {}

    if method_type == "explicit":
        fd_results = fd.explicit()
    elif method_type == "implicit":
        fd_results = fd.implicit()
    elif method_type == "crank_nicolson":
        fd_results = fd.crank_nicolson()
    else:
        raise ValueError(f"Unknown Finite Difference method type: {method_type}")

    # Insert results into the database
    result_data = {
        "option_id": option_id,
        "method": f"FiniteDifference_{method_type}_M{M}_N{N}",
        "price": fd_results["price"],
        "delta": None,  # Finite Difference doesn't directly give Greeks
        "gamma": None,
        "vega": None,
        "theta": None,
        "rho": None,
        "computation_time": fd_results["computation_time"]
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

    # Run Monte Carlo pricing (Standard) and save to DB
    mc_standard_result_info = run_monte_carlo_pricing(conn, test_option, n_paths=10000, method_type="standard", seed=42)
    print(f"Monte Carlo Standard pricing results saved: {mc_standard_result_info}")

    # Run Monte Carlo pricing (Antithetic Variates) and save to DB
    mc_antithetic_result_info = run_monte_carlo_pricing(conn, test_option, n_paths=10000, method_type="antithetic_variates", seed=42)
    print(f"Monte Carlo Antithetic Variates pricing results saved: {mc_antithetic_result_info}")

    # Get analytical price for control variate
    analytical_price_for_control = BlackScholes.price(test_option)["price"]
    # Run Monte Carlo pricing (Control Variates) and save to DB
    mc_control_result_info = run_monte_carlo_pricing(conn, test_option, n_paths=10000, method_type="control_variates", seed=42, control_price=analytical_price_for_control)
    print(f"Monte Carlo Control Variates pricing results saved: {mc_control_result_info}")

    # Run Finite Difference pricing (Explicit) and save to DB
    fd_explicit_result_info = run_finite_difference_pricing(conn, test_option, M=100, N=100, method_type="explicit")
    print(f"Finite Difference Explicit pricing results saved: {fd_explicit_result_info}")

    # Run Finite Difference pricing (Implicit) and save to DB
    fd_implicit_result_info = run_finite_difference_pricing(conn, test_option, M=100, N=100, method_type="implicit")
    print(f"Finite Difference Implicit pricing results saved: {fd_implicit_result_info}")

    # Run Finite Difference pricing (Crank-Nicolson) and save to DB
    fd_crank_nicolson_result_info = run_finite_difference_pricing(conn, test_option, M=100, N=100, method_type="crank_nicolson")
    print(f"Finite Difference Crank-Nicolson pricing results saved: {fd_crank_nicolson_result_info}")

    conn.close()
    print("Database connection closed.")