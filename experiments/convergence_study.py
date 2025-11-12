import sqlite3
import time
import numpy as np
from typing import List, Dict, Any

from src.data import database
from src.options.base import Option
from src.analytical.black_scholes import BlackScholes
from src.numerical.trees import BinomialTree
from src.numerical.monte_carlo import MonteCarlo
from src.numerical.finite_difference import FiniteDifference
from src.analysis.runner import run_analytical_pricing, run_binomial_tree_pricing, run_monte_carlo_pricing, run_finite_difference_pricing

def run_convergence_study(conn: sqlite3.Connection, option: Option,
                          n_steps_range: List[int], n_paths_range: List[int],
                          M_range: List[int], N_range: List[int]):
    """
    Performs a convergence study for numerical pricing methods.

    Parameters:
    -----------
    conn : sqlite3.Connection
        The database connection object.
    option : Option
        The option to price.
    n_steps_range : List[int]
        List of number of steps for Binomial Tree.
    n_paths_range : List[int]
        List of number of paths for Monte Carlo.
    M_range : List[int]
        List of spatial grid points for Finite Difference.
    N_range : List[int]
        List of time steps for Finite Difference.
    """
    print(f"Starting convergence study for option: {option}")

    # Run Analytical Pricing once for benchmark
    analytical_results = run_analytical_pricing(conn, option)
    analytical_price = BlackScholes.price(option)["price"]
    print(f"Analytical Price: {analytical_price:.4f}")

    # --- Binomial Tree Convergence ---
    print("\nRunning Binomial Tree convergence study...")
    for n_steps in n_steps_range:
        print(f"  Binomial Tree (European) with n_steps={n_steps}")
        run_binomial_tree_pricing(conn, option, n_steps=n_steps, american=False)
        print(f"  Binomial Tree (American) with n_steps={n_steps}")
        run_binomial_tree_pricing(conn, option, n_steps=n_steps, american=True)

    # --- Monte Carlo Convergence ---
    print("\nRunning Monte Carlo convergence study...")
    for n_paths in n_paths_range:
        print(f"  Monte Carlo (Standard) with n_paths={n_paths}")
        run_monte_carlo_pricing(conn, option, n_paths=n_paths, method_type="standard", seed=42)
        print(f"  Monte Carlo (Antithetic) with n_paths={n_paths}")
        run_monte_carlo_pricing(conn, option, n_paths=n_paths, method_type="antithetic_variates", seed=42)
        # For control variates, we need the analytical price
        print(f"  Monte Carlo (Control Variates) with n_paths={n_paths}")
        run_monte_carlo_pricing(conn, option, n_paths=n_paths, method_type="control_variates", seed=42, control_price=analytical_price)

    # --- Finite Difference Convergence ---
    print("\nRunning Finite Difference convergence study...")
    for M, N in zip(M_range, N_range): # Assuming M and N scale together for simplicity
        print(f"  Finite Difference (Explicit) with M={M}, N={N}")
        run_finite_difference_pricing(conn, option, M=M, N=N, method_type="explicit")
        print(f"  Finite Difference (Implicit) with M={M}, N={N}")
        run_finite_difference_pricing(conn, option, M=M, N=N, method_type="implicit")
        print(f"  Finite Difference (Crank-Nicolson) with M={M}, N={N}")
        run_finite_difference_pricing(conn, option, M=M, N=N, method_type="crank_nicolson")

    print("\nConvergence study complete.")


if __name__ == "__main__":
    conn = database.connect_db()
    database.create_tables(conn)

    # Define a test option
    test_option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

    # Define ranges for convergence study
    n_steps_range = [50, 100, 200, 400, 800]
    n_paths_range = [10000, 20000, 50000, 100000, 200000]
    M_range = [50, 100, 200, 400, 800] # Spatial steps
    N_range = [50, 100, 200, 400, 800] # Time steps

    run_convergence_study(conn, test_option, n_steps_range, n_paths_range, M_range, N_range)

    conn.close()
    print("Database connection closed.")
