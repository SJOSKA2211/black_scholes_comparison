# Black-Scholes Option Pricing Comparative Analysis

This project conducts a comprehensive comparative analysis of finite difference methods, Monte Carlo simulations, and binomial/trinomial tree methods for solving the Black-Scholes partial differential equation in option pricing.

## Project Structure

```
black_scholes_comparison/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── options/
│   │   ├── __init__.py
│   │   ├── base.py           # Base Option class
│   │   ├── european.py       # European option specifics
│   │   └── american.py       # American option specifics
│   │
│   ├── analytical/
│   │   ├── __init__.py
│   │   └── black_scholes.py  # Analytical B-S formulas
│   │
│   ├── numerical/
│   │   ├── __init__.py
│   │   ├── finite_difference.py  # All FD methods
│   │   ├── monte_carlo.py        # MC methods
│   │   └── trees.py              # Binomial/trinomial trees
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_data.py    # Data collection
│   │   └── database.py       # Database operations
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── metrics.py        # Performance metrics
│       └── visualization.py  # Plotting functions
│
├── tests/
│   ├── __init__.py
│   ├── test_analytical.py
│   ├── test_finite_difference.py
│   ├── test_monte_carlo.py
│   └── test_trees.py
│
├── experiments/
│   ├── run_experiments.py    # Main experiment runner
│   ├── convergence_study.py  # Convergence analysis
│   └── market_validation.py  # Market data validation
│
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_finite_difference.ipynb
│   ├── 03_monte_carlo.ipynb
│   ├── 04_trees.ipynb
│   └── 05_results_analysis.ipynb
│
└── results/
    ├── figures/
    ├── tables/
    └── data/
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/black_scholes_comparison.git
    cd black_scholes_comparison
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```

## Running Tests

To run the unit tests, activate your virtual environment and execute:

```bash
pytest tests/
```

## Usage

**TODO:** This section will be populated with examples of how to use the implemented numerical methods, run experiments, and analyze results.
