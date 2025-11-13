"""
Test cases for Monte Carlo Methods
"""
import pytest
from src.options.base import Option
from src.numerical.monte_carlo import MonteCarlo
from src.analytical.black_scholes import BlackScholes

# Test data for European Call option
@pytest.fixture
def european_call_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

@pytest.fixture
def european_put_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')

def test_standard_monte_carlo_call(european_call_option):
    mc_simulator = MonteCarlo(european_call_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.standard()
    price = result['price']
    std_error = result['std_error']
    analytical_price = BlackScholes.price(european_call_option)['price']
    # Check if the analytical price is within 3 standard errors of the MC price
    assert abs(price - analytical_price) < 3 * std_error

def test_standard_monte_carlo_put(european_put_option):
    mc_simulator = MonteCarlo(european_put_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.standard()
    price = result['price']
    std_error = result['std_error']
    analytical_price = BlackScholes.price(european_put_option)['price']
    assert abs(price - analytical_price) < 3 * std_error

def test_antithetic_variates_monte_carlo_call(european_call_option):
    mc_simulator = MonteCarlo(european_call_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.antithetic_variates()
    price = result['price']
    std_error = result['std_error']
    analytical_price = BlackScholes.price(european_call_option)['price']
    assert abs(price - analytical_price) < 3 * std_error

def test_antithetic_variates_monte_carlo_put(european_put_option):
    mc_simulator = MonteCarlo(european_put_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.antithetic_variates()
    price = result['price']
    std_error = result['std_error']
    analytical_price = BlackScholes.price(european_put_option)['price']
    assert abs(price - analytical_price) < 3 * std_error

def test_control_variates_monte_carlo_call(european_call_option):
    analytical_price = BlackScholes.price(european_call_option)['price']
    mc_simulator = MonteCarlo(european_call_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.control_variates(analytical_price)
    price = result['price']
    std_error = result['std_error']
    assert abs(price - analytical_price) < 3 * std_error

def test_control_variates_monte_carlo_put(european_put_option):
    analytical_price = BlackScholes.price(european_put_option)['price']
    mc_simulator = MonteCarlo(european_put_option, n_paths=50000, n_steps=252, seed=42)
    result = mc_simulator.control_variates(analytical_price)
    price = result['price']
    std_error = result['std_error']
    assert abs(price - analytical_price) < 3 * std_error
