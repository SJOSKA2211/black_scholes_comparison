"""
Test cases for Finite Difference Methods
"""
import pytest
from src.options.base import Option
from src.numerical.finite_difference import FiniteDifference
from src.analytical.black_scholes import BlackScholes

# Test data for European Call option
@pytest.fixture
def european_call_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

@pytest.fixture
def european_put_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')

def test_explicit_finite_difference_call(european_call_option):
    fd_solver = FiniteDifference(european_call_option, M=100, N=100)
    price, _ = fd_solver.explicit()
    analytical_price = BlackScholes.price(european_call_option)
    assert abs(price - analytical_price) < 0.5  # Explicit is less accurate, so tolerance is higher

def test_explicit_finite_difference_put(european_put_option):
    fd_solver = FiniteDifference(european_put_option, M=100, N=100)
    price, _ = fd_solver.explicit()
    analytical_price = BlackScholes.price(european_put_option)
    assert abs(price - analytical_price) < 0.5

def test_implicit_finite_difference_call(european_call_option):
    fd_solver = FiniteDifference(european_call_option, M=100, N=100)
    price, _ = fd_solver.implicit()
    analytical_price = BlackScholes.price(european_call_option)
    assert abs(price - analytical_price) < 0.1

def test_implicit_finite_difference_put(european_put_option):
    fd_solver = FiniteDifference(european_put_option, M=100, N=100)
    price, _ = fd_solver.implicit()
    analytical_price = BlackScholes.price(european_put_option)
    assert abs(price - analytical_price) < 0.1

def test_crank_nicolson_finite_difference_call(european_call_option):
    fd_solver = FiniteDifference(european_call_option, M=100, N=100)
    price, _ = fd_solver.crank_nicolson()
    analytical_price = BlackScholes.price(european_call_option)
    assert abs(price - analytical_price) < 0.05

def test_crank_nicolson_finite_difference_put(european_put_option):
    fd_solver = FiniteDifference(european_put_option, M=100, N=100)
    price, _ = fd_solver.crank_nicolson()
    analytical_price = BlackScholes.price(european_put_option)
    assert abs(price - analytical_price) < 0.05
