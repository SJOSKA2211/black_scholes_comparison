"""
Test cases for Binomial and Trinomial Tree Methods
"""
import pytest
from src.options.base import Option
from src.numerical.trees import BinomialTree
from src.analytical.black_scholes import BlackScholes

# Test data for European Call and Put options
@pytest.fixture
def european_call_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

@pytest.fixture
def european_put_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')

# Test data for American Call and Put options (using European parameters for now, as American analytical is complex)
@pytest.fixture
def american_call_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

@pytest.fixture
def american_put_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')

def test_binomial_european_call(european_call_option):
    bt = BinomialTree(european_call_option, n_steps=500)
    price, _ = bt.european()
    analytical_price = BlackScholes.price(european_call_option)['price']
    assert abs(price - analytical_price) < 0.1

def test_binomial_european_put(european_put_option):
    bt = BinomialTree(european_put_option, n_steps=500)
    price, _ = bt.european()
    analytical_price = BlackScholes.price(european_put_option)['price']
    assert abs(price - analytical_price) < 0.1

def test_binomial_american_call(american_call_option):
    # For American calls on non-dividend paying stocks, early exercise is never optimal.
    # So, American call price should be very close to European call price.
    bt = BinomialTree(american_call_option, n_steps=500)
    price, _ = bt.american()
    analytical_european_price = BlackScholes.price(american_call_option)['price']
    assert abs(price - analytical_european_price) < 0.1

def test_binomial_american_put(american_put_option):
    # American puts can have early exercise value.
    # The price should be greater than or equal to the European put price.
    bt = BinomialTree(american_put_option, n_steps=500)
    price, _ = bt.american()
    analytical_european_price = BlackScholes.price(american_put_option)['price']
    assert price >= analytical_european_price
    # A reasonable upper bound for the difference for this specific case
    assert abs(price - analytical_european_price) < 1.0 # American put price is higher


