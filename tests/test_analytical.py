"""
Test cases for Analytical Black-Scholes Formulas
"""
import pytest
from src.options.base import Option
from src.analytical.black_scholes import BlackScholes

# Test data for European Call and Put options
@pytest.fixture
def european_call_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

@pytest.fixture
def european_put_option():
    return Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')

# Expected values (can be calculated using online calculators or other reliable sources)
# For S=100, K=100, T=1, r=0.05, sigma=0.2
# Call Price: 10.450583572185526
# Put Price: 5.57352602225657
# Delta Call: 0.636831
# Delta Put: -0.363169
# Gamma: 0.01875
# Vega: 37.500
# Theta Call: -6.410
# Theta Put: -1.533

def test_d1_d2(european_call_option):
    d1 = BlackScholes.d1(european_call_option)
    d2 = BlackScholes.d2(european_call_option)
    assert abs(d1 - 0.35) < 1e-6
    assert abs(d2 - 0.15) < 1e-6

def test_call_price(european_call_option):
    price = BlackScholes.call_price(european_call_option)
    assert abs(price - 10.450583572185526) < 1e-6

def test_put_price(european_put_option):
    price = BlackScholes.put_price(european_put_option)
    assert abs(price - 5.57352602225657) < 1e-6

def test_price_method(european_call_option, european_put_option):
    call_price = BlackScholes.price(european_call_option)
    put_price = BlackScholes.price(european_put_option)
    assert abs(call_price - 10.450583572185526) < 1e-6
    assert abs(put_price - 5.57352602225657) < 1e-6

def test_delta(european_call_option, european_put_option):
    delta_call = BlackScholes.delta(european_call_option)
    delta_put = BlackScholes.delta(european_put_option)
    assert abs(delta_call - 0.636831) < 1e-6
    assert abs(delta_put - (-0.363169)) < 1e-6

def test_gamma(european_call_option):
    gamma = BlackScholes.gamma(european_call_option)
    assert abs(gamma - 0.01875) < 1e-6

def test_vega(european_call_option):
    vega = BlackScholes.vega(european_call_option)
    assert abs(vega - 37.500) < 1e-3 # Vega is often quoted per 1% change, so 37.5 is for 100% change.

def test_theta(european_call_option, european_put_option):
    theta_call = BlackScholes.theta(european_call_option)
    theta_put = BlackScholes.theta(european_put_option)
    assert abs(theta_call - (-6.410)) < 1e-3
    assert abs(theta_put - (-1.533)) < 1e-3

def test_option_validation():
    with pytest.raises(ValueError, match="Stock price must be positive"):
        Option(S=0, K=100, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Strike price must be positive"):
        Option(S=100, K=0, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Time to maturity must be positive"):
        Option(S=100, K=100, T=0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Volatility must be positive"):
        Option(S=100, K=100, T=1, r=0.05, sigma=0)
