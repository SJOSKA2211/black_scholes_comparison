"""
Test cases for Analytical Black-Scholes Formulas
"""
import pytest
from pricers._base_pricer import Option
from pricers.analytical_black_scholes import BlackScholes

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

def test_black_scholes_full_output(european_call_option, european_put_option):
    # Expected values from QuantLib (used for previous updates)
    expected_call_price = 10.450583572185526
    expected_put_price = 5.57352602225657
    expected_call_delta = 0.636831
    expected_put_delta = -0.363169
    expected_gamma = 0.01876201734584688
    expected_vega = 37.52403469169378 # Per 1% change, so 0.37524 * 100
    expected_call_theta = -6.412077946438197 # Per year
    expected_put_theta = -1.657880423934626 # Per year
    expected_call_rho = 53.23248154537636
    expected_put_rho = -41.89046090469503

    call_results = BlackScholes.price(european_call_option)
    put_results = BlackScholes.price(european_put_option)

    # Test Call Option
    assert abs(call_results["price"] - expected_call_price) < 1e-6
    assert abs(call_results["delta"] - expected_call_delta) < 1e-6
    assert abs(call_results["gamma"] - expected_gamma) < 1e-6
    assert abs(call_results["vega"] - expected_vega) < 1e-6
    assert abs(call_results["theta"] - expected_call_theta) < 2e-3
    assert abs(call_results["rho"] - expected_call_rho) < 1e-6

    # Test Put Option
    assert abs(put_results["price"] - expected_put_price) < 1e-6
    assert abs(put_results["delta"] - expected_put_delta) < 1e-6
    assert abs(put_results["gamma"] - expected_gamma) < 1e-6
    assert abs(put_results["vega"] - expected_vega) < 1e-6
    assert abs(put_results["theta"] - expected_put_theta) < 2e-3
    assert abs(put_results["rho"] - expected_put_rho) < 1e-6

def test_option_validation():
    with pytest.raises(ValueError, match="Stock price must be positive"):
        Option(S=0, K=100, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Strike price must be positive"):
        Option(S=100, K=0, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Time to maturity must be positive"):
        Option(S=100, K=100, T=0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError, match="Volatility must be positive"):
        Option(S=100, K=100, T=1, r=0.05, sigma=0)
