import QuantLib as ql
import datetime

# Option parameters
S = 100.0
K = 100.0
T = 1.0  # Time to maturity in years
r = 0.05 # Risk-free rate
sigma = 0.2 # Volatility

# Set up market data
calculation_date = ql.Date(1, 1, 2025)
ql.Settings.instance().evaluationDate = calculation_date

# Set up option
maturity_date = calculation_date + ql.Period(int(T * 365), ql.Days)
exercise = ql.EuropeanExercise(maturity_date)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
option = ql.VanillaOption(payoff, exercise)

# Set up Black-Scholes process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, r, ql.Actual365Fixed()))
zero_dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, 0.0, ql.Actual365Fixed())) # Added
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
bsm_process = ql.BlackScholesMertonProcess(spot_handle, zero_dividend_ts, flat_ts, flat_vol_ts)

# Calculate Greeks
engine = ql.AnalyticEuropeanEngine(bsm_process)
option.setPricingEngine(engine)

# Price
price = option.NPV()
delta = option.delta()
gamma = option.gamma()
vega = option.vega() / 100.0 # QuantLib vega is per 1% change in volatility
theta = option.theta() / 365.0 # QuantLib theta is per day
rho = option.rho() # QuantLib rho is per 1% change in interest rate, so no need to divide by 100.0 here.


print(f"Price: {price}")
print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Vega: {vega}")
print(f"Theta: {theta}")
print(f"Rho: {rho}")
