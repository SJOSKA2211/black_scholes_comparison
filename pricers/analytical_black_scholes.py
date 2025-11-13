"""
Analytical Black-Scholes Formulas
Provides closed-form solutions for European options
"""
import numpy as np
from scipy.stats import norm
from ._base_pricer import Option, BasePricer # Import BasePricer
import time # Import time for computation_time
from typing import Dict, Any # Import Dict, Any for type hints

class BlackScholes(BasePricer): # Inherit from BasePricer
    """Analytical Black-Scholes option pricing"""

    def __init__(self, option: Option):
        super().__init__(option) # Call parent constructor
    
    def _d1(self) -> float: # Changed to instance method
        """Calculate d1 parameter"""
        return (np.log(self.option.S / self.option.K) + (self.option.r + 0.5 * self.option.sigma**2) * self.option.T) / (self.option.sigma * np.sqrt(self.option.T))
    
    def _d2(self) -> float: # Changed to instance method
        """Calculate d2 parameter"""
        d1_val = self._d1()
        return d1_val - self.option.sigma * np.sqrt(self.option.T)
    
    def _call_price(self) -> float: # Changed to instance method
        """Calculate European call option price"""
        d1_val = self._d1()
        d2_val = self._d2()
        
        price = (self.option.S * norm.cdf(d1_val) - 
                 self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(d2_val))
        return price
    
    def _put_price(self) -> float: # Changed to instance method
        """Calculate European put option price"""
        d1_val = self._d1()
        d2_val = self._d2()
        
        price = (self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(-d2_val) - 
                 self.option.S * norm.cdf(-d1_val))
        return price
    
    def price(self) -> Dict[str, Any]: # Implements abstract method
        """
        Calculates the option price and returns a dictionary of results.
        The dictionary should at least contain 'price' and 'computation_time'.
        """
        start_time = time.time()
        if self.option.option_type == 'call':
            price_val = self._call_price()
        else:
            price_val = self._put_price()
        elapsed_time = time.time() - start_time
        
        return {
            "price": price_val,
            "computation_time": elapsed_time
        }
    
    def get_greeks(self) -> Dict[str, float]: # Implements abstract method
        """
        Calculates the option Greeks (Delta, Gamma, Vega, Theta, Rho)
        and returns them as a dictionary.
        """
        # Re-using existing static methods, but now calling them with self.option
        delta_val = self._delta()
        gamma_val = self._gamma()
        vega_val = self._vega()
        theta_val = self._theta()
        rho_val = self._rho()
        
        return {
            "delta": delta_val,
            "gamma": gamma_val,
            "vega": vega_val,
            "theta": theta_val,
            "rho": rho_val
        }
    
    def _delta(self) -> float: # Changed to instance method
        """Calculate option delta"""
        d1_val = self._d1()
        if self.option.option_type == 'call':
            return norm.cdf(d1_val)
        else:
            return norm.cdf(d1_val) - 1
    
    def _gamma(self) -> float: # Changed to instance method
        """Calculate option gamma"""
        d1_val = self._d1()
        return norm.pdf(d1_val) / (self.option.S * self.option.sigma * np.sqrt(self.option.T))
    
    def _vega(self) -> float: # Changed to instance method
        """Calculate option vega"""
        d1_val = self._d1()
        return self.option.S * norm.pdf(d1_val) * np.sqrt(self.option.T)
    
    def _theta(self) -> float: # Changed to instance method
        """Calculate option theta"""
        d1_val = self._d1()
        d2_val = self._d2()
        
        term1 = -(self.option.S * norm.pdf(d1_val) * self.option.sigma) / (2 * self.option.T)
        
        if self.option.option_type == 'call':
            term2 = -self.option.r * self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(d2_val)
            return term1 + term2
        else:
            term2 = self.option.r * self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(-d2_val)
            return term1 + term2
    
    def _rho(self) -> float: # Changed to instance method
        """Calculate option rho"""
        d2_val = self._d2()
        if self.option.option_type == 'call':
            return self.option.K * self.option.T * np.exp(-self.option.r * self.option.T) * norm.cdf(d2_val)
        else:
            return -self.option.K * self.option.T * np.exp(-self.option.r * self.option.T) * norm.cdf(-d2_val)