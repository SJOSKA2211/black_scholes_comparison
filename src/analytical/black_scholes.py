"""
Analytical Black-Scholes Formulas
Provides closed-form solutions for European options
"""
import numpy as np
from scipy.stats import norm
from ..options.base import Option

class BlackScholes:
    """Analytical Black-Scholes option pricing"""
    
    @staticmethod
    def d1(option: Option) -> float:
        """Calculate d1 parameter"""
        return (np.log(option.S / option.K) + 
                (option.r + 0.5 * option.sigma**2) * option.T) / 
               (option.sigma * np.sqrt(option.T))
    
    @staticmethod
    def d2(option: Option) -> float:
        """Calculate d2 parameter"""
        d1_val = BlackScholes.d1(option)
        return d1_val - option.sigma * np.sqrt(option.T)
    
    @staticmethod
    def call_price(option: Option) -> float:
        """Calculate European call option price"""
        d1_val = BlackScholes.d1(option)
        d2_val = BlackScholes.d2(option)
        
        price = (option.S * norm.cdf(d1_val) - 
                 option.K * np.exp(-option.r * option.T) * norm.cdf(d2_val))
        return price
    
    @staticmethod
    def put_price(option: Option) -> float:
        """Calculate European put option price"""
        d1_val = BlackScholes.d1(option)
        d2_val = BlackScholes.d2(option)
        
        price = (option.K * np.exp(-option.r * option.T) * norm.cdf(-d2_val) - 
                 option.S * norm.cdf(-d1_val))
        return price
    
    @staticmethod
    def price(option: Option) -> float:
        """Calculate option price based on type"""
        if option.option_type == 'call':
            return BlackScholes.call_price(option)
        else:
            return BlackScholes.put_price(option)
    
    @staticmethod
    def delta(option: Option) -> float:
        """Calculate option delta"""
        d1_val = BlackScholes.d1(option)
        if option.option_type == 'call':
            return norm.cdf(d1_val)
        else:
            return norm.cdf(d1_val) - 1
    
    @staticmethod
    def gamma(option: Option) -> float:
        """Calculate option gamma"""
        d1_val = BlackScholes.d1(option)
        return norm.pdf(d1_val) / (option.S * option.sigma * np.sqrt(option.T))
    
    @staticmethod
    def vega(option: Option) -> float:
        """Calculate option vega"""
        d1_val = BlackScholes.d1(option)
        return option.S * norm.pdf(d1_val) * np.sqrt(option.T)
    
    @staticmethod
    def theta(option: Option) -> float:
        """Calculate option theta"""
        d1_val = BlackScholes.d1(option)
        d2_val = BlackScholes.d2(option)
        
        term1 = -(option.S * norm.pdf(d1_val) * option.sigma) / (2 * np.sqrt(option.T))
        
        if option.option_type == 'call':
            term2 = -option.r * option.K * np.exp(-option.r * option.T) * norm.cdf(d2_val)
            return term1 + term2
        else:
            term2 = option.r * option.K * np.exp(-option.r * option.T) * norm.cdf(-d2_val)
            return term1 + term2
