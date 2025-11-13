"""
Base Option Class
Defines the fundamental option parameters and structure
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class Option:
    """Base class for all option types"""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to maturity (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility
    dividend_yield: float = 0.0 # Dividend yield
    option_type: Literal['call', 'put'] = 'call'
    
    def __post_init__(self):
        """Validate option parameters"""
        if self.S <= 0:
            raise ValueError("Stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (S/K)"""
        return self.S / self.K
    
    def __repr__(self) -> str:
        return (f"Option(S={self.S}, K={self.K}, T={self.T}, "
                f"r={self.r}, sigma={self.sigma}, dividend_yield={self.dividend_yield}, type={self.option_type})")
