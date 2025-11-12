"""
Monte Carlo Methods for Option Pricing
Implements standard MC and variance reduction techniques
"""
import numpy as np
from typing import Tuple
from ..options.base import Option
import time

class MonteCarlo:
    """Monte Carlo simulator for option pricing"""
    
    def __init__(self, option: Option, n_paths: int = 10000, 
                 n_steps: int = 100, seed: int = None):
        """
        Initialize Monte Carlo simulator
        
        Parameters:
        -----------
        option : Option
            Option to price
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
        seed : int
            Random seed for reproducibility
        """
        self.option = option
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = option.T / n_steps
        
        if seed is not None:
            np.random.seed(seed)
    
    def _generate_paths(self, antithetic: bool = False) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion
        
        Parameters:
        -----------
        antithetic : bool
            Use antithetic variates
            
        Returns:
        --------
        paths : ndarray
            Stock price paths (n_paths x n_steps+1)
        """
        if antithetic:
            # Generate half the paths
            Z = np.random.standard_normal((self.n_paths // 2, self.n_steps))
            # Create antithetic pairs
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.standard_normal((self.n_paths, self.n_steps))
        
        # Initialize paths
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.option.S
        
        # Generate paths
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.option.r - 0.5 * self.option.sigma**2) * self.dt +
                self.option.sigma * np.sqrt(self.dt) * Z[:, t-1]
            )
        
        return paths
    
    def standard(self) -> Tuple[float, float, float]:
        """
        Standard Monte Carlo pricing
        
        Returns:
        --------
        price : float
            Option price
        std_error : float
            Standard error of estimate
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Generate paths
        paths = self._generate_paths()
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        if self.option.option_type == 'call':
            payoffs = np.maximum(final_prices - self.option.K, 0)
        else:
            payoffs = np.maximum(self.option.K - final_prices, 0)
        
        # Discount to present value
        price = np.exp(-self.option.r * self.option.T) * np.mean(payoffs)
        std_error = np.exp(-self.option.r * self.option.T) * np.std(payoffs) / \
                    np.sqrt(self.n_paths)
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "std_error": std_error,
            "computation_time": elapsed_time
        }
    
    def antithetic_variates(self) -> Tuple[float, float, float]:
        """
        Monte Carlo with antithetic variates variance reduction
        
        Returns:
        --------
        price : float
            Option price
        std_error : float
            Standard error of estimate
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Generate paths with antithetic variates
        paths = self._generate_paths(antithetic=True)
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        if self.option.option_type == 'call':
            payoffs = np.maximum(final_prices - self.option.K, 0)
        else:
            payoffs = np.maximum(self.option.K - final_prices, 0)
        
        # Discount to present value
        price = np.exp(-self.option.r * self.option.T) * np.mean(payoffs)
        std_error = np.exp(-self.option.r * self.option.T) * np.std(payoffs) / \
                    np.sqrt(self.n_paths)
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "std_error": std_error,
            "computation_time": elapsed_time
        }
    
    def control_variates(self, control_price: float) -> Tuple[float, float, float]:
        """
        Monte Carlo with control variates variance reduction
        
        Parameters:
        -----------
        control_price : float
            Analytical price of control variate (e.g., from Black-Scholes)
            
        Returns:
        --------
        price : float
            Option price
        std_error : float
            Standard error of estimate
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Generate paths
        paths = self._generate_paths()
        
        # Calculate payoffs for target and control
        final_prices = paths[:, -1]
        if self.option.option_type == 'call':
            target_payoffs = np.maximum(final_prices - self.option.K, 0)
            control_payoffs = np.maximum(final_prices - self.option.K, 0)  # Same for European
        else:
            target_payoffs = np.maximum(self.option.K - final_prices, 0)
            control_payoffs = np.maximum(self.option.K - final_prices, 0)
        
        # Discount payoffs
        target_pv = np.exp(-self.option.r * self.option.T) * target_payoffs
        control_pv = np.exp(-self.option.r * self.option.T) * control_payoffs
        
        # Calculate optimal c
        cov = np.cov(target_pv, control_pv)[0, 1]
        var = np.var(control_pv)
        c = cov / var if var > 0 else 0
        
        # Apply control variate
        adjusted_payoffs = target_pv + c * (control_price - control_pv)
        
        price = np.mean(adjusted_payoffs)
        std_error = np.std(adjusted_payoffs) / np.sqrt(self.n_paths)
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "std_error": std_error,
            "computation_time": elapsed_time
        }
