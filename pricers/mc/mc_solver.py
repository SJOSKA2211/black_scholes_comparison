"""
Monte Carlo Methods for Option Pricing
Implements standard MC and variance reduction techniques
"""
import numpy as np
from typing import Tuple, Dict, Any
from .._base_pricer import Option, BasePricer # Import BasePricer
import time

class MonteCarlo(BasePricer): # Inherit from BasePricer
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
        super().__init__(option) # Call parent constructor
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = self.option.T / n_steps
        
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
    
    def _standard_price(self) -> Tuple[float, float]:
        """
        Standard Monte Carlo pricing
        Returns price and standard error.
        """
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
        
        return price, std_error
    
    def _antithetic_variates_price(self) -> Tuple[float, float]:
        """
        Monte Carlo with antithetic variates variance reduction
        Returns price and standard error.
        """
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
        
        return price, std_error
    
    def _control_variates_price(self, control_price: float) -> Tuple[float, float]:
        """
        Monte Carlo with control variates variance reduction
        Returns price and standard error.
        """
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
        
        return price, std_error

    def price(self, method_type: str = "standard", control_price: float = None) -> Dict[str, Any]:
        """
        Calculates the option price using the specified Monte Carlo method.
        
        Parameters:
        -----------
        method_type : str
            Type of MC to use ('standard', 'antithetic', 'control_variates').
            Defaults to 'standard'.
        control_price : float, optional
            Required for 'control_variates' method.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the option price, standard error, and computation time.
        """
        start_time = time.time()
        
        if method_type == "standard":
            price_val, std_error_val = self._standard_price()
        elif method_type == "antithetic":
            price_val, std_error_val = self._antithetic_variates_price()
        elif method_type == "control_variates":
            if control_price is None:
                raise ValueError("control_price must be provided for 'control_variates' method.")
            price_val, std_error_val = self._control_variates_price(control_price)
        else:
            raise ValueError(f"Unknown MC method type: {method_type}")
        
        elapsed_time = time.time() - start_time
        return {
            "price": price_val,
            "std_error": std_error_val,
            "computation_time": elapsed_time
        }

    def get_greeks(self, method_type: str = "standard", dS: float = 0.01, dSigma: float = 0.001, dr: float = 0.0001, dT: float = 0.0001) -> Dict[str, float]:
        """
        Calculates option Greeks using finite difference approximation.
        
        Parameters:
        -----------
        method_type : str
            Type of MC to use for pricing when calculating Greeks.
        dS : float
            Small perturbation for stock price (for Delta, Gamma).
        dSigma : float
            Small perturbation for volatility (for Vega).
        dr : float
            Small perturbation for risk-free rate (for Rho).
        dT : float
            Small perturbation for time to maturity (for Theta).
            
        Returns:
        --------
        Dict[str, float]
            A dictionary containing the calculated Greeks.
        """
        original_option = self.option
        
        # Helper to get price for a given option object and method
        def get_price_for_greeks(opt: Option):
            # Temporarily create a new MC solver with the perturbed option
            # It's important to copy n_paths, n_steps, seed to ensure consistent settings
            temp_mc = MonteCarlo(opt, n_paths=self.n_paths, n_steps=self.n_steps, seed=np.random.randint(0, 100000)) # Use a new random seed for each perturbation
            # For control variates, we would need the analytical price of the control for the perturbed option
            # For simplicity, we'll assume standard MC for Greek calculation here, or pass control_price if needed
            return temp_mc.price(method_type)["price"]

        # Original price
        price_original = get_price_for_greeks(original_option)

        # Delta
        option_plus_dS = Option(S=original_option.S + dS, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type)
        option_minus_dS = Option(S=original_option.S - dS, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type)
        price_plus_dS = get_price_for_greeks(option_plus_dS)
        price_minus_dS = get_price_for_greeks(option_minus_dS)
        delta = (price_plus_dS - price_minus_dS) / (2 * dS)

        # Gamma
        gamma = (price_plus_dS - 2 * price_original + price_minus_dS) / (dS ** 2)

        # Vega
        option_plus_dSigma = Option(S=original_option.S, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma + dSigma, option_type=original_option.option_type)
        option_minus_dSigma = Option(S=original_option.S, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma - dSigma, option_type=original_option.option_type)
        price_plus_dSigma = get_price_for_greeks(option_plus_dSigma)
        price_minus_dSigma = get_price_for_greeks(option_minus_dSigma)
        vega = (price_plus_dSigma - price_minus_dSigma) / (2 * dSigma)

        # Theta (perturb T, but ensure T remains positive)
        option_plus_dT = Option(S=original_option.S, K=original_option.K, T=original_option.T + dT, r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type)
        option_minus_dT = Option(S=original_option.S, K=original_option.K, T=max(1e-6, original_option.T - dT), r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type) # Ensure T > 0
        price_plus_dT = get_price_for_greeks(option_plus_dT)
        price_minus_dT = get_price_for_greeks(option_minus_dT)
        theta = -(price_plus_dT - price_minus_dT) / (2 * dT) # Theta is usually negative

        # Rho
        option_plus_dr = Option(S=original_option.S, K=original_option.K, T=original_option.T, r=original_option.r + dr, sigma=original_option.sigma, option_type=original_option.option_type)
        option_minus_dr = Option(S=original_option.S, K=original_option.K, T=original_option.T, r=original_option.r - dr, sigma=original_option.sigma, option_type=original_option.option_type)
        price_plus_dr = get_price_for_greeks(option_plus_dr)
        price_minus_dr = get_price_for_greeks(option_minus_dr)
        rho = (price_plus_dr - price_minus_dr) / (2 * dr)
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }