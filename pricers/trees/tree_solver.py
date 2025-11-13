"""
Binomial Tree Methods
Implements CRR and Jarrow-Rudd binomial trees
"""
import numpy as np
from typing import Tuple, Dict, Any
from .._base_pricer import Option, BasePricer # Import BasePricer
import time
from utils.decorators import time_it, measure_memory # Import decorators

class BinomialTree(BasePricer): # Inherit from BasePricer
    """Binomial tree method for option pricing"""
    
    def __init__(self, option: Option, n_steps: int = 100):
        """
        Initialize binomial tree
        
        Parameters:
        -----------
        option : Option
            Option to price
        n_steps : int
            Number of time steps
        """
        super().__init__(option) # Call parent constructor
        self.n_steps = n_steps
        self.dt = self.option.T / n_steps
        
        # CRR parameters
        self._u_crr = np.exp(self.option.sigma * np.sqrt(self.dt))
        self._d_crr = 1 / self._u_crr
        self._p_crr = (np.exp(self.option.r * self.dt) - self._d_crr) / (self._u_crr - self._d_crr)
        
        # Jarrow-Rudd parameters
        self._u_jr = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt + self.option.sigma * np.sqrt(self.dt))
        self._d_jr = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt - self.option.sigma * np.sqrt(self.dt))
        self._p_jr = 0.5
    
    def _european_crr_price(self) -> float:
        """
        Price European option using CRR binomial tree
        Returns only the price.
        """
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self._u_crr ** j) * (self._d_crr ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            V = np.exp(-self.option.r * self.dt) * (self._p_crr * V[1:] + (1 - self._p_crr) * V[:-1])
        
        return V[0]
    
    def _american_crr_price(self) -> float:
        """
        Price American option using CRR binomial tree
        Returns only the price.
        """
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self._u_crr ** j) * (self._d_crr ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction with early exercise
        for i in range(self.n_steps - 1, -1, -1):
            # Asset prices at time step i
            S = np.array([self.option.S * (self._u_crr ** j) * (self._d_crr ** (i - j))
                          for j in range(i + 1)])
            
            # Continuation value
            continuation = np.exp(-self.option.r * self.dt) * \
                          (self._p_crr * V[1:] + (1 - self._p_crr) * V[:-1])
            
            # Intrinsic value
            if self.option.option_type == 'call':
                intrinsic = np.maximum(S - self.option.K, 0)
            else:
                intrinsic = np.maximum(self.option.K - S, 0)
            
            # Take maximum (early exercise decision)
            V = np.maximum(continuation, intrinsic)
        
        return V[0]
    
    def _european_jr_price(self) -> float:
        """
        Price European option using Jarrow-Rudd binomial tree
        Returns only the price.
        """
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self._u_jr ** j) * (self._d_jr ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            V = np.exp(-self.option.r * self.dt) * (self._p_jr * V[1:] + (1 - self._p_jr) * V[:-1])
        
        return V[0]
    
    def _american_jr_price(self) -> float:
        """
        Price American option using Jarrow-Rudd binomial tree with early exercise
        Returns only the price.
        """
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self._u_jr ** j) * (self._d_jr ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction with early exercise
        for i in range(self.n_steps - 1, -1, -1):
            # Asset prices at time step i
            S = np.array([self.option.S * (self._u_jr ** j) * (self._d_jr ** (i - j))
                          for j in range(i + 1)])
            
            # Continuation value
            continuation = np.exp(-self.option.r * self.dt) * \
                          (self._p_jr * V[1:] + (1 - self._p_jr) * V[:-1])
            
            # Intrinsic value
            if self.option.option_type == 'call':
                intrinsic = np.maximum(S - self.option.K, 0)
            else:
                intrinsic = np.maximum(self.option.K - S, 0)
            
            # Take maximum (early exercise decision)
            V = np.maximum(continuation, intrinsic)
        
        return V[0]
    
    @time_it # Apply decorator
    @measure_memory # Apply decorator
    def price(self, method_type: str = "european_crr") -> Dict[str, Any]:
        """
        Calculates the option price using the specified binomial tree method.
        
        Parameters:
        -----------
        method_type : str
            Type of binomial tree to use ('european_crr', 'american_crr', 'european_jr', 'american_jr').
            Defaults to 'european_crr'.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the option price and computation time.
        """
        # Removed start_time and elapsed_time, handled by decorator
        
        if method_type == "european_crr":
            price_val = self._european_crr_price()
        elif method_type == "american_crr":
            price_val = self._american_crr_price()
        elif method_type == "european_jr":
            price_val = self._european_jr_price()
        elif method_type == "american_jr":
            price_val = self._american_jr_price()
        else:
            raise ValueError(f"Unknown Binomial Tree method type: {method_type}")
        
        return {
            "price": price_val,
            # "computation_time": elapsed_time # Handled by decorator
        }

    @time_it # Apply decorator
    @measure_memory # Apply decorator
    def get_greeks(self, method_type: str = "european_crr", dS: float = 0.01, dSigma: float = 0.001, dr: float = 0.0001, dT: float = 0.0001) -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference approximation.
        
        Parameters:
        -----------
        method_type : str
            Type of Binomial Tree to use for pricing when calculating Greeks.
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
            # Re-initialize BinomialTree with the new option parameters
            temp_bt = BinomialTree(opt, self.n_steps)
            return temp_bt.price(method_type)("price")

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
