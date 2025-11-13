"""
Binomial Tree Methods
Implements CRR binomial trees
"""
import numpy as np
from typing import Tuple, Dict # Added Dict
from ..options.base import Option
import time

class BinomialTree:
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
        self.option = option
        self.n_steps = n_steps
        self.dt = option.T / n_steps
        
        # Calculate parameters (CRR parametrization)
        self.u = np.exp(option.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(option.r * self.dt) - self.d) / (self.u - self.d)
    
    def european(self) -> Tuple[float, float]:
        """
        Price European option using binomial tree
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self.u ** j) * (self.d ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            V = np.exp(-self.option.r * self.dt) * (self.p * V[1:] + (1 - self.p) * V[:-1])
        
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
    
    def american(self) -> Tuple[float, float]:
        """
        Price American option using binomial tree
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (self.u ** j) * (self.d ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction with early exercise
        for i in range(self.n_steps - 1, -1, -1):
            # Asset prices at time step i
            S = np.array([self.option.S * (self.u ** j) * (self.d ** (i - j))
                          for j in range(i + 1)])
            
            # Continuation value
            continuation = np.exp(-self.option.r * self.dt) * \
                          (self.p * V[1:] + (1 - self.p) * V[:-1])
            
            # Intrinsic value
            if self.option.option_type == 'call':
                intrinsic = np.maximum(S - self.option.K, 0)
            else:
                intrinsic = np.maximum(self.option.K - S, 0)
            
            # Take maximum (early exercise decision)
            V = np.maximum(continuation, intrinsic)
        
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
    
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
    
    def jarrow_rudd_european(self) -> Tuple[float, float]:
        """
        Price European option using Jarrow-Rudd binomial tree
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Jarrow-Rudd parameters
        jr_u = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt + self.option.sigma * np.sqrt(self.dt))
        jr_d = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt - self.option.sigma * np.sqrt(self.dt))
        jr_p = 0.5
        
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (jr_u ** j) * (jr_d ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            V = np.exp(-self.option.r * self.dt) * (jr_p * V[1:] + (1 - jr_p) * V[:-1])
        
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
    
    def jarrow_rudd_american(self) -> Tuple[float, float]:
        """
        Price American option using Jarrow-Rudd binomial tree with early exercise
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Jarrow-Rudd parameters
        jr_u = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt + self.option.sigma * np.sqrt(self.dt))
        jr_d = np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.dt - self.option.sigma * np.sqrt(self.dt))
        jr_p = 0.5
        
        # Initialize asset prices at maturity
        ST = np.array([self.option.S * (jr_u ** j) * (jr_d ** (self.n_steps - j))
                       for j in range(self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction with early exercise
        for i in range(self.n_steps - 1, -1, -1):
            # Asset prices at time step i
            S = np.array([self.option.S * (jr_u ** j) * (jr_d ** (i - j))
                          for j in range(i + 1)])
            
            # Continuation value
            continuation = np.exp(-self.option.r * self.dt) * \
                          (jr_p * V[1:] + (1 - jr_p) * V[:-1])
            
            # Intrinsic value
            if self.option.option_type == 'call':
                intrinsic = np.maximum(S - self.option.K, 0)
            else:
                intrinsic = np.maximum(self.option.K - S, 0)
            
            # Take maximum (early exercise decision)
            V = np.maximum(continuation, intrinsic)
        
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
    
    def calculate_greeks(self, american: bool = False, dS: float = 0.01, dSigma: float = 0.001, dr: float = 0.0001, dT: float = 0.0001) -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference approximation.
        
        Parameters:
        -----------
        american : bool
            Whether to calculate Greeks for an American option.
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
        
        # Helper to get price for a given option object
        def get_price(opt: Option):
            # Re-initialize BinomialTree with the new option parameters
            temp_bt = BinomialTree(opt, self.n_steps)
            if american:
                return temp_bt.american()[0]
            else:
                return temp_bt.european()[0]

        # Original price
        original_price = get_price(original_option)

        # Delta
        option_plus_dS = Option(S=original_option.S + dS, **{k: v for k, v in original_option.__dict__.items() if k != 'S'})
        option_minus_dS = Option(S=original_option.S - dS, **{k: v for k, v in original_option.__dict__.items() if k != 'S'})
        price_plus_dS = get_price(option_plus_dS)
        price_minus_dS = get_price(option_minus_dS)
        delta = (price_plus_dS - price_minus_dS) / (2 * dS)

        # Gamma
        price_original_S = original_price # Already calculated
        gamma = (price_plus_dS - 2 * price_original_S + price_minus_dS) / (dS ** 2)

        # Vega
        option_plus_dSigma = Option(sigma=original_option.sigma + dSigma, **{k: v for k, v in original_option.__dict__.items() if k != 'sigma'})
        option_minus_dSigma = Option(sigma=original_option.sigma - dSigma, **{k: v for k, v in original_option.__dict__.items() if k != 'sigma'})
        price_plus_dSigma = get_price(option_plus_dSigma)
        price_minus_dSigma = get_price(option_minus_dSigma)
        vega = (price_plus_dSigma - price_minus_dSigma) / (2 * dSigma)

        # Theta (perturb T, but ensure T remains positive)
        option_plus_dT = Option(T=original_option.T + dT, **{k: v for k, v in original_option.__dict__.items() if k != 'T'})
        option_minus_dT = Option(T=max(1e-6, original_option.T - dT), **{k: v for k, v in original_option.__dict__.items() if k != 'T'}) # Ensure T > 0
        price_plus_dT = get_price(option_plus_dT)
        price_minus_dT = get_price(option_minus_dT)
        theta = -(price_plus_dT - price_minus_dT) / (2 * dT) # Theta is usually negative

        # Rho
        option_plus_dr = Option(r=original_option.r + dr, **{k: v for k, v in original_option.__dict__.items() if k != 'r'})
        option_minus_dr = Option(r=original_option.r - dr, **{k: v for k, v in original_option.__dict__.items() if k != 'r'})
        price_plus_dr = get_price(option_plus_dr)
        price_minus_dr = get_price(option_minus_dr)
        rho = (price_plus_dr - price_minus_dr) / (2 * dr)
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }