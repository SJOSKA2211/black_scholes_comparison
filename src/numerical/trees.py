"""
Binomial Tree Methods
Implements CRR binomial trees
"""
import numpy as np
from typing import Tuple
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



