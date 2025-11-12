"""
Binomial and Trinomial Tree Methods
Implements CRR binomial trees and trinomial trees
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


class TrinomialTree:
    """Trinomial tree method for option pricing"""
    
    def __init__(self, option: Option, n_steps: int = 100):
        """
        Initialize trinomial tree
        
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
        
        # Calculate parameters
        self.u = np.exp(option.sigma * np.sqrt(2 * self.dt))
        self.d = 1 / self.u
        self.m = 1.0
        
        # Risk-neutral probabilities
        dx = option.sigma * np.sqrt(self.dt)
        nu = option.r - 0.5 * option.sigma**2
        
        self.pu = 0.5 * ((option.sigma**2 * self.dt + nu**2 * self.dt**2) / dx**2 + 
                         nu * self.dt / dx)
        self.pd = 0.5 * ((option.sigma**2 * self.dt + nu**2 * self.dt**2) / dx**2 - 
                         nu * self.dt / dx)
        self.pm = 1 - self.pu - self.pd
    
    def european(self) -> Tuple[float, float]:
        """
        Price European option using trinomial tree
        
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
                       for j in range(-self.n_steps, self.n_steps + 1)])
        
        # Calculate payoffs at maturity
        if self.option.option_type == 'call':
            V = np.maximum(ST - self.option.K, 0)
        else:
            V = np.maximum(self.option.K - ST, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            V = np.exp(-self.option.r * self.dt) * \
                (self.pu * V[2:] + self.pm * V[1:-1] + self.pd * V[:-2])
        
        price = V[0]
        elapsed_time = time.time() - start_time
        return price, elapsed_time
