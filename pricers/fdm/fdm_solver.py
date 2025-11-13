"""
Finite Difference Methods for Option Pricing
Implements explicit, implicit, and Crank-Nicolson schemes
"""
import numpy as np
from typing import Tuple
from .._base_pricer import Option
import time

class FiniteDifference:
    """Finite difference solver for Black-Scholes PDE"""
    
    def __init__(self, option: Option, M: int = 100, N: int = 100,
                 S_max: float = None):
        """
        Initialize finite difference grid
        
        Parameters:
        ----------
        option : Option
            Option to price
        M : int
            Number of spatial grid points
        N : int
            Number of time steps
        S_max : float
            Maximum stock price for grid (default: 3*K)
        """
        self.option = option
        self.M = M
        self.N = N
        self.S_max = S_max or 3 * option.K
        
        # Create grids
        self.dt = option.T / N
        self.dS = self.S_max / M
        
        self.time_grid = np.linspace(0, option.T, N + 1)
        self.price_grid = np.linspace(0, self.S_max, M + 1)
        
        # Initialize solution grid
        self.V = np.zeros((M + 1, N + 1))
        
    def _initial_condition(self):
        """Set initial condition (payoff at maturity)"""
        if self.option.option_type == 'call':
            self.V[:, -1] = np.maximum(self.price_grid - self.option.K, 0)
        else:
            self.V[:, -1] = np.maximum(self.option.K - self.price_grid, 0)
    
    def _boundary_conditions(self, j: int):
        """Set boundary conditions at time step j"""
        if self.option.option_type == 'call':
            self.V[0, j] = 0  # Call worth 0 at S=0
            self.V[-1, j] = self.S_max - self.option.K * \
                            np.exp(-self.option.r * (self.option.T - j * self.dt))
        else:
            self.V[0, j] = self.option.K * \
                           np.exp(-self.option.r * (self.option.T - j * self.dt))
            self.V[-1, j] = 0  # Put worth 0 at S=inf
    
    def explicit(self) -> Tuple[float, float]:
        """
        Explicit finite difference scheme (Forward Euler)
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Set initial and boundary conditions
        self._initial_condition()
        
        # Calculate coefficients
        alpha = 0.5 * self.dt * (self.option.sigma**2 * np.arange(self.M + 1)**2 - 
                                  self.option.r * np.arange(self.M + 1))
        beta = 1 - self.dt * (self.option.sigma**2 * np.arange(self.M + 1)**2 + 
                               self.option.r)
        gamma = 0.5 * self.dt * (self.option.sigma**2 * np.arange(self.M + 1)**2 + 
                                  self.option.r * np.arange(self.M + 1))
        
        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._boundary_conditions(j)
            
            for i in range(1, self.M):
                self.V[i, j] = (alpha[i] * self.V[i-1, j+1] + 
                                beta[i] * self.V[i, j+1] + 
                                gamma[i] * self.V[i+1, j+1])
        
        # Interpolate to get price at S_0
        price = np.interp(self.option.S, self.price_grid, self.V[:, 0])
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "computation_time": elapsed_time
        }
    
    def implicit(self) -> Tuple[float, float]:
        """
        Implicit finite difference scheme (Backward Euler)
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Set initial and boundary conditions
        self._initial_condition()
        
        # Build tridiagonal matrix
        alpha = -0.5 * self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 - 
                                   self.option.r * np.arange(1, self.M))
        beta = 1 + self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 + 
                              self.option.r)
        gamma = -0.5 * self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 + 
                                   self.option.r * np.arange(1, self.M))
        
        # Create tridiagonal matrix
        A = np.diag(beta) + np.diag(gamma[:-1], 1) + np.diag(alpha[1:], -1)
        
        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._boundary_conditions(j)
            
            # Right-hand side
            b = self.V[1:-1, j+1].copy()
            b[0] -= alpha[0] * self.V[0, j]
            b[-1] -= gamma[-1] * self.V[-1, j]
            
            # Solve linear system
            self.V[1:-1, j] = np.linalg.solve(A, b)
        
        # Interpolate to get price at S_0
        price = np.interp(self.option.S, self.price_grid, self.V[:, 0])
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "computation_time": elapsed_time
        }
    
    def crank_nicolson(self) -> Tuple[float, float]:
        """
        Crank-Nicolson finite difference scheme
        
        Returns:
        --------
        price : float
            Option price
        time : float
            Computation time
        """
        start_time = time.time()
        
        # Set initial and boundary conditions
        self._initial_condition()
        
        # Coefficients for Crank-Nicolson
        alpha = 0.25 * self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 - 
                                   self.option.r * np.arange(1, self.M))
        beta_plus = -0.5 * self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 + 
                                       self.option.r)
        gamma = 0.25 * self.dt * (self.option.sigma**2 * np.arange(1, self.M)**2 + 
                                   self.option.r * np.arange(1, self.M))
        
        # Build matrices
        A = np.diag(1 - beta_plus) + np.diag(-gamma[:-1], 1) + np.diag(-alpha[1:], -1)
        B = np.diag(1 + beta_plus) + np.diag(gamma[:-1], 1) + np.diag(alpha[1:], -1)
        
        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._boundary_conditions(j)
            
            # Right-hand side
            b = B @ self.V[1:-1, j+1]
            b[0] += alpha[0] * (self.V[0, j] + self.V[0, j+1])
            b[-1] += gamma[-1] * (self.V[-1, j] + self.V[-1, j+1])
            
            # Solve linear system
            self.V[1:-1, j] = np.linalg.solve(A, b)
        
        # Interpolate to get price at S_0
        price = np.interp(self.option.S, self.price_grid, self.V[:, 0])
        
        elapsed_time = time.time() - start_time
        return {
            "price": price,
            "computation_time": elapsed_time
        }
