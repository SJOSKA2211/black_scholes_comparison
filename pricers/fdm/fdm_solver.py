"""
Finite Difference Methods for Option Pricing
Implements explicit, implicit, and Crank-Nicolson schemes
"""
import numpy as np
from typing import Tuple, Dict, Any
from .._base_pricer import Option, BasePricer # Import BasePricer
import time
from utils.decorators import time_it, measure_memory # Import decorators

class FiniteDifference(BasePricer): # Inherit from BasePricer
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
        super().__init__(option) # Call parent constructor
        self.M = M
        self.N = N
        self.S_max = S_max or 3 * option.K
        
        # Create grids
        self.dt = self.option.T / N
        self.dS = self.S_max / M
        
        # Initialize solution grid (will be reset for each pricing call)
        self.V = None 
    
    def _initialize_grids_and_conditions(self):
        """Initializes V grid and sets initial/boundary conditions."""
        self.V = np.zeros((self.M + 1, self.N + 1))
        # Set initial condition (payoff at maturity)
        s_values = np.linspace(0, self.S_max, self.M + 1)
        if self.option.option_type == 'call':
            self.V[:, -1] = np.maximum(s_values - self.option.K, 0)
        else:
            self.V[:, -1] = np.maximum(self.option.K - s_values, 0)
        
    def _set_boundary_conditions(self, j: int):
        """Set boundary conditions at time step j"""
        s_prices = np.linspace(0, self.S_max, self.M + 1)
        if self.option.option_type == 'call':
            self.V[0, j] = 0  # Call worth 0 at S=0
            self.V[-1, j] = s_prices[-1] - self.option.K * \
                            np.exp(-self.option.r * (self.option.T - j * self.dt))
        else:
            self.V[0, j] = self.option.K * \
                           np.exp(-self.option.r * (self.option.T - j * self.dt))
            self.V[-1, j] = 0  # Put worth 0 at S=inf
            
    def _explicit_price(self) -> float:
        """
        Explicit finite difference scheme (Forward Euler)
        Returns only the price.
        """
        self._initialize_grids_and_conditions()
        
        # Calculate coefficients
        i_values = np.arange(self.M + 1) # Use i_values for coefficients calculation
        alpha = 0.5 * self.dt * (self.option.sigma**2 * i_values**2 - 
                                  self.option.r * i_values)
        beta = 1 - self.dt * (self.option.sigma**2 * i_values**2 + 
                               self.option.r)
        gamma = 0.5 * self.dt * (self.option.sigma**2 * i_values**2 + 
                                  self.option.r * i_values)

        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._set_boundary_conditions(j)
            
            for i in range(1, self.M):
                self.V[i, j] = (alpha[i] * self.V[i-1, j+1] + 
                                beta[i] * self.V[i, j+1] + 
                                gamma[i] * self.V[i+1, j+1])
        
        # Interpolate to get price at option.S
        return np.interp(self.option.S, np.linspace(0, self.S_max, self.M + 1), self.V[:, 0])
    
    def _implicit_price(self) -> float:
        """
        Implicit finite difference scheme (Backward Euler)
        Returns only the price.
        """
        self._initialize_grids_and_conditions()

        # Build tridiagonal matrix coefficients
        i_values_center = np.arange(1, self.M) # i from 1 to M-1
        alpha_coeff = -0.5 * self.dt * (self.option.sigma**2 * i_values_center**2 - 
                                         self.option.r * i_values_center)
        beta_coeff = 1 + self.dt * (self.option.sigma**2 * i_values_center**2 + 
                                    self.option.r)
        gamma_coeff = -0.5 * self.dt * (self.option.sigma**2 * i_values_center**2 + 
                                         self.option.r * i_values_center)
        
        # Create tridiagonal matrix A
        A = np.diag(beta_coeff) + np.diag(gamma_coeff[:-1], 1) + np.diag(alpha_coeff[1:], -1)
        
        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._set_boundary_conditions(j)
            
            # Right-hand side vector b
            b = self.V[1:-1, j+1].copy()
            # Adjust b for boundary conditions
            b[0] -= alpha_coeff[0] * self.V[0, j]
            b[-1] -= gamma_coeff[-1] * self.V[-1, j]
            
            # Solve linear system
            self.V[1:-1, j] = np.linalg.solve(A, b)
        
        # Interpolate to get price at option.S
        return np.interp(self.option.S, np.linspace(0, self.S_max, self.M + 1), self.V[:, 0])
    
    def _crank_nicolson_price(self) -> float:
        """
        Crank-Nicolson finite difference scheme
        Returns only the price.
        """
        self._initialize_grids_and_conditions()
        
        # Coefficients for Crank-Nicolson
        i_values_center = np.arange(1, self.M) # i from 1 to M-1
        alpha_cn = 0.25 * self.dt * (self.option.sigma**2 * i_values_center**2 - 
                                   self.option.r * i_values_center)
        beta_cn_plus = -0.5 * self.dt * (self.option.sigma**2 * i_values_center**2 + 
                                       self.option.r)
        gamma_cn = 0.25 * self.dt * (self.option.sigma**2 * i_values_center**2 + 
                                   self.option.r * i_values_center)
        
        # Build matrices A and B (A * V_j = B * V_{j+1})
        # A matrix (for V_j)
        A = np.diag(1 - beta_cn_plus) + np.diag(-gamma_cn[:-1], 1) + np.diag(-alpha_cn[1:], -1)
        # B matrix (for V_{j+1})
        B = np.diag(1 + beta_cn_plus) + np.diag(gamma_cn[:-1], 1) + np.diag(alpha_cn[1:], -1)
        
        # Backward iteration in time
        for j in range(self.N - 1, -1, -1):
            self._set_boundary_conditions(j)
            
            # Right-hand side
            b = B @ self.V[1:-1, j+1]
            b[0] += alpha_cn[0] * (self.V[0, j] + self.V[0, j+1])
            b[-1] += gamma_cn[-1] * (self.V[-1, j] + self.V[-1, j+1])
            
            # Solve linear system
            self.V[1:-1, j] = np.linalg.solve(A, b)
        
        # Interpolate to get price at option.S
        return np.interp(self.option.S, np.linspace(0, self.S_max, self.M + 1), self.V[:, 0])

    @time_it # Apply decorator
    @measure_memory # Apply decorator
    def price(self, method_type: str = "crank_nicolson") -> Dict[str, Any]:
        """
        Calculates the option price using the specified finite difference method.
        
        Parameters:
        -----------
        method_type : str
            Type of FDM to use ('explicit', 'implicit', 'crank_nicolson').
            Defaults to 'crank_nicolson'.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the option price and computation time.
        """
        # start_time = time.time() # Removed, handled by decorator
        
        if method_type == "explicit":
            price_val = self._explicit_price()
        elif method_type == "implicit":
            price_val = self._implicit_price()
        elif method_type == "crank_nicolson":
            price_val = self._crank_nicolson_price()
        else:
            raise ValueError(f"Unknown FDM method type: {method_type}")
        
        # elapsed_time = time.time() - start_time # Removed, handled by decorator
        return {
            "price": price_val,
            # "computation_time": elapsed_time # Removed, handled by decorator
        }

    @time_it # Apply decorator
    @measure_memory # Apply decorator
    def get_greeks(self, method_type: str = "crank_nicolson", dS: float = 0.01, dSigma: float = 0.001, dr: float = 0.0001, dT: float = 0.0001) -> Dict[str, float]:
        """
        Calculates option Greeks using finite difference approximation.
        
        Parameters:
        -----------
        method_type : str
            Type of FDM to use for pricing when calculating Greeks.
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
            # Temporarily create a new FDM solver with the perturbed option
            # It's important to copy M, N, S_max to ensure consistent settings
            temp_fdm = FiniteDifference(opt, M=self.M, N=self.N, S_max=self.S_max)
            return temp_fdm.price(method_type)["price"]

        # Original price
        price_original = get_price_for_greeks(original_option)

        # Delta
        option_plus_dS = Option(S=original_option.S + dS, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type)
        option_minus_dS = Option(S=original_option.S - dS, K=original_option.K, T=original_option.T, r=original_option.r, sigma=original_option.sigma, option_type=original_option.option_type)
        price_plus_dS = get_price_for_greeks(option_plus_dS)
        price_minus_dS = get_price_for_greeks(option_minus_dS)
        delta = (price_plus_dS - price_minus_dS) / (2 * dS)

        # Gamma
        # Need to calculate price at original S. We already have price_original.
        # option_plus_dS is (S+dS), option_minus_dS is (S-dS)
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
