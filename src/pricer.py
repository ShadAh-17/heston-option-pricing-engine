"""
Heston model pricing engines using FFT and Monte Carlo methods.
"""

import numpy as np
from typing import Tuple, Optional, List
from .models import HestonModel


class HestonPricerFFT:
    """
    FFT-based pricer using Carr-Madan (1999) approach.
    Prices European options in O(N log N) complexity.
    """
    
    def __init__(self, model: HestonModel, N: int = 4096, alpha: float = 1.5,
                 eta: float = 0.25):
        self.model = model
        self.N = N
        self.alpha = alpha  # Dampening factor
        self.eta = eta      # Grid spacing in log-strike space
        
        # Derived params
        self.lambda_ = 2 * np.pi / (N * eta)  # Strike spacing
        self.b = N * self.lambda_ / 2         # Upper bound
    
    def _modified_characteristic(self, v: np.ndarray, S: float, K: float,
                                  r: float, T: float) -> np.ndarray:
        """Modified CF for Carr-Madan integration."""
        u = v - (self.alpha + 1) * 1j
        
        cf = self.model.characteristic_function(u, S, K, r, T)
        
        denom = self.alpha ** 2 + self.alpha - v ** 2 + 1j * (2 * self.alpha + 1) * v
        
        return np.exp(-r * T) * cf / denom
    
    def price(self, S: float, K: float, r: float, T: float,
              option_type: str = 'call') -> float:
        """Price a European option using FFT."""
        
        # Simpson's rule weights
        j = np.arange(self.N)
        weights = 3 + (-1) ** (j + 1) - (j == 0)
        weights = weights * self.eta / 3
        
        # Log-strike grid
        k = -self.b + self.lambda_ * j
        
        # Integration grid
        v = self.eta * j
        
        # Build integrand
        log_SK = np.log(S / K)
        x = self.model.characteristic_function(
            v - (self.alpha + 1) * 1j, S, K, r, T
        )
        
        denom = (self.alpha ** 2 + self.alpha - v ** 2 + 
                 1j * (2 * self.alpha + 1) * v)
        
        integrand = np.exp(-r * T) * x / denom
        integrand = integrand * np.exp(1j * v * (log_SK + self.b)) * weights
        
        # FFT
        fft_result = np.fft.fft(integrand)
        
        # Extract call prices
        call_prices = np.real(np.exp(-self.alpha * k) / np.pi * fft_result)
        
        # Interpolate to target strike
        target_k = np.log(K / S)
        call_price = np.interp(-target_k, k, call_prices)
        call_price = max(0, call_price)  # Floor at intrinsic
        
        if option_type == 'call':
            return call_price
        else:
            # Put-call parity
            return call_price - S + K * np.exp(-r * T)
    
    def price_surface(self, S: float, r: float, 
                      strikes: np.ndarray, maturities: np.ndarray,
                      option_type: str = 'call') -> np.ndarray:
        """Price options across K x T grid."""
        prices = np.zeros((len(strikes), len(maturities)))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                prices[i, j] = self.price(S, K, r, T, option_type)
        
        return prices


class HestonPricerMC:
    """
    Monte Carlo pricer with Euler discretization.
    Slower but more flexible than FFT (can handle path-dependent payoffs).
    """
    
    def __init__(self, model: HestonModel, n_paths: int = 100000,
                 n_steps: int = 252, seed: Optional[int] = None):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
    
    def _simulate_paths(self, S0: float, r: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and variance paths using Euler scheme.
        Uses full truncation for variance to prevent negative values.
        """
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.full(self.n_paths, S0)
        v = np.full(self.n_paths, self.model.v0)
        
        # Pre-generate all random numbers
        Z1 = self.rng.standard_normal((self.n_steps, self.n_paths))
        Z2 = self.rng.standard_normal((self.n_steps, self.n_paths))
        
        # Correlate the Brownian motions
        W1 = Z1
        W2 = self.model.rho * Z1 + np.sqrt(1 - self.model.rho ** 2) * Z2
        
        for i in range(self.n_steps):
            v_pos = np.maximum(v, 0)  # Full truncation
            sqrt_v = np.sqrt(v_pos)
            
            # Stock process
            S = S * np.exp((r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W1[i])
            
            # Variance process  
            v = v + self.model.kappa * (self.model.theta - v_pos) * dt + \
                self.model.sigma * sqrt_v * sqrt_dt * W2[i]
        
        return S, v
    
    def price(self, S: float, K: float, r: float, T: float,
              option_type: str = 'call') -> Tuple[float, float]:
        """
        Price option via MC simulation.
        Returns (price, standard_error).
        """
        S_T, _ = self._simulate_paths(S, r, T)
        
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount
        discounted = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted)
        std_err = np.std(discounted) / np.sqrt(self.n_paths)
        
        return price, std_err
    
    def price_with_antithetic(self, S: float, K: float, r: float, T: float,
                               option_type: str = 'call') -> Tuple[float, float]:
        """
        Antithetic variates for variance reduction.
        Uses both Z and -Z for the random draws.
        """
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        
        half_paths = self.n_paths // 2
        
        S_pos = np.full(half_paths, S)
        S_neg = np.full(half_paths, S)
        v_pos = np.full(half_paths, self.model.v0)
        v_neg = np.full(half_paths, self.model.v0)
        
        for _ in range(self.n_steps):
            Z1 = self.rng.standard_normal(half_paths)
            Z2 = self.rng.standard_normal(half_paths)
            
            W1 = Z1
            W2 = self.model.rho * Z1 + np.sqrt(1 - self.model.rho ** 2) * Z2
            
            # Positive path
            v_pos_trunc = np.maximum(v_pos, 0)
            sqrt_v_pos = np.sqrt(v_pos_trunc)
            S_pos = S_pos * np.exp((r - 0.5 * v_pos_trunc) * dt + sqrt_v_pos * sqrt_dt * W1)
            v_pos = v_pos + self.model.kappa * (self.model.theta - v_pos_trunc) * dt + \
                    self.model.sigma * sqrt_v_pos * sqrt_dt * W2
            
            # Negative path (antithetic)
            v_neg_trunc = np.maximum(v_neg, 0)
            sqrt_v_neg = np.sqrt(v_neg_trunc)
            S_neg = S_neg * np.exp((r - 0.5 * v_neg_trunc) * dt + sqrt_v_neg * sqrt_dt * (-W1))
            v_neg = v_neg + self.model.kappa * (self.model.theta - v_neg_trunc) * dt + \
                    self.model.sigma * sqrt_v_neg * sqrt_dt * (-W2)
        
        # Combine payoffs
        if option_type == 'call':
            payoffs_pos = np.maximum(S_pos - K, 0)
            payoffs_neg = np.maximum(S_neg - K, 0)
        else:
            payoffs_pos = np.maximum(K - S_pos, 0)
            payoffs_neg = np.maximum(K - S_neg, 0)
        
        # Average antithetic pairs
        payoffs = (payoffs_pos + payoffs_neg) / 2
        discounted = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted)
        std_err = np.std(discounted) / np.sqrt(half_paths)
        
        return price, std_err
