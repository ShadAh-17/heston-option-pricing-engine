"""
Heston model implementation for stochastic volatility pricing.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class HestonModel:
    """
    Heston (1993) stochastic volatility model.
    
    dS = r*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
    corr(dW1, dW2) = rho
    """
    
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-run variance
    sigma: float   # Vol of vol
    rho: float     # Correlation
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Check Feller condition and parameter bounds."""
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")
        
        # Feller condition check
        feller = 2 * self.kappa * self.theta / (self.sigma ** 2)
        if feller <= 1:
            import warnings
            warnings.warn(f"Feller condition violated: 2κθ/σ² = {feller:.3f} <= 1")
    
    def characteristic_function(self, u: complex, S: float, K: float, 
                                 r: float, T: float) -> complex:
        """
        Compute the characteristic function φ(u) for Heston model.
        Uses the formulation from Gatheral (2006).
        """
        x = np.log(S / K) + r * T
        
        # Complex-valued helpers
        d = np.sqrt((self.rho * self.sigma * 1j * u - self.kappa) ** 2 
                    + self.sigma ** 2 * (1j * u + u ** 2))
        
        g = (self.kappa - self.rho * self.sigma * 1j * u - d) / \
            (self.kappa - self.rho * self.sigma * 1j * u + d)
        
        # Avoid numerical issues
        exp_dT = np.exp(-d * T)
        
        C = (self.kappa - self.rho * self.sigma * 1j * u - d) / (self.sigma ** 2) * \
            ((1 - exp_dT) / (1 - g * exp_dT))
        
        D = self.kappa * self.theta / (self.sigma ** 2) * \
            ((self.kappa - self.rho * self.sigma * 1j * u - d) * T - 
             2 * np.log((1 - g * exp_dT) / (1 - g)))
        
        return np.exp(1j * u * x + C * self.v0 + D)
    
    @property
    def feller_ratio(self) -> float:
        """Returns 2κθ/σ², should be > 1 for well-behaved process."""
        return 2 * self.kappa * self.theta / (self.sigma ** 2)
    
    def to_dict(self) -> dict:
        return {
            'v0': self.v0, 'kappa': self.kappa, 'theta': self.theta,
            'sigma': self.sigma, 'rho': self.rho
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HestonModel':
        return cls(**d)
    
    def __repr__(self):
        return (f"HestonModel(v0={self.v0:.4f}, κ={self.kappa:.2f}, "
                f"θ={self.theta:.4f}, σ={self.sigma:.2f}, ρ={self.rho:.2f})")


def black_scholes_price(S: float, K: float, r: float, T: float, 
                        sigma: float, option_type: str = 'call') -> float:
    """
    Standard Black-Scholes formula for European options.
    """
    if T <= 0:
        # At expiry
        if option_type == 'call':
            return max(S - K, 0)
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_iv(price: float, S: float, K: float, r: float, 
                     T: float, option_type: str = 'call',
                     max_iter: int = 100, tol: float = 1e-8) -> Optional[float]:
    """
    Implied volatility via Newton-Raphson.
    Returns None if no valid IV found.
    """
    if T <= 0:
        return None
    
    # Initial guess from Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = max(0.01, min(sigma, 3.0))  # Bound it
    
    for _ in range(max_iter):
        bs_price = black_scholes_price(S, K, r, T, sigma, option_type)
        diff = bs_price - price
        
        if abs(diff) < tol:
            return sigma
        
        # Vega for Newton step
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            return None
        
        sigma = sigma - diff / vega
        sigma = max(0.001, min(sigma, 5.0))
    
    return None
