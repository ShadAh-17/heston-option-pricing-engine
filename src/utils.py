"""
Utility functions for Greeks calculation and visualization.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
import matplotlib.pyplot as plt

from .models import HestonModel, black_scholes_price
from .pricer import HestonPricerFFT


class GreeksCalculator:
    """
    Compute option Greeks via finite differences.
    Works for any pricing model.
    """
    
    def __init__(self, model: HestonModel, method: str = 'fft'):
        self.model = model
        self.method = method
        
        if method == 'fft':
            self.pricer = HestonPricerFFT(model)
    
    def delta(self, S: float, K: float, r: float, T: float,
              option_type: str = 'call', bump: float = 0.01) -> float:
        """∂V/∂S via central difference."""
        dS = S * bump
        
        p_up = self.pricer.price(S + dS, K, r, T, option_type)
        p_down = self.pricer.price(S - dS, K, r, T, option_type)
        
        return (p_up - p_down) / (2 * dS)
    
    def gamma(self, S: float, K: float, r: float, T: float,
              option_type: str = 'call', bump: float = 0.01) -> float:
        """∂²V/∂S² via central difference."""
        dS = S * bump
        
        p_up = self.pricer.price(S + dS, K, r, T, option_type)
        p_mid = self.pricer.price(S, K, r, T, option_type)
        p_down = self.pricer.price(S - dS, K, r, T, option_type)
        
        return (p_up - 2 * p_mid + p_down) / (dS ** 2)
    
    def vega(self, S: float, K: float, r: float, T: float,
             option_type: str = 'call', bump: float = 0.01) -> float:
        """∂V/∂v0 (sensitivity to initial variance)."""
        dv = bump
        
        model_up = HestonModel(
            v0=self.model.v0 + dv,
            kappa=self.model.kappa,
            theta=self.model.theta,
            sigma=self.model.sigma,
            rho=self.model.rho
        )
        model_down = HestonModel(
            v0=max(0.001, self.model.v0 - dv),
            kappa=self.model.kappa,
            theta=self.model.theta,
            sigma=self.model.sigma,
            rho=self.model.rho
        )
        
        pricer_up = HestonPricerFFT(model_up)
        pricer_down = HestonPricerFFT(model_down)
        
        p_up = pricer_up.price(S, K, r, T, option_type)
        p_down = pricer_down.price(S, K, r, T, option_type)
        
        # Scale to 1% vol move (standard vega convention)
        return (p_up - p_down) / (2 * dv) * 2 * np.sqrt(self.model.v0) * 0.01
    
    def theta(self, S: float, K: float, r: float, T: float,
              option_type: str = 'call', bump: float = 1/365) -> float:
        """∂V/∂T (time decay per day)."""
        if T <= bump:
            return 0.0
        
        p_now = self.pricer.price(S, K, r, T, option_type)
        p_later = self.pricer.price(S, K, r, T - bump, option_type)
        
        return p_later - p_now  # Negative for long options
    
    def rho_rate(self, S: float, K: float, r: float, T: float,
                 option_type: str = 'call', bump: float = 0.01) -> float:
        """∂V/∂r (rate sensitivity per 1%)."""
        p_up = self.pricer.price(S, K, r + bump, T, option_type)
        p_down = self.pricer.price(S, K, r - bump, T, option_type)
        
        return (p_up - p_down) / 2
    
    def all_greeks(self, S: float, K: float, r: float, T: float,
                   option_type: str = 'call') -> Dict[str, float]:
        """Compute all Greeks at once."""
        return {
            'delta': self.delta(S, K, r, T, option_type),
            'gamma': self.gamma(S, K, r, T, option_type),
            'vega': self.vega(S, K, r, T, option_type),
            'theta': self.theta(S, K, r, T, option_type),
            'rho': self.rho_rate(S, K, r, T, option_type)
        }


def plot_iv_surface(strikes: np.ndarray, maturities: np.ndarray,
                    ivs: np.ndarray, title: str = 'IV Surface',
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot implied volatility surface as 3D mesh."""
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing='ij')
    
    ax.plot_surface(K_grid, T_grid, ivs * 100, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('IV (%)')
    ax.set_title(title)
    
    return ax


def plot_calibration_fit(strikes: np.ndarray, maturities: np.ndarray,
                         market_ivs: np.ndarray, model_ivs: np.ndarray,
                         figsize: tuple = (12, 5)) -> plt.Figure:
    """Compare market vs model IVs across maturities."""
    fig, axes = plt.subplots(1, len(maturities), figsize=figsize)
    
    if len(maturities) == 1:
        axes = [axes]
    
    for j, (T, ax) in enumerate(zip(maturities, axes)):
        ax.plot(strikes, market_ivs[:, j] * 100, 'ko', label='Market', markersize=6)
        ax.plot(strikes, model_ivs[:, j] * 100, 'r-', label='Model', linewidth=2)
        ax.set_xlabel('Strike')
        ax.set_ylabel('IV (%)')
        ax.set_title(f'T = {T:.2f}Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_greeks_profile(model: HestonModel, K: float, r: float, T: float,
                        spot_range: tuple = (0.7, 1.3),
                        n_points: int = 50) -> plt.Figure:
    """Plot all Greeks as function of spot price."""
    
    calc = GreeksCalculator(model)
    S_range = np.linspace(K * spot_range[0], K * spot_range[1], n_points)
    
    greeks = {'delta': [], 'gamma': [], 'vega': [], 'theta': []}
    
    for S in S_range:
        g = calc.all_greeks(S, K, r, T, 'call')
        for k in greeks:
            greeks[k].append(g[k])
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for ax, (name, values) in zip(axes.flat, greeks.items()):
        ax.plot(S_range, values, 'b-', linewidth=2)
        ax.axvline(K, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Spot')
        ax.set_ylabel(name.capitalize())
        ax.set_title(name.capitalize())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Greeks Profile (K={K}, T={T:.2f}Y)', y=1.02)
    plt.tight_layout()
    return fig


def print_pricing_summary(model: HestonModel, S: float, K: float, 
                          r: float, T: float):
    """Print formatted pricing and Greeks summary."""
    pricer = HestonPricerFFT(model)
    calc = GreeksCalculator(model)
    
    call_price = pricer.price(S, K, r, T, 'call')
    put_price = pricer.price(S, K, r, T, 'put')
    
    greeks_call = calc.all_greeks(S, K, r, T, 'call')
    
    print(f"\n{'='*50}")
    print(f"Heston Model Pricing Summary")
    print(f"{'='*50}")
    print(f"Spot: {S:.2f}  |  Strike: {K:.2f}  |  Rate: {r:.2%}  |  T: {T:.2f}Y")
    print(f"\nModel: {model}")
    print(f"Feller ratio: {model.feller_ratio:.3f} {'✓' if model.feller_ratio > 1 else '✗'}")
    print(f"\n{'Prices':^25}")
    print(f"-" * 25)
    print(f"Call: ${call_price:>10.4f}")
    print(f"Put:  ${put_price:>10.4f}")
    print(f"\n{'Greeks (Call)':^25}")
    print(f"-" * 25)
    for name, val in greeks_call.items():
        print(f"{name.capitalize():>8}: {val:>12.6f}")
    print(f"{'='*50}\n")
