"""
Options Pricing Engine Package

A production-ready Heston model implementation for option pricing and calibration.
"""

from .models import HestonModel, black_scholes_price, black_scholes_iv
from .pricer import HestonPricerFFT, HestonPricerMC, compare_pricing_methods
from .calibrator import HestonCalibrator, create_synthetic_market_data, compute_implied_volatilities
from .utils import GreeksCalculator, plot_volatility_surface, plot_calibration_fit, plot_greeks_profile, print_summary_statistics

__all__ = [
    'HestonModel',
    'black_scholes_price',
    'black_scholes_iv',
    'HestonPricerFFT',
    'HestonPricerMC',
    'compare_pricing_methods',
    'HestonCalibrator',
    'create_synthetic_market_data',
    'compute_implied_volatilities',
    'GreeksCalculator',
    'plot_volatility_surface',
    'plot_calibration_fit',
    'plot_greeks_profile',
    'print_summary_statistics'
]

__version__ = '1.0.0'
__author__ = 'Shadaab Ahmed'
