"""
Heston model calibration to market implied volatilities.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

from .models import HestonModel, black_scholes_iv
from .pricer import HestonPricerFFT


@dataclass
class CalibrationResult:
    """Container for calibration output."""
    model: HestonModel
    rmse: float
    success: bool
    n_evals: int
    message: str


class HestonCalibrator:
    """
    Calibrate Heston parameters to observed option prices/IVs.
    Uses differential evolution for global optimization.
    """
    
    # Default parameter bounds
    BOUNDS = {
        'v0':    (0.001, 1.0),
        'kappa': (0.1, 10.0),
        'theta': (0.001, 1.0),
        'sigma': (0.01, 2.0),
        'rho':   (-0.99, 0.0)  # Usually negative for equities
    }
    
    def __init__(self, S: float, r: float):
        self.S = S
        self.r = r
        self._n_evals = 0
    
    def calibrate(self, strikes: np.ndarray, maturities: np.ndarray,
                  market_ivs: np.ndarray, weights: Optional[np.ndarray] = None,
                  bounds: Optional[Dict] = None, maxiter: int = 200,
                  polish: bool = True) -> CalibrationResult:
        """
        Calibrate to IV surface.
        
        Args:
            strikes: Strike prices (1D array)
            maturities: Time to expiry in years (1D array)
            market_ivs: Market IVs, shape (len(strikes), len(maturities))
            weights: Optional weights for each point
            bounds: Override default parameter bounds
            maxiter: Max differential evolution iterations
            polish: Run local optimization after DE
        
        Returns:
            CalibrationResult with fitted model and diagnostics
        """
        self._n_evals = 0
        
        if weights is None:
            weights = np.ones_like(market_ivs)
        
        bounds_list = self._get_bounds(bounds)
        
        def objective(params):
            self._n_evals += 1
            return self._iv_rmse(params, strikes, maturities, market_ivs, weights)
        
        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=maxiter,
            polish=polish,
            seed=42,
            disp=False
        )
        
        v0, kappa, theta, sigma, rho = result.x
        model = HestonModel(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        
        return CalibrationResult(
            model=model,
            rmse=result.fun,
            success=result.success,
            n_evals=self._n_evals,
            message=result.message
        )
    
    def calibrate_to_prices(self, strikes: np.ndarray, maturities: np.ndarray,
                            market_prices: np.ndarray, 
                            option_types: np.ndarray,
                            **kwargs) -> CalibrationResult:
        """
        Calibrate to option prices instead of IVs.
        Converts prices to IVs first, then calibrates.
        """
        market_ivs = np.zeros_like(market_prices)
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                opt_type = option_types[i, j] if option_types.ndim == 2 else option_types[i]
                iv = black_scholes_iv(
                    market_prices[i, j], self.S, K, self.r, T, opt_type
                )
                market_ivs[i, j] = iv if iv is not None else np.nan
        
        # Mask invalid IVs
        valid = ~np.isnan(market_ivs)
        weights = kwargs.pop('weights', None)
        if weights is None:
            weights = valid.astype(float)
        else:
            weights = weights * valid
        
        return self.calibrate(strikes, maturities, market_ivs, weights, **kwargs)
    
    def _iv_rmse(self, params: np.ndarray, strikes: np.ndarray, 
                 maturities: np.ndarray, market_ivs: np.ndarray,
                 weights: np.ndarray) -> float:
        """Compute weighted RMSE between model and market IVs."""
        
        v0, kappa, theta, sigma, rho = params
        
        try:
            model = HestonModel(v0=v0, kappa=kappa, theta=theta, 
                               sigma=sigma, rho=rho)
        except ValueError:
            return 1e10  # Invalid params
        
        pricer = HestonPricerFFT(model)
        
        total_sq_error = 0.0
        total_weight = 0.0
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if weights[i, j] <= 0 or np.isnan(market_ivs[i, j]):
                    continue
                
                try:
                    price = pricer.price(self.S, K, self.r, T, 'call')
                    model_iv = black_scholes_iv(price, self.S, K, self.r, T, 'call')
                    
                    if model_iv is None:
                        continue
                    
                    error = model_iv - market_ivs[i, j]
                    total_sq_error += weights[i, j] * error ** 2
                    total_weight += weights[i, j]
                except:
                    continue
        
        if total_weight == 0:
            return 1e10
        
        return np.sqrt(total_sq_error / total_weight)
    
    def _get_bounds(self, custom_bounds: Optional[Dict]) -> List[Tuple]:
        """Build bounds list from defaults + overrides."""
        bounds = self.BOUNDS.copy()
        if custom_bounds:
            bounds.update(custom_bounds)
        
        return [
            bounds['v0'],
            bounds['kappa'],
            bounds['theta'],
            bounds['sigma'],
            bounds['rho']
        ]
    
    def compute_model_ivs(self, model: HestonModel, strikes: np.ndarray,
                          maturities: np.ndarray) -> np.ndarray:
        """Compute IV surface for a given model."""
        pricer = HestonPricerFFT(model)
        ivs = np.zeros((len(strikes), len(maturities)))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                price = pricer.price(self.S, K, self.r, T, 'call')
                iv = black_scholes_iv(price, self.S, K, self.r, T, 'call')
                ivs[i, j] = iv if iv is not None else np.nan
        
        return ivs


def generate_sample_market_data(S: float = 100, r: float = 0.05,
                                 n_strikes: int = 10, n_maturities: int = 5,
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic market IVs from a known Heston model + noise.
    Useful for testing calibration.
    """
    np.random.seed(seed)
    
    # True model
    true_model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    
    strikes = np.linspace(0.8 * S, 1.2 * S, n_strikes)
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])[:n_maturities]
    
    pricer = HestonPricerFFT(true_model)
    
    ivs = np.zeros((n_strikes, n_maturities))
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            price = pricer.price(S, K, r, T, 'call')
            iv = black_scholes_iv(price, S, K, r, T, 'call')
            # Add some noise
            iv = iv * (1 + 0.02 * np.random.randn()) if iv else 0.2
            ivs[i, j] = iv
    
    return strikes, maturities, ivs
