"""
Python Pricing Service - Flask API wrapper for options pricing engine
Integrates with Go API gateway to provide pricing, Greeks, and calibration services
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
import os
import time
import logging
from typing import Dict, Any, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models import HestonModel, black_scholes_iv
from pricer import HestonPricerFFT, HestonPricerMC
from calibrator import HestonCalibrator
from utils import GreeksCalculator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_PORT = int(os.getenv('PRICING_SERVICE_PORT', 5000))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, LOG_LEVEL))


def validate_pricing_params(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate pricing request parameters"""
    required = ['spot', 'strike', 'rate', 'dividend', 'maturity', 'option_type']
    model_params = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    
    # Check required fields
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Check model parameters
    for param in model_params:
        if param not in data:
            return False, f"Missing model parameter: {param}"
    
    # Validate ranges
    if data['spot'] <= 0:
        return False, "spot must be positive"
    if data['strike'] <= 0:
        return False, "strike must be positive"
    if data['maturity'] <= 0:
        return False, "maturity must be positive"
    if data['option_type'] not in ['call', 'put']:
        return False, "option_type must be 'call' or 'put'"
    if data['v0'] <= 0:
        return False, "v0 (initial variance) must be positive"
    if data['kappa'] <= 0:
        return False, "kappa (mean reversion) must be positive"
    if data['theta'] <= 0:
        return False, "theta (long-term variance) must be positive"
    if data['sigma'] <= 0:
        return False, "sigma (vol of vol) must be positive"
    if not -1 <= data['rho'] <= 1:
        return False, "rho (correlation) must be between -1 and 1"
    
    return True, ""


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'options-pricing-service',
        'version': '1.0.0',
        'timestamp': time.time()
    }), 200


@app.route('/api/v1/price', methods=['POST'])
def calculate_price():
    """
    Calculate option price using Heston model
    
    Request body:
    {
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.05,
        "dividend": 0.02,
        "maturity": 1.0,
        "option_type": "call",
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "method": "fft"  // Optional: "fft" or "mc"
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Validate parameters
        is_valid, error_msg = validate_pricing_params(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Extract parameters
        S0 = float(data['spot'])
        K = float(data['strike'])
        r = float(data['rate'])
        q = float(data['dividend'])
        T = float(data['maturity'])
        option_type = data['option_type'].lower()
        
        # Model parameters
        v0 = float(data['v0'])
        kappa = float(data['kappa'])
        theta = float(data['theta'])
        sigma = float(data['sigma'])
        rho = float(data['rho'])
        
        # Method (default to FFT)
        method = data.get('method', 'fft').lower()
        
        # Create model
        model = HestonModel(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        
        # Price based on method
        if method == 'fft':
            pricer = HestonPricerFFT(model, r, q)
            price = pricer.price(S0, K, T, option_type)
        elif method == 'mc':
            pricer = HestonPricerMC(model, r, q)
            price = pricer.price(S0, K, T, option_type, n_paths=50000, n_steps=100)
        else:
            return jsonify({'error': f'Invalid method: {method}. Use "fft" or "mc"'}), 400
        
        computation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate implied volatility for comparison
        try:
            implied_vol = black_scholes_iv(price, S0, K, r, T, option_type)
        except:
            implied_vol = None
        
        logger.info(f"Priced {option_type} option: S={S0}, K={K}, T={T}, price={price:.6f}, method={method}, time={computation_time:.2f}ms")
        
        return jsonify({
            'price': float(price),
            'implied_volatility': float(implied_vol) if implied_vol else None,
            'method': method,
            'model': 'heston',
            'computation_time_ms': computation_time,
            'parameters': {
                'spot': S0,
                'strike': K,
                'maturity': T,
                'rate': r,
                'dividend': q,
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calculating price: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/greeks', methods=['POST'])
def calculate_greeks():
    """
    Calculate option Greeks using Heston model
    
    Request body: Same as /price endpoint
    """
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Validate parameters
        is_valid, error_msg = validate_pricing_params(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Extract parameters
        S0 = float(data['spot'])
        K = float(data['strike'])
        r = float(data['rate'])
        q = float(data['dividend'])
        T = float(data['maturity'])
        option_type = data['option_type'].lower()
        
        # Model parameters
        v0 = float(data['v0'])
        kappa = float(data['kappa'])
        theta = float(data['theta'])
        sigma = float(data['sigma'])
        rho = float(data['rho'])
        
        # Create model and pricer
        model = HestonModel(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        pricer = HestonPricerFFT(model, r, q)
        
        # Calculate Greeks
        calculator = GreeksCalculator(model, pricer, r, q)
        greeks = calculator.calculate_all_greeks(S0, K, T, option_type)
        
        computation_time = (time.time() - start_time) * 1000
        
        logger.info(f"Calculated Greeks for {option_type}: S={S0}, K={K}, T={T}, time={computation_time:.2f}ms")
        
        return jsonify({
            'price': float(greeks['price']),
            'delta': float(greeks['delta']),
            'gamma': float(greeks['gamma']),
            'vega': float(greeks['vega']),
            'theta': float(greeks['theta']),
            'rho': float(greeks['rho']),
            'computation_time_ms': computation_time,
            'model': 'heston',
            'parameters': {
                'spot': S0,
                'strike': K,
                'maturity': T,
                'rate': r,
                'dividend': q,
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calculating Greeks: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/calibrate', methods=['POST'])
def calibrate_model():
    """
    Calibrate Heston model to market data
    
    Request body:
    {
        "spot": 100.0,
        "rate": 0.05,
        "dividend": 0.02,
        "strikes": [90, 95, 100, 105, 110],
        "maturities": [0.25, 0.5, 1.0],
        "market_prices": [[...], [...], [...]],  // 2D array: maturities x strikes
        "option_type": "call",
        "initial_guess": {  // Optional
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7
        }
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Validate required fields
        required = ['spot', 'rate', 'dividend', 'strikes', 'maturities', 'market_prices', 'option_type']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        S0 = float(data['spot'])
        r = float(data['rate'])
        q = float(data['dividend'])
        strikes = np.array(data['strikes'], dtype=float)
        maturities = np.array(data['maturities'], dtype=float)
        market_prices = np.array(data['market_prices'], dtype=float)
        option_type = data['option_type'].lower()
        
        # Validate shapes
        if market_prices.shape != (len(maturities), len(strikes)):
            return jsonify({
                'error': f'market_prices shape {market_prices.shape} does not match (maturities, strikes) = ({len(maturities)}, {len(strikes)})'
            }), 400
        
        # Initial guess
        if 'initial_guess' in data:
            ig = data['initial_guess']
            v0_guess = float(ig.get('v0', 0.04))
            kappa_guess = float(ig.get('kappa', 2.0))
            theta_guess = float(ig.get('theta', 0.04))
            sigma_guess = float(ig.get('sigma', 0.3))
            rho_guess = float(ig.get('rho', -0.7))
        else:
            v0_guess = 0.04
            kappa_guess = 2.0
            theta_guess = 0.04
            sigma_guess = 0.3
            rho_guess = -0.7
        
        # Create calibrator
        initial_model = HestonModel(v0=v0_guess, kappa=kappa_guess, theta=theta_guess, 
                                    sigma=sigma_guess, rho=rho_guess)
        calibrator = HestonCalibrator(initial_model, r, q)
        
        # Calibrate
        logger.info(f"Starting calibration: {len(strikes)} strikes, {len(maturities)} maturities")
        calibrated_model, rmse = calibrator.calibrate(S0, strikes, maturities, market_prices, option_type)
        
        computation_time = (time.time() - start_time) * 1000
        
        # Calculate calibrated prices for comparison
        pricer = HestonPricerFFT(calibrated_model, r, q)
        calibrated_prices = np.zeros_like(market_prices)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                calibrated_prices[i, j] = pricer.price(S0, K, T, option_type)
        
        # Calculate residuals
        residuals = market_prices - calibrated_prices
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        logger.info(f"Calibration complete: RMSE={rmse:.6f}, MAE={mae:.6f}, time={computation_time:.2f}ms")
        
        return jsonify({
            'success': True,
            'parameters': {
                'v0': float(calibrated_model.v0),
                'kappa': float(calibrated_model.kappa),
                'theta': float(calibrated_model.theta),
                'sigma': float(calibrated_model.sigma),
                'rho': float(calibrated_model.rho)
            },
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'max_error': float(max_error),
                'num_options': int(len(strikes) * len(maturities))
            },
            'calibrated_prices': calibrated_prices.tolist(),
            'market_prices': market_prices.tolist(),
            'residuals': residuals.tolist(),
            'computation_time_ms': computation_time,
            'initial_guess': {
                'v0': v0_guess,
                'kappa': kappa_guess,
                'theta': theta_guess,
                'sigma': sigma_guess,
                'rho': rho_guess
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/v1/surface', methods=['POST'])
def calculate_surface():
    """
    Calculate volatility surface for given parameters
    
    Request body:
    {
        "spot": 100.0,
        "rate": 0.05,
        "dividend": 0.02,
        "strikes": [90, 95, 100, 105, 110],
        "maturities": [0.25, 0.5, 1.0, 2.0],
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "option_type": "call"
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Validate required fields
        required = ['spot', 'rate', 'dividend', 'strikes', 'maturities', 
                   'v0', 'kappa', 'theta', 'sigma', 'rho', 'option_type']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        S0 = float(data['spot'])
        r = float(data['rate'])
        q = float(data['dividend'])
        strikes = np.array(data['strikes'], dtype=float)
        maturities = np.array(data['maturities'], dtype=float)
        option_type = data['option_type'].lower()
        
        # Model parameters
        v0 = float(data['v0'])
        kappa = float(data['kappa'])
        theta = float(data['theta'])
        sigma = float(data['sigma'])
        rho = float(data['rho'])
        
        # Create model and pricer
        model = HestonModel(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        pricer = HestonPricerFFT(model, r, q)
        
        # Calculate prices and implied volatilities
        prices = np.zeros((len(maturities), len(strikes)))
        implied_vols = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                price = pricer.price(S0, K, T, option_type)
                prices[i, j] = price
                
                try:
                    iv = black_scholes_iv(price, S0, K, r, T, option_type)
                    implied_vols[i, j] = iv if iv else np.nan
                except:
                    implied_vols[i, j] = np.nan
        
        computation_time = (time.time() - start_time) * 1000
        
        logger.info(f"Calculated surface: {len(strikes)} strikes x {len(maturities)} maturities, time={computation_time:.2f}ms")
        
        return jsonify({
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'prices': prices.tolist(),
            'implied_volatilities': implied_vols.tolist(),
            'model': 'heston',
            'computation_time_ms': computation_time,
            'parameters': {
                'spot': S0,
                'rate': r,
                'dividend': q,
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calculating surface: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info(f"Starting Options Pricing Service on port {SERVICE_PORT}")
    logger.info(f"Log level: {LOG_LEVEL}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=SERVICE_PORT,
        debug=(os.getenv('FLASK_ENV') == 'development'),
        threaded=True
    )
