# Results Directory

This directory stores output figures and analysis results from the notebooks.

## Generated Files

When you run the notebooks, the following files will be created here:

### From Notebook 1 (Model Exploration):
- `price_comparison.png` - Heston vs Black-Scholes prices
- `volatility_smile.png` - Implied volatility smile
- `parameter_sensitivity.png` - Parameter sensitivity analysis
- `greeks_profile.png` - Greeks across spot prices

### From Notebook 2 (Calibration):
- `market_iv_surface.png` - Market implied volatility
- `calibration_fit.png` - Model vs market prices
- `error_analysis.png` - Calibration error analysis
- `iv_comparison.png` - IV fit quality

### From Notebook 3 (Results):
- `performance_comparison.png` - FFT vs Monte Carlo speed
- `project_summary.png` - Overall metrics dashboard
- `project_metrics.csv` - Summary statistics table

## Viewing Results

All charts are saved in high resolution (300 DPI) PNG format, suitable for presentations and portfolio.

## Note

Results folder is gitignored by default (except this README) to keep repository clean.
