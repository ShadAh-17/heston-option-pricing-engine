# Data Directory

This directory stores market data for calibration.

## Files

- `sample_market_data.csv` - Example market options data (not included, generated in notebooks)

## Data Format

Expected CSV format for market data:

```csv
strike,maturity,price,spot,risk_free_rate,dividend_yield,option_type
90,0.25,11.23,100,0.05,0.02,call
95,0.25,7.45,100,0.05,0.02,call
...
```

## Usage

Market data is generated synthetically in the calibration notebook using `create_synthetic_market_data()`.

For real market data:
1. Download options chain from Bloomberg, Yahoo Finance, or other provider
2. Format according to schema above
3. Load using pandas: `pd.read_csv('data/market_data.csv')`
