# Quick Start Guide

## Running the Script

### Option 1: Using the venv (Recommended)

```bash
cd "/Users/aarushkumar/Desktop/programming/AARUSH CODE/python/index_predictor/venv"
source bin/activate
python nifty_predict_full.py
```

### Option 2: Using the venv Python directly

```bash
cd "/Users/aarushkumar/Desktop/programming/AARUSH CODE/python/index_predictor"
./venv/bin/python venv/nifty_predict_full.py
```

## What the Script Does

1. **Downloads Data**: NIFTY index data from 2010, plus India VIX and USDINR
2. **Feature Engineering**: Creates 50+ technical indicators
3. **Model Training**: Trains multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, etc.)
4. **Ensemble**: Combines models for better predictions
5. **Backtesting**: Simulates trading strategies and calculates performance metrics
6. **Visualization**: Generates charts and reports

## Output

All results are saved in `results_nifty_enhanced/` directory:
- Model performance metrics
- Backtest results
- Trade logs
- Visualizations (equity curves, drawdowns, etc.)

## Troubleshooting

If you get "ModuleNotFoundError":
1. Make sure you're using the correct Python from the venv
2. Activate the venv: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Configuration

Edit these parameters at the top of `nifty_predict_full.py`:
- `USE_WEEKLY = True/False` - Use weekly or daily data
- `ENABLE_BACKTESTING = True/False` - Enable backtesting
- `BACKTEST_STRATEGY` - Choose "long_only", "long_short", or "probability_based"
- `TRANSACTION_COST = 0.001` - 0.1% transaction cost
- `INITIAL_CAPITAL = 100000` - Starting capital for backtesting

