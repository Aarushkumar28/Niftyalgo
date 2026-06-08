# NIFTY Prediction and Backtesting System

## Setup Instructions

### 1. Install Dependencies

The script requires several Python packages. Install them using:

```bash
pip install -r requirements.txt
```

Or if using the venv:
```bash
cd venv
source bin/activate  # On macOS/Linux
pip install -r ../requirements.txt
```

### 2. Run the Script

From the `python/index_predictor` directory:

```bash
python venv/nifty_predict_full.py
```

Or if using the venv:
```bash
cd venv
source bin/activate
python nifty_predict_full.py
```

## Features

- **Advanced Technical Indicators**: MACD, Bollinger Bands, EMA crossovers, RSI, Volume analysis
- **Feature Selection**: Automated selection based on importance
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM
- **Ensemble Modeling**: Weighted ensemble of multiple models
- **Macro Features**: India VIX and USDINR integration
- **Comprehensive Backtesting**:
  - Multiple trading strategies (long-only, long-short, probability-based)
  - Performance metrics (Sharpe ratio, drawdown, win rate)
  - Detailed trade analysis
  - Visualization charts

## Configuration

Edit the configuration parameters in the script:
- `USE_WEEKLY`: Set to `True` for weekly predictions
- `ENABLE_BACKTESTING`: Enable/disable backtesting
- `BACKTEST_STRATEGY`: Choose trading strategy
- `TRANSACTION_COST`: Transaction cost per trade
- `INITIAL_CAPITAL`: Starting capital for backtesting

## Output Files

Results are saved in `results_nifty_enhanced/`:
- Model performance metrics
- Feature importances
- ROC curves
- Backtest results and trades
- Visualizations

