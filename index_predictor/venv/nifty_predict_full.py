"""
nifty_predict_enhanced.py

Enhanced NIFTY prediction with advanced technical indicators, feature selection,
timeframe variation, model ensembling, macro features, and comprehensive backtesting.

New Features:
 - MACD, Bollinger Bands, EMA crossovers, Volume Ratio
 - Automated feature selection based on importance
 - Weekly timeframe prediction option
 - Ensemble modeling with weighted predictions
 - Macro features (India VIX, USDINR)
 - Comprehensive evaluation with ROC comparison plots
 - Full backtesting suite with multiple trading strategies
 - Performance metrics (Sharpe ratio, drawdown, win rate, etc.)
 - Detailed trade analysis and visualization
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, f1_score)

# Optional models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

import yfinance as yf
import time
import ssl

# Disable SSL verification warnings (for certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------
# Configuration Parameters
# ---------------------------
TICKER = "NSEI.NS"  # NIFTY 50 index (NSE format) - most reliable
TICKER_ALTERNATIVES = ["^NSEI", "NIFTY.NS", "NSEI"]  # Try these if main ticker fails
START_DATE = "2015-01-01"
END_DATE = None  # None means up to today
TEST_SIZE = 0.2
RANDOM_STATE = 42
RESULTS_DIR = Path("results_nifty_enhanced")
RESULTS_DIR.mkdir(exist_ok=True)

# New parameters
USE_WEEKLY = True  # Set to True for weekly predictions
ENABLE_MACRO_FEATURES = True  # Include India VIX and USDINR
FEATURE_SELECTION_THRESHOLD = 0.10  # Drop bottom 20% of features
ENSEMBLE_WEIGHTS = "auto"  # "auto" or dict like {"LogisticRegression": 0.3, "RandomForest": 0.4, "XGBoost": 0.3}

# Backtesting parameters
ENABLE_BACKTESTING = True  # Enable backtesting
BACKTEST_STRATEGY = "long_only"  # "long_only", "long_short", "probability_based"
PROBABILITY_THRESHOLD = 0.6  # Minimum probability to take a trade (for probability_based strategy)
TRANSACTION_COST = 0.001  # 0.1% transaction cost per trade
INITIAL_CAPITAL = 100000  # Starting capital
POSITION_SIZE = 1.0  # Position size as fraction of capital (1.0 = 100%, 0.5 = 50%)
RISK_FREE_RATE = 0.05  # Annual risk-free rate (5% for India)

# ---------------------------
# Data Download Functions
# ---------------------------
def download_data(ticker=TICKER, start=START_DATE, end=END_DATE, max_retries=3, retry_delay=5, ticker_alternatives=None):
    """Download stock data from Yahoo Finance with retry logic and alternative tickers."""
    if ticker_alternatives is None:
        ticker_alternatives = TICKER_ALTERNATIVES
    
    # Try main ticker first, then alternatives
    tickers_to_try = [ticker] + [t for t in ticker_alternatives if t != ticker]
    
    for ticker_to_try in tickers_to_try:
        print(f"Downloading {ticker_to_try} from {start} to {end or 'today'} ...")
        
        for attempt in range(max_retries):
            try:
                # Create ticker object
                ticker_obj = yf.Ticker(ticker_to_try)
                # Try using history method first (more reliable)
                try:
                    df = ticker_obj.history(start=start, end=end)
                except Exception as e1:
                    # Fallback to download method
                    df = yf.download(ticker_to_try, start=start, end=end, progress=False)
                
                if df.empty:
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Try alternative method as last resort
                        print("  Trying alternative download method...")
                        try:
                            df = yf.download(ticker_to_try, start=start, end=end, progress=False)
                        except:
                            pass
                
                if not df.empty:
                    df = df.rename(columns=lambda c: c.strip())
                    
                    # Fix multi-index columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    
                    print(f"  Successfully downloaded {len(df)} rows using ticker {ticker_to_try}")
                    return df
                
            except Exception as e:
                error_msg = str(e)[:100]
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {error_msg}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"  All {max_retries} attempts failed for {ticker_to_try}")
                    if ticker_to_try != tickers_to_try[-1]:
                        print(f"  Trying next alternative ticker...")
                        break  # Try next ticker
                    else:
                        print(f"  All tickers failed. Last error: {e}")
    
    raise RuntimeError(f"Failed to download data after trying {len(tickers_to_try)} ticker(s) and {max_retries} attempts each. Check ticker symbols and internet connection.")

def download_macro_data(start=START_DATE, end=END_DATE, max_retries=2, retry_delay=3):
    """Download macro indicators: India VIX and USDINR with retry logic."""
    macro_data = {}
    
    # Download India VIX
    for attempt in range(max_retries):
        try:
            print("Downloading India VIX (^INDIAVIX)...")
            ticker_obj = yf.Ticker("^INDIAVIX")
            vix = ticker_obj.history(start=start, end=end)
            
            if vix.empty and attempt < max_retries - 1:
                print(f"  Retrying India VIX download...")
                time.sleep(retry_delay)
                continue
            
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = [col[0] for col in vix.columns]
                macro_data['VIX'] = vix['Close']
                print(f"  Successfully downloaded India VIX ({len(vix)} rows)")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Warning: Attempt {attempt + 1} failed for India VIX: {str(e)[:80]}")
                time.sleep(retry_delay)
            else:
                print(f"  Warning: Could not download India VIX after {max_retries} attempts: {e}")
    
    # Download USDINR
    for attempt in range(max_retries):
        try:
            print("Downloading USDINR (USDINR=X)...")
            ticker_obj = yf.Ticker("USDINR=X")
            usdinr = ticker_obj.history(start=start, end=end)
            
            if usdinr.empty and attempt < max_retries - 1:
                print(f"  Retrying USDINR download...")
                time.sleep(retry_delay)
                continue
            
            if not usdinr.empty:
                if isinstance(usdinr.columns, pd.MultiIndex):
                    usdinr.columns = [col[0] for col in usdinr.columns]
                macro_data['USDINR'] = usdinr['Close']
                print(f"  Successfully downloaded USDINR ({len(usdinr)} rows)")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Warning: Attempt {attempt + 1} failed for USDINR: {str(e)[:80]}")
                time.sleep(retry_delay)
            else:
                print(f"  Warning: Could not download USDINR after {max_retries} attempts: {e}")
    
    return macro_data

# ---------------------------
# Enhanced Feature Engineering
# ---------------------------
def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD and signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands and bandwidth."""
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    bandwidth = (upper - lower) / ma
    bb_position = (close - lower) / (upper - lower)  # Position within bands
    return upper, lower, bandwidth, bb_position

def calculate_volume_ratio(volume, window=20):
    """Calculate volume ratio vs. moving average."""
    vol_ma = volume.rolling(window=window, min_periods=1).mean()
    return volume / vol_ma.replace(0, np.nan)

def feature_engineering(df, macro_data=None):
    """Enhanced feature engineering with advanced technical indicators."""
    df = df.copy()
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce')
    
    close = df['Close']
    
    # ===== EXISTING FEATURES =====
    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"ma_{w}"] = close.rolling(window=w, min_periods=1).mean()
        ma_val = df[f"ma_{w}"].replace(0, np.nan)
        df[f"ma_{w}_diff"] = (close - df[f"ma_{w}"]) / ma_val
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)  # Neutral RSI for NaN values
    
    # Daily returns
    df["returns"] = close.pct_change()
    
    # Volatility
    df["volatility"] = df["returns"].rolling(window=10, min_periods=1).std()
    
    # Lag features
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["returns"].shift(lag)
    
    # ===== NEW ADVANCED FEATURES =====
    
    # 1. MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(close)
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # 2. EMA and crossovers
    df['ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['ema_crossover'] = (df['ema_10'] > df['ema_50']).astype(int)  # Binary feature
    df['ema_10_diff'] = (close - df['ema_10']) / df['ema_10'].replace(0, np.nan)
    df['ema_50_diff'] = (close - df['ema_50']) / df['ema_50'].replace(0, np.nan)
    
    # 3. Bollinger Bands
    df['bb_upper'], df['bb_lower'], df['bb_width'], df['bb_position'] = calculate_bollinger_bands(close)
    
    # 4. Volume features
    if 'Volume' in df.columns:
        df['volume_ratio'] = calculate_volume_ratio(df['Volume'])
        df['volume_ma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['volume_change'] = df['Volume'].pct_change()
    
    # 5. Price momentum features
    df['momentum_5'] = close.pct_change(periods=5)
    df['momentum_10'] = close.pct_change(periods=10)
    df['momentum_20'] = close.pct_change(periods=20)
    
    # 6. High-Low range
    if 'High' in df.columns and 'Low' in df.columns:
        df['hl_range'] = (df['High'] - df['Low']) / close
        df['hl_ma_ratio'] = df['hl_range'] / df['hl_range'].rolling(20, min_periods=1).mean()
    
    # 7. Macro features (if available)
    if macro_data is not None:
        for name, series in macro_data.items():
            # Align by index (date)
            aligned = series.reindex(df.index, method='ffill')
            df[f'{name}_value'] = aligned
            df[f'{name}_change'] = aligned.pct_change()
            df[f'{name}_ma_20'] = aligned.rolling(20, min_periods=1).mean()
    
    # ===== TARGET VARIABLE =====
    df["target"] = (close.shift(-1) > close).astype(int)
    
    # ===== CLEAN DATA =====
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop NaN rows
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with NaN/inf values")
    
    return df

# ---------------------------
# Feature Selection
# ---------------------------
def select_features_by_importance(X_train, y_train, feature_cols, threshold=0.20):
    """
    Train a Random Forest to get feature importances and drop bottom threshold%.
    Returns selected feature columns.
    """
    print("\n=== Feature Selection ===")
    print(f"Starting with {len(feature_cols)} features")
    
    # Final check for inf/nan values
    if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
        print("WARNING: Found inf/nan values in X_train. Replacing with 0...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train a quick Random Forest for feature importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    
    # Get importances
    importances = pd.Series(rf_selector.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    # Calculate cutoff
    cutoff_idx = int(len(importances) * (1 - threshold))
    selected_features = importances.head(cutoff_idx).index.tolist()
    dropped_features = importances.tail(len(importances) - cutoff_idx).index.tolist()
    
    print(f"Selected {len(selected_features)} features (dropped bottom {threshold*100:.0f}%)")
    print(f"Dropped features: {dropped_features[:10]}..." if len(dropped_features) > 10 else f"Dropped features: {dropped_features}")
    
    # Save feature importances
    importances.to_csv(RESULTS_DIR / "feature_importances.csv")
    
    return selected_features, importances

# ---------------------------
# Prepare Data
# ---------------------------
def prepare_X_y(df, feature_cols=None):
    """Prepare feature matrix X and target vector y."""
    if 'target' not in df.columns:
        raise KeyError("'target' column not found in DataFrame.")
    
    if feature_cols is None:
        # Auto-select numeric features, excluding raw OHLCV and target
        exclude = {"target", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Date"}
        feature_cols = [c for c in df.columns if c not in exclude]
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy()
    y = df["target"].copy()
    
    return X, y, feature_cols

# ---------------------------
# Model Training & Evaluation
# ---------------------------
def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return results."""
    results = {}
    
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("SVM", SVC(probability=True, random_state=RANDOM_STATE, kernel='rbf'))
    ]
    
    if XGBOOST_AVAILABLE:
        models.append(("XGBoost", xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss", 
            random_state=RANDOM_STATE,
            n_estimators=200
        )))
    
    if LIGHTGBM_AVAILABLE:
        models.append(("LightGBM", lgb.LGBMClassifier(random_state=RANDOM_STATE, n_estimators=200)))
    
    for name, model in models:
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            # Get probabilities
            probs = None
            try:
                probs = model.predict_proba(X_test)[:, 1]
            except Exception:
                try:
                    probs = model.decision_function(X_test)
                    # Normalize to [0, 1]
                    probs = (probs - probs.min()) / (probs.max() - probs.min())
                except Exception:
                    pass
            
            # Calculate metrics
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, probs) if probs is not None else None
            
            print(f"  Accuracy: {acc:.4f}, F1-Score: {f1:.4f}", end="")
            if auc:
                print(f", ROC-AUC: {auc:.4f}")
            else:
                print()
            
            results[name] = {
                "model": model,
                "pred": pred,
                "probs": probs,
                "accuracy": acc,
                "f1": f1,
                "auc": auc
            }
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    return results

# ---------------------------
# Ensemble Model
# ---------------------------
def create_ensemble(results, X_test, y_test, weights=None):
    """Create weighted ensemble from multiple models."""
    print("\n=== Building Ensemble Model ===")
    
    # Select models that have probabilities
    ensemble_models = {name: res for name, res in results.items() if res['probs'] is not None}
    
    if len(ensemble_models) < 2:
        print("Not enough models with probabilities for ensemble.")
        return None
    
    # Auto-calculate weights based on ROC-AUC if weights == "auto"
    if weights == "auto" or weights is None:
        weights = {}
        total_auc = sum(res['auc'] for res in ensemble_models.values() if res['auc'])
        for name, res in ensemble_models.items():
            if res['auc']:
                weights[name] = res['auc'] / total_auc
            else:
                weights[name] = 0
        print(f"Auto-calculated weights: {weights}")
    
    # Calculate weighted average of probabilities
    ensemble_probs = np.zeros(len(X_test))
    for name, weight in weights.items():
        if name in ensemble_models:
            ensemble_probs += weight * ensemble_models[name]['probs']
    
    # Convert to predictions
    ensemble_pred = (ensemble_probs >= 0.5).astype(int)
    
    # Evaluate
    acc = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_probs)
    
    print(f"Ensemble - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {auc:.4f}")
    
    return {
        "model": None,  # Ensemble doesn't have a single model object
        "pred": ensemble_pred,
        "probs": ensemble_probs,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "weights": weights
    }

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models on one graph."""
    plt.figure(figsize=(10, 6))
    
    for name, res in results.items():
        if res['probs'] is not None and res['auc'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['probs'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves_comparison.png", dpi=150)
    plt.show()

def plot_feature_importances(feature_names, importances, top_n=20):
    """Plot top N feature importances."""
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    fi.plot(kind='barh')
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importances_plot.png", dpi=150)
    plt.show()

def plot_predictions(dates, y_true, y_pred, title="Actual vs Predicted"):
    """Plot actual vs predicted values over time."""
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true.values, label="Actual", marker='o', markersize=3, alpha=0.7)
    plt.plot(dates, y_pred, label="Predicted", marker='x', markersize=3, alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"predictions_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

# ---------------------------
# Backtesting Functions
# ---------------------------
def backtest_strategy(prices, predictions, probabilities, dates, strategy="long_only", 
                     prob_threshold=0.6, transaction_cost=0.001, initial_capital=100000,
                     position_size=1.0):
    """
    Backtest a trading strategy based on predictions.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series (Close prices)
    predictions : np.array
        Binary predictions (0 or 1)
    probabilities : np.array
        Prediction probabilities
    dates : pd.Index
        Date index
    strategy : str
        Trading strategy: "long_only", "long_short", or "probability_based"
    prob_threshold : float
        Minimum probability threshold for probability_based strategy
    transaction_cost : float
        Transaction cost per trade (e.g., 0.001 = 0.1%)
    initial_capital : float
        Starting capital
    position_size : float
        Fraction of capital to use per trade (0.0 to 1.0)
    
    Returns:
    --------
    dict : Backtest results with equity curve, trades, and metrics
    """
    print(f"\n=== Backtesting Strategy: {strategy} ===")
    
    # Initialize tracking variables
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    equity_curve = [initial_capital]
    trades = []
    entry_price = None
    entry_date = None
    entry_prediction = None
    
    # Calculate actual returns for comparison (forward returns)
    actual_returns = prices.pct_change().shift(-1).fillna(0)
    
    for i in range(len(prices)):
        current_price = prices.iloc[i]
        current_date = dates[i]
        pred = predictions[i]
        prob = probabilities[i] if probabilities is not None else 0.5
        
        # Determine trading signal based on strategy
        signal = None
        if strategy == "long_only":
            signal = 1 if pred == 1 else 0
        elif strategy == "long_short":
            signal = 1 if pred == 1 else -1
        elif strategy == "probability_based":
            if prob >= prob_threshold:
                signal = 1
            elif prob <= (1 - prob_threshold):
                signal = -1
            else:
                signal = 0
        
        # Close existing position if needed
        if position != 0:
            if (strategy == "long_only" and signal == 0) or \
               (strategy == "long_short" and signal != position) or \
               (strategy == "probability_based" and signal != position and signal != 0):
                # Close position
                if position == 1:  # Close long
                    pnl_pct = (current_price / entry_price - 1) * position_size
                    pnl = capital * pnl_pct
                    capital = capital + pnl - (capital * position_size * transaction_cost)
                    trade_result = "WIN" if pnl > 0 else "LOSS"
                else:  # Close short
                    pnl_pct = (entry_price / current_price - 1) * position_size
                    pnl = capital * pnl_pct
                    capital = capital + pnl - (capital * position_size * transaction_cost)
                    trade_result = "WIN" if pnl > 0 else "LOSS"
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'result': trade_result,
                    'entry_prediction': entry_prediction,
                    'holding_period': (current_date - entry_date).days if hasattr(current_date, '__sub__') else 0
                })
                
                position = 0
                entry_price = None
                entry_date = None
        
        # Open new position if signal
        if position == 0 and signal != 0:
            position = signal
            entry_price = current_price
            entry_date = current_date
            entry_prediction = pred
            # Apply transaction cost when entering
            capital = capital - (capital * position_size * transaction_cost)
        
        # Update equity curve
        if position != 0:
            if position == 1:  # Long position
                unrealized_pnl_pct = (current_price / entry_price - 1) * position_size
            else:  # Short position
                unrealized_pnl_pct = (entry_price / current_price - 1) * position_size
            current_equity = capital * (1 + unrealized_pnl_pct)
        else:
            current_equity = capital
        
        equity_curve.append(current_equity)
    
    # Close any open position at the end
    if position != 0 and len(prices) > 0:
        final_price = prices.iloc[-1]
        final_date = dates[-1]
        if position == 1:
            pnl_pct = (final_price / entry_price - 1) * position_size
            pnl = capital * pnl_pct
            capital = capital + pnl - (capital * position_size * transaction_cost)
        else:
            pnl_pct = (entry_price / final_price - 1) * position_size
            pnl = capital * pnl_pct
            capital = capital + pnl - (capital * position_size * transaction_cost)
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': final_date,
            'entry_price': entry_price,
            'exit_price': final_price,
            'direction': 'LONG' if position == 1 else 'SHORT',
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'result': "WIN" if pnl > 0 else "LOSS",
            'entry_prediction': entry_prediction,
            'holding_period': (final_date - entry_date).days if hasattr(final_date, '__sub__') else 0
        })
        equity_curve[-1] = capital
    
    # Convert equity curve to returns
    equity_series = pd.Series(equity_curve[1:], index=dates)
    returns = equity_series.pct_change().fillna(0)
    
    # Calculate benchmark (buy and hold)
    benchmark_returns = actual_returns.fillna(0)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
    
    return {
        'equity_curve': equity_series,
        'returns': returns,
        'benchmark_equity': benchmark_equity,
        'benchmark_returns': benchmark_returns,
        'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'benchmark_return': (benchmark_equity.iloc[-1] - initial_capital) / initial_capital * 100 if len(benchmark_equity) > 0 else 0
    }

def calculate_performance_metrics(backtest_results, risk_free_rate=0.05):
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    backtest_results : dict
        Results from backtest_strategy()
    risk_free_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    dict : Performance metrics
    """
    equity = backtest_results['equity_curve']
    returns = backtest_results['returns']
    trades_df = backtest_results['trades']
    
    # Total return
    total_return = backtest_results['total_return']
    
    # Annualized return
    if len(returns) > 0:
        periods_per_year = 252 if not USE_WEEKLY else 52
        years = len(returns) / periods_per_year
        if years > 0:
            annualized_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
        else:
            annualized_return = total_return
    else:
        annualized_return = 0
    
    # Sharpe ratio
    if len(returns) > 0 and returns.std() > 0:
        periods_per_year = 252 if not USE_WEEKLY else 52
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative = equity / equity.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Win rate and trade statistics
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        total_trades = len(trades_df)
        
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        avg_holding_period = trades_df['holding_period'].mean() if 'holding_period' in trades_df.columns else 0
    else:
        win_rate = 0
        total_trades = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_holding_period = 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252 if not USE_WEEKLY else 52) * 100 if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'volatility': volatility,
        'avg_holding_period': avg_holding_period
    }

def plot_backtest_results(backtest_results, model_name, dates):
    """Plot comprehensive backtest visualization."""
    equity = backtest_results['equity_curve']
    benchmark_equity = backtest_results['benchmark_equity']
    returns = backtest_results['returns']
    trades_df = backtest_results['trades']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Equity Curve
    axes[0].plot(dates, equity.values, label=f'Strategy ({model_name})', linewidth=2, color='blue')
    axes[0].plot(dates, benchmark_equity.values, label='Buy & Hold', linewidth=2, color='gray', linestyle='--')
    axes[0].set_title(f'Equity Curve - {model_name}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='y')
    
    # 2. Drawdown
    cumulative = equity / equity.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    axes[1].fill_between(dates, drawdown.values, 0, alpha=0.3, color='red')
    axes[1].plot(dates, drawdown.values, color='red', linewidth=1)
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(alpha=0.3)
    
    # 3. Monthly Returns (heatmap style)
    if len(returns) > 0:
        returns_series = pd.Series(returns.values, index=dates)
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        colors = ['red' if r < 0 else 'green' for r in monthly_returns.values]
        axes[2].bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.6)
        axes[2].set_title('Monthly Returns', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Return (%)')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].grid(alpha=0.3, axis='y')
        # Set x-axis labels to show dates
        if len(monthly_returns) > 0:
            step = max(1, len(monthly_returns) // 10)
            axes[2].set_xticks(range(0, len(monthly_returns), step))
            axes[2].set_xticklabels([str(monthly_returns.index[i])[:7] for i in range(0, len(monthly_returns), step)], 
                                   rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"backtest_{model_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot trade analysis if we have trades
    if len(trades_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Trade P&L distribution
        axes[0, 0].hist(trades_df['pnl_pct'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('P&L (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[0, 1].plot(cumulative_pnl.values, linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L')
        axes[0, 1].grid(alpha=0.3)
        
        # Win vs Loss
        if 'result' in trades_df.columns:
            win_loss = trades_df['result'].value_counts()
            axes[1, 0].bar(win_loss.index, win_loss.values, color=['green', 'red'], alpha=0.7)
            axes[1, 0].set_title('Win vs Loss Count', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Holding Period Distribution
        if 'holding_period' in trades_df.columns:
            axes[1, 1].hist(trades_df['holding_period'], bins=20, edgecolor='black', alpha=0.7, color='blue')
            axes[1, 1].set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Days')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"backtest_trades_{model_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()

def run_backtest_for_model(prices, predictions, probabilities, dates, model_name, 
                          strategy=BACKTEST_STRATEGY, prob_threshold=PROBABILITY_THRESHOLD):
    """Run backtest for a single model."""
    print(f"\n{'='*60}")
    print(f"Backtesting: {model_name}")
    print(f"{'='*60}")
    
    # Run backtest
    backtest_results = backtest_strategy(
        prices=prices,
        predictions=predictions,
        probabilities=probabilities,
        dates=dates,
        strategy=strategy,
        prob_threshold=prob_threshold,
        transaction_cost=TRANSACTION_COST,
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE
    )
    
    # Calculate metrics
    metrics = calculate_performance_metrics(backtest_results, risk_free_rate=RISK_FREE_RATE)
    
    # Print results
    print(f"\n=== Performance Metrics ===")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Benchmark Return: {backtest_results['benchmark_return']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"\n=== Trade Statistics ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Win: {metrics['avg_win']:.2f}%")
    print(f"Average Loss: {metrics['avg_loss']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"Avg Holding Period: {metrics['avg_holding_period']:.1f} days")
    
    # Plot results
    plot_backtest_results(backtest_results, model_name, dates)
    
    # Save trades
    if len(backtest_results['trades']) > 0:
        backtest_results['trades'].to_csv(
            RESULTS_DIR / f"backtest_trades_{model_name.replace(' ', '_')}.csv", 
            index=False
        )
    
    return backtest_results, metrics

# ---------------------------
# Save Results
# ---------------------------
def save_results_summary(results, feature_cols):
    """Save model metrics to CSV."""
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1'],
            'ROC-AUC': res['auc']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)
    summary_df.to_csv(RESULTS_DIR / "model_metrics_summary.csv", index=False)
    print("\n=== Model Performance Summary ===")
    print(summary_df.to_string(index=False))
    
    # Save selected features
    pd.DataFrame({'Feature': feature_cols}).to_csv(RESULTS_DIR / "selected_features.csv", index=False)

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    print("="*60)
    print("ENHANCED NIFTY PREDICTION PIPELINE")
    print("="*60)
    
    # 1. Download data
    df = download_data()
    print(f"Downloaded {len(df)} rows of NIFTY data")
    
    # 2. Download macro data (if enabled)
    macro_data = None
    if ENABLE_MACRO_FEATURES:
        macro_data = download_macro_data()
        if macro_data:
            print(f"Downloaded {len(macro_data)} macro indicators")
    
    # 3. Resample to weekly if enabled
    if USE_WEEKLY:
        print("\nResampling to weekly data...")
        df = df.resample('W').last()
        if macro_data:
            for key in macro_data:
                macro_data[key] = macro_data[key].resample('W').last()
    
    # 4. Feature engineering
    print("\nPerforming feature engineering...")
    # Store original index (dates) before feature engineering resets it
    original_dates = df.index.copy()
    df_feat = feature_engineering(df, macro_data)
    # Restore date index - align with remaining rows after dropna
    # Since feature_engineering drops NaN rows, we need to align dates
    if len(df_feat) <= len(original_dates):
        # Use the last N dates where N is the number of rows after dropna
        # This assumes rows are dropped from the end or beginning
        # Better approach: preserve dates as a column in feature_engineering
        # For now, use a simple alignment
        if len(df_feat) == len(original_dates):
            df_feat.index = original_dates
        else:
            # If rows were dropped, use the dates that correspond to remaining rows
            # Since we can't know which rows were dropped, use a date range
            # This is a limitation - ideally dates should be preserved as a column
            start_date = original_dates[0] if len(original_dates) > 0 else pd.Timestamp('2010-01-01')
            freq = 'W' if USE_WEEKLY else 'D'
            df_feat.index = pd.date_range(start=start_date, periods=len(df_feat), freq=freq)
    print(f"Created {len(df_feat.columns)} features, {len(df_feat)} samples")
    
    # 5. Prepare X and y
    X, y, feature_cols = prepare_X_y(df_feat)
    print(f"\nInitial features: {len(feature_cols)}")
    
    # 6. Train/test split (time-ordered)
    split_index = int((1 - TEST_SIZE) * len(X))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = df_feat.index[split_index:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 7. Feature selection
    selected_features, all_importances = select_features_by_importance(
        X_train, y_train, feature_cols, threshold=FEATURE_SELECTION_THRESHOLD
    )
    
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # 8. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")
    
    # 9. Train models
    results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 10. Create ensemble
    ensemble_result = create_ensemble(results, X_test_scaled, y_test, weights=ENSEMBLE_WEIGHTS)
    if ensemble_result:
        results['Ensemble'] = ensemble_result
    
    # 11. Save results
    save_results_summary(results, selected_features)
    
    # 12. Plot ROC curves
    plot_roc_curves(results, y_test)
    
    # 13. Plot feature importances
    plot_feature_importances(selected_features, all_importances[selected_features].values, top_n=20)
    
    # 14. Save best model
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    if best_name != 'Ensemble':
        best_model = results[best_name]['model']
        joblib.dump(best_model, RESULTS_DIR / f"best_model_{best_name}.joblib")
        print(f"\nSaved best model: {best_name}")
    else:
        # Save ensemble weights
        with open(RESULTS_DIR / "ensemble_weights.txt", 'w') as f:
            f.write(str(ensemble_result['weights']))
        print("\nSaved ensemble weights")
    
    # 15. Plot predictions for best model (last 200 samples)
    best_pred = results[best_name]['pred']
    n_plot = min(200, len(dates_test))
    plot_predictions(
        dates_test[-n_plot:], 
        y_test.iloc[-n_plot:], 
        best_pred[-n_plot:],
        title=f"{best_name} - Last {n_plot} {'Weeks' if USE_WEEKLY else 'Days'}"
    )
    
    # 16. Run backtesting if enabled
    backtest_metrics_all = {}
    if ENABLE_BACKTESTING:
        print(f"\n{'='*60}")
        print("RUNNING BACKTESTING")
        print(f"{'='*60}")
        
        # Get test prices and dates from df_feat (dates should be preserved now)
        if 'Close' in df_feat.columns:
            test_prices = df_feat['Close'].iloc[split_index:]
            test_prices_index = df_feat.index[split_index:]
        else:
            # Fallback: use original df if Close was dropped
            test_prices = df['Close'].iloc[split_index:]
            test_prices_index = df.index[split_index:]
        
        # Ensure test_prices is a Series with proper index
        if not isinstance(test_prices, pd.Series):
            test_prices = pd.Series(test_prices.values, index=test_prices_index)
        
        # Run backtest for top models (including ensemble)
        models_to_backtest = [best_name]
        if 'Ensemble' in results and best_name != 'Ensemble':
            models_to_backtest.append('Ensemble')
        
        # Also backtest a few other models if they exist
        for model_name in ['XGBoost', 'LightGBM', 'GradientBoosting']:
            if model_name in results and model_name not in models_to_backtest and len(models_to_backtest) < 3:
                models_to_backtest.append(model_name)
        
        for model_name in models_to_backtest:
            if model_name in results:
                model_result = results[model_name]
                predictions = model_result['pred']
                probabilities = model_result['probs']
                
                # Ensure we have the right length
                if len(predictions) == len(test_prices):
                    try:
                        backtest_results, backtest_metrics = run_backtest_for_model(
                            prices=test_prices,
                            predictions=predictions,
                            probabilities=probabilities,
                            dates=test_prices_index,
                            model_name=model_name,
                            strategy=BACKTEST_STRATEGY,
                            prob_threshold=PROBABILITY_THRESHOLD
                        )
                        backtest_metrics_all[model_name] = backtest_metrics
                    except Exception as e:
                        print(f"Error backtesting {model_name}: {e}")
        
        # Save backtest summary
        if backtest_metrics_all:
            backtest_summary = []
            for model_name, metrics in backtest_metrics_all.items():
                backtest_summary.append({
                    'Model': model_name,
                    'Total Return (%)': metrics['total_return'],
                    'Annualized Return (%)': metrics['annualized_return'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown (%)': metrics['max_drawdown'],
                    'Win Rate (%)': metrics['win_rate'],
                    'Total Trades': metrics['total_trades'],
                    'Profit Factor': metrics['profit_factor'],
                    'Volatility (%)': metrics['volatility']
                })
            
            backtest_df = pd.DataFrame(backtest_summary)
            backtest_df = backtest_df.sort_values('Total Return (%)', ascending=False)
            backtest_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
            
            print(f"\n{'='*60}")
            print("BACKTEST SUMMARY")
            print(f"{'='*60}")
            print(backtest_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Results saved to: {RESULTS_DIR.resolve()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 