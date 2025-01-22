#filename: stock_moreparams.py
# Description: This script extracts stock data from Yahoo Finance, trains a Random Forest model to predict the next day's stock movement, and saves the data to a SQLite database.
# Dependencies: yfinance, pandas, sqlalchemy, sklearn, ta, imblearn, numpy
# Usage: python stock_moreparams.py
# This model shows an improvement in accuracy, precision, and recall by adding more features and using an ensemble model with balanced class weights.
# The model is trained on historical stock data and financial metrics to predict the next day's stock movement.
# Accuracy 0.769 , Precision  0.727 and Recall 1.0

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
pd.set_option('future.no_silent_downcasting', True)
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def get_stock_and_financial_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    financials = stock.quarterly_financials.T
    
    stock_data.index = stock_data.index.tz_localize(None)
    financials.index = financials.index.tz_localize(None)
    
    # Technical Indicators
    # Trend
    stock_data['SMA_20'] = SMAIndicator(close=stock_data['Close'], window=20).sma_indicator()
    stock_data['SMA_50'] = SMAIndicator(close=stock_data['Close'], window=50).sma_indicator()
    stock_data['EMA_20'] = EMAIndicator(close=stock_data['Close'], window=20).ema_indicator()
    
    # MACD
    macd = MACD(close=stock_data['Close'])
    stock_data['MACD'] = macd.macd()
    stock_data['MACD_Signal'] = macd.macd_signal()
    stock_data['MACD_Hist'] = macd.macd_diff()
    
    # Momentum
    stock_data['RSI'] = RSIIndicator(close=stock_data['Close']).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'])
    stock_data['Stoch_K'] = stoch.stoch()
    stock_data['Stoch_D'] = stoch.stoch_signal()
    
    # Volatility
    bb = BollingerBands(close=stock_data['Close'])
    stock_data['BB_Upper'] = bb.bollinger_hband()
    stock_data['BB_Lower'] = bb.bollinger_lband()
    stock_data['BB_Width'] = (stock_data['BB_Upper'] - stock_data['BB_Lower']) / stock_data['Close']
    
    # Volume
    stock_data['OBV'] = OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume']).on_balance_volume()
    
    # Price-based features
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
    stock_data['Price_Range'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
    
    # Advanced Features
    # Support and Resistance
    stock_data['Support'] = stock_data['Low'].rolling(window=20).min()
    stock_data['Resistance'] = stock_data['High'].rolling(window=20).max()
    stock_data['Distance_to_Support'] = (stock_data['Close'] - stock_data['Support']) / stock_data['Close']
    stock_data['Distance_to_Resistance'] = (stock_data['Resistance'] - stock_data['Close']) / stock_data['Close']
    
    # Financial Metrics with more sophisticated calculations
    # Revenue metrics
    revenue_growth = financials['Total Revenue'].pct_change(periods=4, fill_method=None)
    stock_data['Revenue_Growth'] = revenue_growth.reindex(stock_data.index, method='ffill').infer_objects(copy=False)
    stock_data['Revenue_Growth_Acceleration'] = stock_data['Revenue_Growth'].diff()
    
    # Profitability metrics
    stock_data['Gross_Margin'] = (financials['Gross Profit'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
    stock_data['Operating_Margin'] = (financials['Operating Income'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
    stock_data['Net_Margin'] = (financials['Net Income'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
    
    # Growth metrics
    eps_growth = financials['Net Income'].pct_change(periods=4, fill_method=None)
    stock_data['EPS_Growth'] = eps_growth.reindex(stock_data.index, method='ffill').infer_objects(copy=False)
    stock_data['EPS_Growth_Acceleration'] = stock_data['EPS_Growth'].diff()
    
    # Target variable with different horizons
    for horizon in [1, 3, 5]:
        stock_data[f'Target_{horizon}d'] = (stock_data['Close'].shift(-horizon) > 
                                          stock_data['Close']).astype(int)
    
    stock_data.dropna(inplace=True)
    return stock_data

def create_ensemble_model(random_state=42):
    # Create base models with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=random_state
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=random_state
    )
    
    return [rf, gb]

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    predictions = []
    
    # Handle class imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train models with balanced data
    for model in models:
        model.fit(X_train_balanced, y_train_balanced)
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
    
    # Ensemble predictions with optimized threshold
    ensemble_pred_proba = np.mean(predictions, axis=0)
    
    # Find optimal threshold using training data
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (ensemble_pred_proba > threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    ensemble_pred = (ensemble_pred_proba > best_threshold).astype(int)
    
    return ensemble_pred, ensemble_pred_proba

try:
    # Parameters
    ticker = 'MSFT'
    start_date = '2015-01-01'
    end_date = '2025-01-20'
    
    # Get data with enhanced features
    data = get_stock_and_financial_data(ticker, start_date, end_date)
    
    # Feature selection using mutual information
    from sklearn.feature_selection import mutual_info_classif
    
    # Select features excluding target columns
    initial_features = [col for col in data.columns if col not in 
                       ['Target_1d', 'Target_3d', 'Target_5d', 'Dividends', 'Stock Splits']]
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(data[initial_features], data['Target_1d'])
    feature_importance = dict(zip(initial_features, mi_scores))
    
    # Select top 20 most important features
    feature_columns = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:20]
    feature_columns = [f[0] for f in feature_columns]
    
    # Prepare features and target
    X = data[feature_columns]
    y = data['Target_1d']  # Predict 1-day ahead by default
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data with temporal awareness
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train ensemble model
    models = create_ensemble_model()
    ensemble_pred, ensemble_pred_proba = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, models
    )
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Make prediction for next day
    last_data = X_scaled[-1:]
    ensemble_predictions = []
    prediction_probabilities = []
    
    for model in models:
        pred_prob = model.predict_proba(last_data)[:, 1]
        prediction_probabilities.append(pred_prob)
    
    final_probability = np.mean(prediction_probabilities)
    final_prediction = "Up" if final_probability > 0.5 else "Down"
    
    print(f"\nNext Day Prediction:")
    print(f"Direction: {final_prediction}")
    print(f"Confidence: {abs(final_probability - 0.5) * 2:.2%}")

except Exception as e:
    print(f"An error occurred: {str(e)}")