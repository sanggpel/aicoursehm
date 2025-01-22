#filename: stock_price_prediction.py
import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import ta
from imblearn.over_sampling import SMOTE
import logging
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('future.no_silent_downcasting', True)

def get_stock_and_financial_data(ticker, start_date, end_date):
    try:
        # Get main stock data
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date)
        financials = stock.quarterly_financials.T
        
        # Get market index (S&P 500) for market context
        spy = yf.Ticker("SPY")
        market_data = spy.history(start=start_date, end=end_date)
        
        # Standardize timezone handling
        stock_data.index = stock_data.index.tz_localize(None)
        financials.index = financials.index.tz_localize(None)
        market_data.index = market_data.index.tz_localize(None)
        
        # Price Pattern Features
        stock_data['Higher_High'] = (stock_data['High'] > stock_data['High'].shift(1)) & \
                                   (stock_data['High'].shift(1) > stock_data['High'].shift(2))
        stock_data['Lower_Low'] = (stock_data['Low'] < stock_data['Low'].shift(1)) & \
                                 (stock_data['Low'].shift(1) < stock_data['Low'].shift(2))
        
        # Multi-timeframe Analysis
        for window in [5, 10, 20, 50]:
            # Price momentum
            stock_data[f'ROC_{window}'] = ((stock_data['Close'] - stock_data['Close'].shift(window)) / 
                                          stock_data['Close'].shift(window)) * 100
            
            # Volume momentum
            stock_data[f'Volume_ROC_{window}'] = ((stock_data['Volume'] - stock_data['Volume'].shift(window)) / 
                                                 stock_data['Volume'].shift(window)) * 100
            
            # Moving averages
            stock_data[f'SMA_{window}'] = ta.trend.SMAIndicator(close=stock_data['Close'], 
                                                               window=window).sma_indicator()
            stock_data[f'EMA_{window}'] = ta.trend.EMAIndicator(close=stock_data['Close'], 
                                                               window=window).ema_indicator()
            
            # Trend strength
            stock_data[f'ADX_{window}'] = ta.trend.ADXIndicator(
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                window=window
            ).adx()
        
        # Enhanced Volatility Metrics
        atr = ta.volatility.AverageTrueRange(
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )
        stock_data['ATR'] = atr.average_true_range()
        stock_data['ATR_Pct'] = stock_data['ATR'] / stock_data['Close']
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )
        stock_data['KC_Upper'] = keltner.keltner_channel_hband()
        stock_data['KC_Lower'] = keltner.keltner_channel_lband()
        stock_data['KC_Width'] = (stock_data['KC_Upper'] - stock_data['KC_Lower']) / stock_data['Close']
        
        # Basic Returns calculations
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Market_Return'] = market_data['Close'].pct_change()
        
        # Market Context Features
        stock_data['Beta'] = stock_data['Returns'].rolling(window=60).cov(stock_data['Market_Return']) / \
                            stock_data['Market_Return'].rolling(window=60).var()
        stock_data['Relative_Strength'] = stock_data['Close'] / market_data['Close']
        stock_data['RS_Momentum'] = stock_data['Relative_Strength'].pct_change(periods=20)
        
        # Financial Metrics
        revenue_growth = financials['Total Revenue'].pct_change(periods=4, fill_method=None)
        stock_data['Revenue_Growth'] = revenue_growth.reindex(stock_data.index, method='ffill').infer_objects(copy=False)
        stock_data['Revenue_Growth_Acceleration'] = stock_data['Revenue_Growth'].diff()
        
        stock_data['Gross_Margin'] = (financials['Gross Profit'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
        stock_data['Operating_Margin'] = (financials['Operating Income'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
        stock_data['Net_Margin'] = (financials['Net Income'] / financials['Total Revenue']).reindex(stock_data.index, method='ffill')
        
        eps_growth = financials['Net Income'].pct_change(periods=4, fill_method=None)
        stock_data['EPS_Growth'] = eps_growth.reindex(stock_data.index, method='ffill').infer_objects(copy=False)
        stock_data['EPS_Growth_Acceleration'] = stock_data['EPS_Growth'].diff()
        
        # Calculate target
        stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
        
        stock_data.dropna(inplace=True)
        return stock_data
        
    except Exception as e:
        logger.error(f"Error in get_stock_and_financial_data: {str(e)}")
        raise

def create_ensemble_model(random_state=42):
    # Create base models with enhanced parameters
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.8,
        max_features='sqrt',
        random_state=random_state
    )
    
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    return [rf, gb, xgb]

def predict_price_targets(data, current_price, final_prediction, model_confidence):
    """Calculate price targets and timeframes based on historical volatility and trends"""
    
    # Calculate historical volatility
    returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate average daily movement
    avg_daily_move = abs(data['Close'].pct_change()).mean()
    
    # Calculate trend strength using ADX
    adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx()
    trend_strength = adx.iloc[-1] / 100  # Normalize to 0-1
    
    # Calculate short-term momentum
    momentum_5d = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1)
    momentum_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1)
    
    # Define timeframes and calculate targets
    timeframes = [5, 10, 20]
    targets = {}
    
    for days in timeframes:
        # Calculate price range based on volatility
        confidence_factor = model_confidence  # Use actual model confidence
        daily_vol = volatility / np.sqrt(252)
        price_range = daily_vol * np.sqrt(days) * current_price * confidence_factor        

        # Adjust based on trend strength and momentum
        trend_adjustment = trend_strength * avg_daily_move * days
        momentum_adjustment = (momentum_5d + momentum_20d) / 2 * price_range
        
        if final_prediction == "Up":
            upper_target = current_price + price_range + trend_adjustment + momentum_adjustment
            lower_target = current_price + (price_range * 0.3)  # Support level
        else:
            upper_target = current_price - (price_range * 0.3)  # Resistance level
            lower_target = current_price - price_range - trend_adjustment - momentum_adjustment
            
        targets[days] = {
            'upper': upper_target,
            'lower': lower_target,
            'confidence': model_confidence,
            'avg_daily_move': avg_daily_move
        }
    
    return targets

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    predictions = []
    feature_importances = {}
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train models and collect feature importances
    for model in models:
        model.fit(X_train_balanced, y_train_balanced)
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances[model.__class__.__name__] = model.feature_importances_
    
    # Ensemble predictions with confidence-based threshold
    ensemble_pred_proba = np.mean(predictions, axis=0)
    
    # Dynamic threshold based on prediction confidence
    confidence_threshold = np.percentile(ensemble_pred_proba, 60)  # More conservative threshold
    ensemble_pred = (ensemble_pred_proba > confidence_threshold).astype(int)
    
    # Additional confidence filter
    confidence_mask = np.abs(ensemble_pred_proba - 0.5) > 0.15  # Only keep high-confidence predictions
    ensemble_pred[~confidence_mask] = 0  # Set low-confidence predictions to 0 (no trade)
    
    return ensemble_pred, ensemble_pred_proba, feature_importances, confidence_threshold

try:
    # Parameters
    ticker = 'MSFT'
    start_date = '2015-01-01'
    end_date = '2025-01-20'
    print(f"Running stock price prediction for {ticker} from {start_date} to {end_date}...")
    # Get and process data
    data = get_stock_and_financial_data(ticker, start_date, end_date)
    
    # Feature selection using mutual information
    from sklearn.feature_selection import mutual_info_classif
    
    # Select initial features
    initial_features = [col for col in data.columns if col not in 
                       ['Target_1d', 'Target_3d', 'Target_5d', 'Dividends', 'Stock Splits']]
    
    # Handle NaN values before feature selection
    X_temp = data[initial_features].fillna(0)
    y_temp = data['Target']
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X_temp, y_temp)
    feature_importance = dict(zip(initial_features, mi_scores))
    
    # Select top 20 most important features
    feature_columns = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:20]
    feature_columns = [f[0] for f in feature_columns]
    
    # Prepare features and target
    X = data[feature_columns]
    y = data['Target']  # Predict 1-day ahead
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data with temporal awareness
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train ensemble model
    models = create_ensemble_model()
    ensemble_pred, ensemble_pred_proba, feature_importances, confidence_threshold = train_and_evaluate_models(
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
    
    # Print feature importances
    print("\nTop 5 Most Important Features:")
    for model_name, importances in feature_importances.items():
        feature_imp = dict(zip(feature_columns, importances))
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n{model_name}:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.3f}")
    
    # Make prediction for next day
    last_data = X_scaled[-1:]
    ensemble_predictions = []
    prediction_probabilities = []
    
    print(f"Stock price prediction for {ticker}...")
    for model in models:
        pred_prob = model.predict_proba(last_data)[:, 1]
        prediction_probabilities.append(pred_prob)
    
    final_probability = np.mean(prediction_probabilities)
    final_prediction = "Up" if final_probability > confidence_threshold else "Down"
    
    # Get current price
    current_price = data['Close'].iloc[-1]
    
    # Calculate price targets
    # Calculate model confidence from prediction probability
    model_confidence = abs(final_probability - 0.5) * 2  # Convert to 0-1 scale
    price_targets = predict_price_targets(
        data=data,
        current_price=current_price,
        final_prediction=final_prediction,
        model_confidence=model_confidence
    )    
    print(f"\nCurrent Price: ${current_price:.2f}")
    print("\nPrice Targets:")
    for days, targets in price_targets.items():
        print(f"\n{days}-Day Forecast:")
        print(f"Upper Target: ${targets['upper']:.2f}")
        print(f"Lower Target: ${targets['lower']:.2f}")
        print(f"Model Confidence Level: {targets['confidence']*100:.0f}%")
        
        # Calculate expected timeline for targets
        if final_prediction == "Up":
            target_price = targets['upper']
            move_direction = "upward"
        else:
            target_price = targets['lower']
            move_direction = "downward"
            
        price_move = abs(target_price - current_price)
        avg_daily_move = abs(data['Close'].pct_change()).mean() * current_price
        expected_days = int(price_move / avg_daily_move)
        
        print(f"Expected timeline for {move_direction} move: {expected_days} trading days")

except Exception as e:
    print(f"An error occurred: {str(e)}")