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

def calculate_dynamic_market_regime(data):
    """
    Calculate a dynamic market regime multiplier based on multiple technical factors
    Returns a multiplier between 0.8 and 1.1
    """
    # Get latest data
    current_price = data['Close'].iloc[-1]
    
    # Moving average relationships
    sma_50 = data['SMA_50'].iloc[-1]
    sma_200 = data['SMA_200'].iloc[-1]
    ema_20 = data['EMA_20'].iloc[-1]
    
    # Calculate various strength indicators
    price_ma_strength = (current_price / sma_200 - 1)  # Price strength vs 200 SMA
    ma_trend_strength = (sma_50 / sma_200 - 1)  # 50 vs 200 SMA relationship
    short_term_strength = (current_price / ema_20 - 1)  # Short-term momentum
    
    # Volume analysis
    volume_ratio = data['Volume'].iloc[-20:].mean() / data['Volume'].iloc[-100:].mean()
    volume_factor = min(1.05, max(0.95, volume_ratio))
    
    # Momentum indicators
    roc_20 = data['ROC_20'].iloc[-1] / 100  # Convert ROC to decimal
    roc_50 = data['Close'].pct_change(50).iloc[-1]  # 50-day ROC
    
    # Trend strength from ADX
    adx = data['ADX_20'].iloc[-1] / 100  # Normalize ADX to 0-1
    
    # Volatility consideration
    volatility = data['ATR_Pct'].iloc[-1]
    volatility_factor = 1 - (volatility * 2)  # Reduce multiplier in high volatility
    
    # Combine all factors with weights
    base_multiplier = 1.0
    weights = {
        'price_strength': 0.20,
        'trend_strength': 0.20,
        'short_term': 0.15,
        'volume': 0.10,
        'momentum': 0.15,
        'adx': 0.10,
        'volatility': 0.10
    }
    
    adjusted_multiplier = base_multiplier + (
        (price_ma_strength * weights['price_strength']) +
        (ma_trend_strength * weights['trend_strength']) +
        (short_term_strength * weights['short_term']) +
        ((volume_factor - 1) * weights['volume']) +
        ((roc_20 + roc_50) / 2 * weights['momentum']) +
        (adx * weights['adx']) +
        (volatility_factor * weights['volatility'])
    )
    
    # Calculate trend consistency
    up_days = sum(data['Returns'].iloc[-20:] > 0)
    trend_consistency = up_days / 20
    
    # Final adjustments based on trend consistency
    if trend_consistency > 0.7:  # Strong uptrend
        final_multiplier = min(1.1, adjusted_multiplier * 1.05)
    elif trend_consistency < 0.3:  # Strong downtrend
        final_multiplier = max(0.8, adjusted_multiplier * 0.95)
    else:
        final_multiplier = adjusted_multiplier
    
    # Ensure multiplier stays within reasonable bounds
    final_multiplier = max(0.8, min(1.1, final_multiplier))
    
    # Calculate confidence in the regime assessment
    regime_confidence = {
        'price_trend': 1 if current_price > sma_200 else 0,
        'ma_trend': 1 if sma_50 > sma_200 else 0,
        'momentum': 1 if roc_20 > 0 else 0,
        'volume': 1 if volume_factor > 1 else 0,
        'trend_strength': 1 if adx > 0.25 else 0
    }
    
    confidence_score = sum(regime_confidence.values()) / len(regime_confidence)
    
    regime_characteristics = {
        'multiplier': final_multiplier,
        'confidence': confidence_score,
        'trend_consistency': trend_consistency,
        'volatility_factor': volatility_factor,
        'volume_factor': volume_factor,
        'trend_strength': adx
    }
    
    return regime_characteristics

def apply_market_regime_to_prediction(base_prediction, regime_chars, timeframe):
    """
    Apply market regime characteristics to modify price predictions
    """
    # Adjust multiplier based on timeframe
    timeframe_factor = 1 - (timeframe / 360)  # Reduce impact for longer timeframes
    effective_multiplier = 1 + ((regime_chars['multiplier'] - 1) * timeframe_factor)
    
    # Apply multiplier with confidence weighting
    confidence_adjusted_mult = 1 + ((effective_multiplier - 1) * regime_chars['confidence'])
    
    # Calculate final adjusted prediction
    adjusted_prediction = base_prediction * confidence_adjusted_mult
    
    return adjusted_prediction, confidence_adjusted_mult

def get_long_term_features(data):
    """Add long-term technical and fundamental features"""
    # Long-term moving averages
    for window in [100, 150, 200]:
        data[f'SMA_{window}'] = ta.trend.SMAIndicator(
            close=data['Close'], 
            window=window
        ).sma_indicator()
        
        data[f'EMA_{window}'] = ta.trend.EMAIndicator(
            close=data['Close'], 
            window=window
        ).ema_indicator()
    
    # Long-term trend indicators
    data['Bull_Market'] = data['Close'] > data['SMA_200']
    data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    data['Death_Cross'] = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    
    # Quarterly and yearly momentum
    data['Momentum_90d'] = data['Close'].pct_change(90)
    data['Momentum_180d'] = data['Close'].pct_change(180)
    data['Momentum_360d'] = data['Close'].pct_change(360)
    
    return data

def predict_long_term_targets(data, current_price, models, X_scaled, timeframes=[30, 60, 90]):
    """Generate long-term price predictions with dynamic market regime adjustment"""
    
    predictions = {}
    
    # Calculate market regime characteristics
    regime_chars = calculate_dynamic_market_regime(data)
    
    for timeframe in timeframes:
        # Calculate long-term volatility
        long_term_volatility = data['Returns'].rolling(window=timeframe).std() * np.sqrt(252)
        current_volatility = long_term_volatility.iloc[-1]
        
        # Get ensemble predictions
        ensemble_predictions = []
        confidence_scores = []
        
        for model in models:
            pred_proba = model.predict_proba(X_scaled[-1:])[:, 1]
            ensemble_predictions.append(pred_proba[0])
            
            if hasattr(model, 'feature_importances_'):
                confidence_scores.append(np.max(model.feature_importances_))
        
        # Calculate base prediction
        avg_prediction = np.mean(ensemble_predictions)
        base_confidence = np.mean(confidence_scores) if confidence_scores else 0.6
        
        # Adjust confidence based on timeframe
        timeframe_factor = 1 - (timeframe / 360)  # Reduce confidence for longer timeframes
        adjusted_confidence = base_confidence * timeframe_factor
        
        # Calculate base price target
        if avg_prediction > 0.5:  # Bullish prediction
            volatility_factor = current_volatility * np.sqrt(timeframe/252)
            base_target = current_price * (1 + volatility_factor)
            
            # Apply market regime adjustment
            target_price, effective_multiplier = apply_market_regime_to_prediction(
                base_target, 
                regime_chars, 
                timeframe
            )
            
            # Calculate support levels with regime consideration
            support_level = current_price * (1 + volatility_factor * 0.3 * effective_multiplier)
            
        else:  # Bearish prediction
            volatility_factor = current_volatility * np.sqrt(timeframe/252)
            base_target = current_price * (1 - volatility_factor)
            
            # Apply market regime adjustment
            target_price, effective_multiplier = apply_market_regime_to_prediction(
                base_target, 
                regime_chars, 
                timeframe
            )
            
            # Calculate resistance levels
            support_level = current_price * (1 - volatility_factor * 0.3 * effective_multiplier)
        
        # Final confidence calculation
        final_confidence = (
            adjusted_confidence * 
            regime_chars['confidence'] * 
            (1 - current_volatility * 0.3) *  # Reduce confidence in high volatility
            (0.7 + regime_chars['trend_consistency'] * 0.3)  # Add trend consistency impact
        )
        
        # Risk ratio calculation
        if avg_prediction > 0.5:
            risk_ratio = (target_price - current_price) / (current_price - support_level)
        else:
            risk_ratio = (current_price - target_price) / (support_level - current_price)
        
        predictions[timeframe] = {
            'target_price': target_price,
            'direction': 'Up' if avg_prediction > 0.5 else 'Down',
            'confidence': final_confidence,
            'support_resistance': support_level,
            'risk_ratio': risk_ratio,
            'volatility': current_volatility,
            'trend_strength': regime_chars['trend_strength'],
            'market_regime': 'Bullish' if regime_chars['multiplier'] > 1 else 'Bearish',
            'regime_multiplier': regime_chars['multiplier'],
            'volume_quality': regime_chars['volume_factor'],
            'trend_consistency': regime_chars['trend_consistency']
        }
    
    return predictions

def analyze_prediction_quality(predictions, data, timeframes=[30, 60, 90]):
    """Analyze the quality and reliability of long-term predictions"""
    analysis = {}
    
    for timeframe in timeframes:
        # Calculate historical accuracy for similar timeframes
        historical_predictions = []
        actual_movements = []
        
        for i in range(len(data) - timeframe):
            pred = data['Close'].iloc[i:i+timeframe].mean() > data['Close'].iloc[i]
            actual = data['Close'].iloc[i+timeframe] > data['Close'].iloc[i]
            historical_predictions.append(pred)
            actual_movements.append(actual)
        
        accuracy = np.mean(np.array(historical_predictions) == np.array(actual_movements))
        
        # Calculate volatility stability
        vol_stability = 1 - data['Returns'].rolling(timeframe).std().std()
        
        # Calculate trend consistency
        trend_changes = np.sum(np.diff(data['Bull_Market'].astype(int)))
        trend_consistency = 1 - (trend_changes / len(data))
        
        analysis[timeframe] = {
            'historical_accuracy': accuracy,
            'volatility_stability': vol_stability,
            'trend_consistency': trend_consistency
        }
    
    return analysis

def create_enhanced_ensemble_model(random_state=42):
    """Enhanced ensemble model with better hyperparameters and additional features"""
    rf = RandomForestClassifier(
        n_estimators=1500,  # Increased from 1000
        max_depth=15,       # Increased from 12
        min_samples_split=6,# Reduced from 8
        min_samples_leaf=3, # Reduced from 4
        class_weight='balanced_subsample',
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1           # Utilize all CPU cores
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=1500,
        learning_rate=0.008,  # Reduced from 0.01 for better generalization
        max_depth=8,         # Increased from 6
        min_samples_split=6,
        min_samples_leaf=3,
        subsample=0.85,      # Slightly increased from 0.8
        max_features='sqrt',
        random_state=random_state
    )
    
    xgb = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.008,
        max_depth=8,
        min_child_weight=3,  # Reduced from 4
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        eval_metric='logloss',
        tree_method='hist'    # Faster training
    )
    
    return [rf, gb, xgb]

def adaptive_confidence_threshold(ensemble_pred_proba, market_volatility, trend_strength):
    """Dynamically adjust confidence threshold based on market conditions"""
    base_threshold = 0.6  # Base threshold
    
    # Adjust threshold based on volatility
    volatility_factor = 1 - (market_volatility * 0.5)  # Reduce threshold when volatility is high
    
    # Adjust threshold based on trend strength
    trend_factor = 1 + (trend_strength * 0.2)  # Increase threshold in strong trends
    
    # Calculate final threshold
    adjusted_threshold = base_threshold * volatility_factor * trend_factor
    
    # Keep threshold within reasonable bounds
    return max(0.55, min(0.75, adjusted_threshold))

def calculate_prediction_confidence(predictions, historical_accuracy, market_conditions):
    """Calculate enhanced prediction confidence score"""
    # Base confidence from model predictions
    base_confidence = np.mean(predictions)
    
    # Adjust based on historical accuracy
    accuracy_weight = 0.3
    accuracy_adjusted = base_confidence * (1 + (historical_accuracy - 0.5) * accuracy_weight)
    
    # Market conditions adjustment
    market_weight = 0.2
    market_adjustment = 1 + (market_conditions['trend_strength'] * market_weight)
    
    # Volatility penalty
    volatility_penalty = max(0, 1 - market_conditions['volatility'] * 0.5)
    
    # Calculate final confidence
    final_confidence = accuracy_adjusted * market_adjustment * volatility_penalty
    
    # Normalize to 0-1 range
    return max(0, min(1, final_confidence))

def predict_price_targets_enhanced(data, current_price, final_prediction, model_confidence):
    """Enhanced price target prediction with improved confidence calculation"""
    # Calculate historical volatility with exponential weighting
    returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)
    
    # Calculate trend strength using multiple indicators
    adx = ta.trend.ADXIndicator(
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    ).adx().iloc[-1] / 100
    
    # Calculate momentum scores
    momentum_5d = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1)
    momentum_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1)
    momentum_score = (momentum_5d * 0.7 + momentum_20d * 0.3)  # Weight recent momentum more
    
    # Enhanced trend strength calculation
    trend_strength = (adx * 0.6 + abs(momentum_score) * 0.4)
    
    # Calculate market conditions score
    market_conditions = {
        'volatility': volatility,
        'trend_strength': trend_strength,
        'momentum': momentum_score
    }
    
    # Adjust confidence based on market conditions
    adjusted_confidence = calculate_prediction_confidence(
        [model_confidence],
        0.7,  # Historical accuracy baseline
        market_conditions
    )
    
    timeframes = [5, 10, 20]
    targets = {}
    
    for days in timeframes:
        # Calculate more precise price ranges
        daily_vol = volatility / np.sqrt(252)
        base_range = daily_vol * np.sqrt(days) * current_price
        
        # Enhanced trend adjustment
        trend_adjustment = (
            trend_strength * daily_vol * current_price * np.sqrt(days) * 
            (1 + abs(momentum_score))
        )
        
        if final_prediction == "Up":
            upper_target = current_price + base_range + trend_adjustment
            lower_target = current_price + (base_range * 0.4)  # Increased support level
        else:
            upper_target = current_price - (base_range * 0.4)  # Increased resistance level
            lower_target = current_price - base_range - trend_adjustment
        
        targets[days] = {
            'upper': upper_target,
            'lower': lower_target,
            'confidence': adjusted_confidence,
            'trend_strength': trend_strength,
            'volatility': volatility
        }
    
    return targets

def evaluate_prediction_quality(predictions, actual_movements, timeframe):
    """Evaluate the quality of predictions for confidence calibration"""
    correct_predictions = np.sum(predictions == actual_movements)
    accuracy = correct_predictions / len(predictions)
    
    # Calculate confidence score based on prediction accuracy
    base_confidence = accuracy * 0.7 + 0.3  # Minimum confidence of 0.3
    
    # Adjust confidence based on prediction consistency
    consistency = 1 - np.std(predictions) * 2
    
    return base_confidence * consistency

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
#    models = create_ensemble_model()

    models = create_enhanced_ensemble_model()
    ensemble_pred, ensemble_pred_proba, feature_importances, confidence_threshold = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, models
    )

    market_volatility = data['ATR_Pct'].rolling(window=20).mean().iloc[-1]
    adx = ta.trend.ADXIndicator(
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    ).adx().iloc[-1] / 100
    trend_strength = adx

    confidence_threshold = adaptive_confidence_threshold(
        ensemble_pred_proba,
        market_volatility,
        trend_strength
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
    price_targets = predict_price_targets_enhanced(
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

    # Predict long-term price targets
    # Add long-term features
    data = get_long_term_features(data)
    
    # Generate long-term predictions
    long_term_predictions = predict_long_term_targets(
        data=data,
        current_price=current_price,
        models=models,
        X_scaled=X_scaled[-1:],
        timeframes=[30, 60, 90]
    )
    
    # Analyze prediction quality
    prediction_quality = analyze_prediction_quality(
        long_term_predictions,
        data,
        timeframes=[30, 60, 90]
    )
    
    # Print long-term predictions
    print("\nLong-term Price Predictions:")
    for timeframe, pred in long_term_predictions.items():
        print(f"\n{timeframe}-Day Forecast:")
        print(f"Direction: {pred['direction']}")
        print(f"Target Price: ${pred['target_price']:.2f}")
        print(f"Confidence: {pred['confidence']*100:.1f}%")
        print(f"Support/Resistance: ${pred['support_resistance']:.2f}")
        print(f"Risk Ratio: {pred['risk_ratio']:.2f}")
        print(f"Market Regime: {pred['market_regime']}")
        
        quality = prediction_quality[timeframe]
        print(f"Historical Accuracy: {quality['historical_accuracy']*100:.1f}%")
        print(f"Trend Consistency: {quality['trend_consistency']*100:.1f}%")


except Exception as e:
    print(f"An error occurred: {str(e)}")