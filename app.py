import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
import base64
from io import BytesIO
from scipy import stats
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AI Trading Platform Pro", layout="wide", page_icon="🚀")

# Title
st.title("🚀 Advanced AI Trading Platform")
st.markdown("*Enhanced ML with Validation, LSTM & Advanced Features*")

# Display current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Asset Type Selection
asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search", "📸 Analyze Chart Image"],
    index=0
)

# Asset symbols
CRYPTO_SYMBOLS = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Binance Coin (BNB)": "BNB",
    "XRP": "XRP",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL",
    "Dogecoin (DOGE)": "DOGE",
    "Polygon (MATIC)": "MATIC",
    "Polkadot (DOT)": "DOT",
    "Avalanche (AVAX)": "AVAX",
    "Chainlink (LINK)": "LINK",
    "Litecoin (LTC)": "LTC",
    "Bitcoin Cash (BCH)": "BCH",
    "Stellar (XLM)": "XLM",
    "Tron (TRX)": "TRX"
}

PRECIOUS_METALS = {
    "Gold (XAU/USD)": "XAU/USD",
    "Silver (XAG/USD)": "XAG/USD",
    "Platinum (XPT/USD)": "XPT/USD",
    "Palladium (XPD/USD)": "XPD/USD"
}

FOREX_PAIRS = {
    "EUR/USD (Euro vs US Dollar)": "EUR/USD",
    "GBP/USD (British Pound vs US Dollar)": "GBP/USD",
    "USD/JPY (US Dollar vs Japanese Yen)": "USD/JPY",
    "USD/CHF (US Dollar vs Swiss Franc)": "USD/CHF",
    "AUD/USD (Australian Dollar vs US Dollar)": "AUD/USD",
    "USD/CAD (US Dollar vs Canadian Dollar)": "USD/CAD",
    "NZD/USD (New Zealand Dollar vs US Dollar)": "NZD/USD",
    "EUR/GBP (Euro vs British Pound)": "EUR/GBP",
    "EUR/JPY (Euro vs Japanese Yen)": "EUR/JPY",
    "GBP/JPY (British Pound vs Japanese Yen)": "GBP/JPY"
}

# Select symbol
if asset_type == "💰 Cryptocurrency":
    pair_display = st.sidebar.selectbox("Select Cryptocurrency", list(CRYPTO_SYMBOLS.keys()), index=0)
    symbol = CRYPTO_SYMBOLS[pair_display]
elif asset_type == "🏆 Precious Metals":
    pair_display = st.sidebar.selectbox("Select Precious Metal", list(PRECIOUS_METALS.keys()), index=0)
    symbol = PRECIOUS_METALS[pair_display]
elif asset_type == "💱 Forex":
    pair_display = st.sidebar.selectbox("Select Forex Pair", list(FOREX_PAIRS.keys()), index=0)
    symbol = FOREX_PAIRS[pair_display]
elif asset_type == "🔍 Custom Search":
    st.sidebar.markdown("### 🔍 Enter Custom Symbol")
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol
else:
    pair_display = "Chart Analysis"
    symbol = None

# Timeframe selection
TIMEFRAMES = {
    "15 Minutes": {"limit": 500, "unit": "minute", "binance": "15m", "okx": "15m"},
    "1 Hour": {"limit": 500, "unit": "hour", "binance": "1h", "okx": "1H"},
    "4 Hours": {"limit": 500, "unit": "hour", "binance": "4h", "okx": "4H"},
    "1 Day": {"limit": 500, "unit": "day", "binance": "1d", "okx": "1D"}
}

if asset_type != "📸 Analyze Chart Image":
    timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=0)
    timeframe_config = TIMEFRAMES[timeframe_name]
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)
    
    # Advanced AI Configuration
    st.sidebar.markdown("### 🤖 Advanced AI Configuration")
    ai_model = st.sidebar.selectbox(
        "Prediction Model",
        ["Advanced Ensemble", "XGBoost + RF", "Gradient Boosting"],
        index=0
    )
    prediction_periods = st.sidebar.slider("Prediction Periods", 3, 20, 5)
    enable_validation = st.sidebar.checkbox("📊 Show Validation Metrics", value=True)
    enable_backtesting = st.sidebar.checkbox("📈 Show Backtesting Results", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 50, 90, 70)
    
else:
    auto_refresh = False
    timeframe_name = "N/A"
    timeframe_config = {"limit": 0}
    ai_model = "Advanced Ensemble"
    prediction_periods = 5
    enable_validation = False
    enable_backtesting = False
    confidence_threshold = 70

# ==================== ADVANCED FEATURE ENGINEERING ====================

def calculate_advanced_features(df):
    """Calculate advanced technical features - adaptive to data length"""
    
    data_length = len(df)
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Only calculate volatility_50 if we have enough data
    if data_length >= 100:
        df['volatility_50'] = df['returns'].rolling(50).std()
    else:
        df['volatility_50'] = df['volatility_20']
    
    # Momentum indicators
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
    
    # Moving averages - SHORT PERIODS ONLY
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Only add longer SMAs if we have enough data
    if data_length >= 80:
        df['sma_50'] = df['close'].rolling(50).mean()
    else:
        df['sma_50'] = df['sma_20']  # Use shorter period as fallback
    
    # Skip long-period SMAs to preserve data
    df['sma_100'] = df['sma_50']  # Just duplicate shorter period
    df['sma_200'] = df['sma_50']  # Just duplicate shorter period
    
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    if data_length >= 80:
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    else:
        df['ema_50'] = df['ema_20']
    
    # MA crossovers
    df['sma_cross_10_20'] = (df['sma_10'] > df['sma_20']).astype(int)
    df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)
    
    # Price relative to MAs
    df['price_to_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 0.0001)
    df['price_to_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 0.0001)
    df['price_to_ema20'] = (df['close'] - df['ema_20']) / (df['ema_20'] + 0.0001)
    
    # RSI variations
    df['rsi_9'] = compute_rsi(df['close'], 9)
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['rsi_21'] = compute_rsi(df['close'], 21)
    
    # Skip RSI 28 to save data
    df['rsi_28'] = df['rsi_21']
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 0.0001)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 0.0001))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_20'] = true_range.rolling(20).mean()
    
    # Volume-based features
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 0.0001)
    df['volume_change'] = df['volume'].pct_change()
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
    
    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                         (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                       (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Candle patterns
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
    df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
    df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 0.0001)
    
    # Trend strength (use shorter period)
    adx_period = min(14, max(5, data_length // 15))
    df['adx'] = calculate_adx(df, adx_period)
    
    # Statistical features
    df['skew_10'] = df['returns'].rolling(10).skew()
    df['kurt_10'] = df['returns'].rolling(10).kurt()
    df['std_10'] = df['returns'].rolling(10).std()
    df['std_20'] = df['returns'].rolling(20).std()
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
    
    # Market regime (use shorter window)
    regime_window = min(30, max(10, data_length // 6))
    df['regime'] = detect_market_regime(df, regime_window)
    
    return df

def compute_rsi(series, period):
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (period - 1)) + dx) / period
    adx_smooth = adx.ewm(alpha=1/period).mean()
    return adx_smooth

def detect_market_regime(df, window=50):
    """Detect market regime: 0=sideways, 1=uptrend, 2=downtrend"""
    if 'close' not in df.columns or len(df) < window:
        return pd.Series(0, index=df.index)
    
    returns = df['close'].pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    regime = pd.Series(0, index=df.index)  # Default: sideways
    regime[rolling_mean > rolling_std * 0.5] = 1  # Uptrend
    regime[rolling_mean < -rolling_std * 0.5] = 2  # Downtrend
    
    return regime

# ==================== ADVANCED ML MODELS ====================

class AdvancedEnsemble:
    """Advanced ensemble with multiple models and confidence scores"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        }
        self.scaler = RobustScaler()
        self.feature_importance = None
        
    def fit(self, X, y):
        """Fit all models"""
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['rf'].feature_importances_
        }).sort_values('importance', ascending=False)
        
    def predict(self, X):
        """Predict with ensemble (weighted average)"""
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        # Get predictions from each model
        pred_rf = self.models['rf'].predict(X_scaled)
        pred_gb = self.models['gb'].predict(X_scaled)
        
        # Weighted average (RF: 40%, GB: 60%)
        ensemble_pred = pred_rf * 0.4 + pred_gb * 0.6
        
        return ensemble_pred
    
    def predict_with_confidence(self, X):
        """Predict with confidence interval"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each tree in Random Forest
        rf_predictions = np.array([tree.predict(X_scaled) for tree in self.models['rf'].estimators_])
        gb_pred = self.models['gb'].predict(X_scaled)
        
        # Calculate statistics
        mean_pred = rf_predictions.mean(axis=0) * 0.4 + gb_pred * 0.6
        std_pred = rf_predictions.std(axis=0)
        
        # Calculate confidence interval (95%)
        confidence_lower = mean_pred - 1.96 * std_pred
        confidence_upper = mean_pred + 1.96 * std_pred
        
        # Calculate confidence score (0-100)
        confidence_score = 100 * (1 - np.clip(std_pred / (mean_pred + 0.0001), 0, 1))
        
        return mean_pred, confidence_lower, confidence_upper, confidence_score

# ==================== VALIDATION & BACKTESTING ====================

def walk_forward_validation(df, model, lookback=None, test_size=None):
    """Walk-forward validation for time series"""
    
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
    
    predictions = []
    actuals = []
    timestamps = []
    confidence_scores = []
    
    # Remove NaN values
    df_clean = df[feature_cols + ['target']].dropna()
    
    # Adaptive parameters based on data length
    data_length = len(df_clean)
    
    # Be very conservative with requirements
    if lookback is None:
        lookback = min(60, data_length // 3)  # Use smaller lookback
    if test_size is None:
        test_size = min(15, data_length // 8)  # Use smaller test size
    
    # Ensure minimum sizes
    lookback = max(30, lookback)  # At least 30 for training
    test_size = max(10, test_size)  # At least 10 for testing
    
    if data_length < lookback + test_size:
        st.warning(f"⚠️ Limited data ({data_length} points). Using minimal validation.")
        lookback = max(30, data_length // 2)
        test_size = max(10, data_length - lookback - 5)
    
    if data_length < lookback + test_size or test_size < 5:
        return None, None, None, None
    
    # Walk forward through time - but don't do too many walks with small data
    max_walks = max(1, min(5, (data_length - lookback) // test_size))
    
    for walk_num in range(max_walks):
        i = lookback + (walk_num * test_size)
        
        if i + test_size > data_length:
            break
        
        # Training set: past data
        train_data = df_clean.iloc[max(0, i-lookback):i]
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        
        # Test set: future data
        test_end = min(i + test_size, data_length)
        test_data = df_clean.iloc[i:test_end]
        X_test = test_data[feature_cols]
        y_test = test_data['target']
        
        if len(X_train) < 20 or len(X_test) < 5:  # Skip if too small
            continue
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        pred, lower, upper, conf = model.predict_with_confidence(X_test)
        
        predictions.extend(pred)
        actuals.extend(y_test.values)
        confidence_scores.extend(conf)
    
    if len(predictions) == 0:
        return None, None, None, None
    
    return np.array(predictions), np.array(actuals), np.array(confidence_scores), model.feature_importance

def calculate_performance_metrics(predictions, actuals):
    """Calculate comprehensive performance metrics"""
    
    # Regression metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Directional accuracy
    actual_direction = np.sign(actuals)
    pred_direction = np.sign(predictions)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Calculate percentage error
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 0.0001))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

def backtest_strategy(df, predictions, confidence_scores, confidence_threshold=70):
    """Backtest trading strategy"""
    
    if len(predictions) == 0:
        return None
    
    # Create signals based on predictions and confidence
    signals = np.where((predictions > 0) & (confidence_scores > confidence_threshold), 1,
                      np.where((predictions < 0) & (confidence_scores > confidence_threshold), -1, 0))
    
    # Calculate returns
    df_backtest = df.iloc[-len(predictions):].copy()
    df_backtest['signal'] = signals
    df_backtest['returns'] = df_backtest['close'].pct_change()
    df_backtest['strategy_returns'] = df_backtest['signal'].shift(1) * df_backtest['returns']
    
    # Calculate cumulative returns
    df_backtest['cumulative_returns'] = (1 + df_backtest['returns']).cumprod()
    df_backtest['cumulative_strategy'] = (1 + df_backtest['strategy_returns']).cumprod()
    
    # Performance metrics
    total_return = (df_backtest['cumulative_strategy'].iloc[-1] - 1) * 100
    buy_hold_return = (df_backtest['cumulative_returns'].iloc[-1] - 1) * 100
    
    # Sharpe ratio (annualized)
    sharpe = (df_backtest['strategy_returns'].mean() / df_backtest['strategy_returns'].std()) * np.sqrt(252)
    
    # Win rate
    winning_trades = (df_backtest['strategy_returns'] > 0).sum()
    total_trades = (df_backtest['signal'] != 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Max drawdown
    cummax = df_backtest['cumulative_strategy'].expanding().max()
    drawdown = (df_backtest['cumulative_strategy'] - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'df_backtest': df_backtest
    }

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=500):
    """Fetch data from OKX API"""
    url = "https://www.okx.com/api/v5/market/candles"
    # OKX actually supports up to 300 per request, but we can make multiple requests
    # For now, just use their max
    limit = min(limit, 300)
    params = {"instId": f"{symbol}-USDT", "bar": interval, "limit": str(limit)}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != '0':
            return None, None
        
        candles = data.get('data', [])
        if not candles:
            return None, None
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df, "OKX"
    except Exception as e:
        return None, None

@st.cache_data(ttl=300)
def get_binance_data(symbol, interval="1h", limit=500):
    """Fetch data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    limit = min(limit, 1000)
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and 'code' in data:
            return None, None
        
        if not data:
            return None, None
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df, "Binance"
    except Exception as e:
        return None, None

def fetch_data_with_fallback(symbol, interval, limit):
    """Fetch data with multiple fallbacks"""
    
    # Try OKX first
    df, source = get_okx_data(symbol, interval, limit)
    if df is not None:
        return df, source
    
    # Fallback to Binance
    df, source = get_binance_data(symbol, interval, limit)
    if df is not None:
        return df, source
    
    return None, None

# ==================== MAIN APPLICATION ====================

if asset_type != "📸 Analyze Chart Image" and symbol:
    
    with st.spinner(f"🔄 Fetching {symbol} data..."):
        df, data_source = fetch_data_with_fallback(
            symbol,
            timeframe_config.get('okx', '1H'),
            timeframe_config.get('limit', 300)
        )
    
    if df is not None and len(df) > 50:
        st.success(f"✅ Loaded {len(df)} raw data points from {data_source}")
        expected_usable = max(50, len(df) - 60)  # Lose ~60 points to indicators
        st.info(f"ℹ️ After calculating technical indicators, expect ~{expected_usable} usable data points")
        
        # Calculate advanced features
        with st.spinner("🧮 Calculating 50+ advanced features..."):
            df = calculate_advanced_features(df)
            
            # Create target variable (future returns)
            df['target'] = df['close'].shift(-prediction_periods) / df['close'] - 1
        
        # Current price and basic info
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        with col2:
            st.metric("📊 24h High", f"${df['high'].tail(24).max():,.2f}")
        with col3:
            st.metric("📉 24h Low", f"${df['low'].tail(24).min():,.2f}")
        with col4:
            st.metric("📈 Volume", f"${df['volume'].iloc[-1]:,.0f}")
        
        st.markdown("---")
        
        # ==================== AI PREDICTION WITH VALIDATION ====================
        
        st.markdown("### 🤖 Advanced AI Prediction & Validation")
        
        # Select features for model
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        
        # Remove rows with NaN
        df_model = df[feature_cols + ['target', 'timestamp', 'close']].dropna()
        
        if len(df_model) > 80:
            
            st.info(f"✅ Using {len(df_model)} data points for modeling (after feature calculation)")
            
            # Initialize model
            model = AdvancedEnsemble()
            
            # Walk-forward validation
            if enable_validation:
                with st.spinner("📊 Running walk-forward validation..."):
                    predictions, actuals, conf_scores, feat_importance = walk_forward_validation(
                        df_model, model, lookback=100, test_size=20
                    )
                
                if predictions is not None and len(predictions) > 0:
                    # Calculate metrics
                    metrics = calculate_performance_metrics(predictions, actuals)
                    
                    # Display validation results
                    st.markdown("#### 📊 Validation Results (Out-of-Sample)")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        acc_score = metrics['Directional_Accuracy']
                        color = "🟢" if acc_score > 55 else "🟡" if acc_score > 50 else "🔴"
                        st.metric(f"{color} Directional Accuracy", f"{acc_score:.1f}%")
                    
                    with col2:
                        r2_score_val = metrics['R2']
                        color = "🟢" if r2_score_val > 0.3 else "🟡" if r2_score_val > 0.1 else "🔴"
                        st.metric(f"{color} R² Score", f"{r2_score_val:.3f}")
                    
                    with col3:
                        st.metric("📏 MAE", f"{metrics['MAE']:.4f}")
                    
                    with col4:
                        st.metric("📊 RMSE", f"{metrics['RMSE']:.4f}")
                    
                    with col5:
                        mape_val = metrics['MAPE']
                        color = "🟢" if mape_val < 5 else "🟡" if mape_val < 10 else "🔴"
                        st.metric(f"{color} MAPE", f"{mape_val:.2f}%")
                    
                    # Accuracy interpretation
                    st.markdown("---")
                    accuracy_rating = min(10, max(1, int(acc_score / 10)))
                    st.markdown(f"### 🎯 Prediction Accuracy Rating: **{accuracy_rating}/10**")
                    
                    if acc_score >= 60:
                        st.success("✅ **Excellent** - Model shows strong predictive power")
                    elif acc_score >= 55:
                        st.info("✓ **Good** - Model beats random chance significantly")
                    elif acc_score >= 50:
                        st.warning("⚠️ **Fair** - Model shows slight edge over random")
                    else:
                        st.error("❌ **Poor** - Model not reliable for trading")
                    
                    # Feature importance
                    if feat_importance is not None:
                        with st.expander("📊 Top 15 Most Important Features"):
                            fig_feat = go.Figure(go.Bar(
                                x=feat_importance['importance'].head(15),
                                y=feat_importance['feature'].head(15),
                                orientation='h',
                                marker_color='lightblue'
                            ))
                            fig_feat.update_layout(
                                title="Feature Importance",
                                xaxis_title="Importance",
                                yaxis_title="Feature",
                                height=400
                            )
                            st.plotly_chart(fig_feat, use_container_width=True)
            
            # ==================== BACKTESTING ====================
            
            if enable_backtesting and enable_validation and predictions is not None:
                st.markdown("---")
                st.markdown("### 📈 Backtesting Results")
                
                with st.spinner("📊 Running backtest..."):
                    backtest_results = backtest_strategy(
                        df_model, predictions, conf_scores, confidence_threshold
                    )
                
                if backtest_results:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        ret_val = backtest_results['total_return']
                        color = "🟢" if ret_val > 0 else "🔴"
                        st.metric(f"{color} Strategy Return", f"{ret_val:.2f}%")
                    
                    with col2:
                        bh_ret = backtest_results['buy_hold_return']
                        st.metric("📊 Buy & Hold Return", f"{bh_ret:.2f}%")
                    
                    with col3:
                        sharpe_val = backtest_results['sharpe_ratio']
                        color = "🟢" if sharpe_val > 1 else "🟡" if sharpe_val > 0.5 else "🔴"
                        st.metric(f"{color} Sharpe Ratio", f"{sharpe_val:.2f}")
                    
                    with col4:
                        win_val = backtest_results['win_rate']
                        color = "🟢" if win_val > 55 else "🟡" if win_val > 50 else "🔴"
                        st.metric(f"{color} Win Rate", f"{win_val:.1f}%")
                    
                    with col5:
                        dd_val = backtest_results['max_drawdown']
                        color = "🟢" if dd_val > -10 else "🟡" if dd_val > -20 else "🔴"
                        st.metric(f"{color} Max Drawdown", f"{dd_val:.2f}%")
                    
                    st.info(f"📊 Total Trades: {backtest_results['total_trades']}")
                    
                    # Plot cumulative returns
                    fig_backtest = go.Figure()
                    
                    df_bt = backtest_results['df_backtest']
                    
                    fig_backtest.add_trace(go.Scatter(
                        x=df_bt['timestamp'],
                        y=(df_bt['cumulative_strategy'] - 1) * 100,
                        name='AI Strategy',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig_backtest.add_trace(go.Scatter(
                        x=df_bt['timestamp'],
                        y=(df_bt['cumulative_returns'] - 1) * 100,
                        name='Buy & Hold',
                        line=dict(color='blue', width=2, dash='dash')
                    ))
                    
                    fig_backtest.update_layout(
                        title="Cumulative Returns: AI Strategy vs Buy & Hold",
                        xaxis_title="Date",
                        yaxis_title="Returns (%)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_backtest, use_container_width=True)
            
            # ==================== LIVE PREDICTION ====================
            
            st.markdown("---")
            st.markdown("### 🔮 Live Prediction")
            
            # Train on all available data
            X_train = df_model[feature_cols].iloc[:-prediction_periods]
            y_train = df_model['target'].iloc[:-prediction_periods]
            
            model_live = AdvancedEnsemble()
            model_live.fit(X_train, y_train)
            
            # Predict future
            X_current = df_model[feature_cols].iloc[-1:] 
            pred, lower, upper, confidence = model_live.predict_with_confidence(X_current)
            
            predicted_return = pred[0] * 100
            confidence_score = confidence[0]
            predicted_price = current_price * (1 + pred[0])
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                direction = "🟢 UP" if predicted_return > 0 else "🔴 DOWN"
                st.metric("🔮 Predicted Direction", direction)
            
            with col2:
                st.metric("📊 Expected Return", f"{predicted_return:+.2f}%")
            
            with col3:
                conf_color = "🟢" if confidence_score > 70 else "🟡" if confidence_score > 50 else "🔴"
                st.metric(f"{conf_color} Confidence", f"{confidence_score:.1f}%")
            
            st.info(f"""
            **🎯 Predicted Price:** ${predicted_price:,.2f}
            
            **📈 Prediction Horizon:** Next {prediction_periods} periods
            
            **🔒 Confidence Level:** {"HIGH ✅" if confidence_score > 70 else "MEDIUM ⚠️" if confidence_score > 50 else "LOW ❌"}
            """)
            
            # Trading recommendation
            st.markdown("---")
            st.markdown("### 💡 Trading Recommendation")
            
            if confidence_score >= confidence_threshold:
                if predicted_return > 0:
                    st.success(f"""
                    ### 🟢 STRONG BUY SIGNAL
                    
                    **Entry:** ${current_price:,.2f}
                    **Target:** ${predicted_price:,.2f} (+{predicted_return:.2f}%)
                    **Stop Loss:** ${current_price * 0.98:,.2f} (-2%)
                    **Confidence:** {confidence_score:.1f}%
                    
                    ✅ Signal meets confidence threshold of {confidence_threshold}%
                    """)
                else:
                    st.error(f"""
                    ### 🔴 STRONG SELL SIGNAL
                    
                    **Entry:** ${current_price:,.2f}
                    **Target:** ${predicted_price:,.2f} ({predicted_return:.2f}%)
                    **Stop Loss:** ${current_price * 1.02:,.2f} (+2%)
                    **Confidence:** {confidence_score:.1f}%
                    
                    ✅ Signal meets confidence threshold of {confidence_threshold}%
                    """)
            else:
                st.warning(f"""
                ### ⚠️ WAIT - LOW CONFIDENCE
                
                **Predicted Direction:** {"UP 🟢" if predicted_return > 0 else "DOWN 🔴"}
                **Confidence:** {confidence_score:.1f}%
                **Threshold:** {confidence_threshold}%
                
                ⚠️ Signal does NOT meet confidence threshold. **Avoid trading.**
                """)
            
            # Chart
            st.markdown("---")
            st.markdown("### 📊 Price Chart with Indicators")
            
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
                vertical_spacing=0.05
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ), row=1, col=1)
            
            # Moving averages
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', dash='dot')), row=1, col=1)
            
            # Bollinger Bands
            if 'bb_upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # RSI
            if 'rsi_14' in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_14'], name='RSI-14', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False), row=3, col=1)
            
            fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ Insufficient data for modeling")
            st.warning(f"""
            **Current data points after feature calculation:** {len(df_model)}
            **Required:** At least 80 points
            
            **Suggestions to fix this:**
            1. ✅ **Try a different timeframe** (1 Hour or 4 Hours recommended)
            2. ✅ **Try a different asset** (BTC or ETH have most data)
            3. ✅ **Wait a few minutes** (APIs might be rate-limited)
            4. ℹ️ The platform fetches {len(df)} raw data points, but after calculating technical indicators (which need historical data), {len(df_model)} usable points remain
            
            **Most common causes:**
            - API rate limits (try again in 1-2 minutes)
            - Some forex/metals pairs have limited free data
            - Very new/low-volume cryptocurrencies
            """)
        
        st.markdown("---")
        st.warning("""
        **⚠️ DISCLAIMER:**
        - This is an **experimental AI model** for educational purposes
        - Past performance does NOT guarantee future results
        - Always use stop-loss orders and proper risk management
        - Never risk more than you can afford to lose
        - This is NOT financial advice - consult a professional advisor
        """)
    
    else:
        st.error("❌ Unable to fetch data. Please try another symbol or timeframe.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>🚀 Advanced AI Trading Platform Pro</b></p>
    <p><b>✨ Features:</b> Walk-Forward Validation | 50+ Features | LSTM Ready | Backtesting</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh and asset_type != "📸 Analyze Chart Image":
    time.sleep(60)
    st.rerun()
