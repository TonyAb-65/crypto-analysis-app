import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
from scipy import stats
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

# Title
st.title("🤖 AI Trading Analysis Platform")
st.markdown("*Crypto, Forex, Metals + Multi-Timeframe Analysis + AI Predictions*")

# Display current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Asset Type Selection
asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search"],
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

# Select symbol based on asset type
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
    st.sidebar.info("""
    **Examples:**
    - Crypto: BTC, ETH, DOGE
    - Forex: EUR/USD, GBP/JPY, AUD/USD
    - Metals: XAU/USD, XAG/USD
    - Stocks: AAPL, TSLA, GOOGL
    """)
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol

# ORIGINAL TIMEFRAMES - ALL RESTORED
TIMEFRAMES = {
    "1 Minute": {"limit": 60, "unit": "minute", "binance": "1m", "okx": "1m", "hold_time": "2-5 minutes"},
    "5 Minutes": {"limit": 100, "unit": "minute", "binance": "5m", "okx": "5m", "hold_time": "10-30 minutes"},
    "10 Minutes": {"limit": 100, "unit": "minute", "binance": "10m", "okx": "10m", "hold_time": "20-60 minutes"},
    "15 Minutes": {"limit": 150, "unit": "minute", "binance": "15m", "okx": "15m", "hold_time": "30 minutes to 2 hours"},
    "30 Minutes": {"limit": 150, "unit": "minute", "binance": "30m", "okx": "30m", "hold_time": "1-3 hours"},
    "1 Hour": {"limit": 200, "unit": "hour", "binance": "1h", "okx": "1H", "hold_time": "2-8 hours"},
    "4 Hours": {"limit": 200, "unit": "hour", "binance": "4h", "okx": "4H", "hold_time": "8-24 hours"},
    "1 Day": {"limit": 200, "unit": "day", "binance": "1d", "okx": "1D", "hold_time": "2-7 days"},
    "1 Week": {"limit": 100, "unit": "week", "binance": "1w", "okx": "1W", "hold_time": "1-4 weeks"}
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=5)
timeframe_config = TIMEFRAMES[timeframe_name]
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)

# AI Model Selection
st.sidebar.markdown("### 🤖 AI Configuration")
ai_model = st.sidebar.selectbox(
    "Prediction Model",
    ["Advanced Ensemble (Recommended)", "Random Forest", "Gradient Boosting"],
    index=0
)
prediction_periods = st.sidebar.slider("Prediction Periods", 1, 20, 5)

# Advanced Options
st.sidebar.markdown("### 📊 Advanced Options")
enable_multi_timeframe = st.sidebar.checkbox("🔍 Multi-Timeframe Analysis", value=True, help="Checks lower timeframes to confirm trend")
enable_validation = st.sidebar.checkbox("📊 Show Validation Metrics", value=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 50, 90, 70)

# Technical Indicators
st.sidebar.markdown("### 📊 Technical Indicators")
use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
use_ema = st.sidebar.checkbox("EMA (20, 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (12, 16, 24)", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# ==================== API FUNCTIONS ====================

@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=200):
    """Fetch data from OKX API"""
    url = "https://www.okx.com/api/v5/market/candles"
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
def get_binance_data(symbol, interval="1h", limit=200):
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

# ==================== MULTI-TIMEFRAME ANALYSIS ====================

def get_lower_timeframes_trend(symbol, main_timeframe):
    """
    Check lower timeframes to confirm trend
    For 1h: Check 5m, 10m, 15m, 30m
    For 15m: Check 1m, 5m, 10m
    """
    
    lower_tf_map = {
        "1 Hour": ["5m", "10m", "15m", "30m"],
        "15 Minutes": ["1m", "5m", "10m"],
        "30 Minutes": ["5m", "10m", "15m"],
        "4 Hours": ["15m", "30m", "1H"],
        "1 Day": ["1H", "4H"]
    }
    
    if main_timeframe not in lower_tf_map:
        return None
    
    lower_timeframes = lower_tf_map[main_timeframe]
    trend_signals = []
    
    for tf in lower_timeframes:
        df_lower, _ = get_okx_data(symbol, tf, 50)
        if df_lower is None:
            df_lower, _ = get_binance_data(symbol, tf, 50)
        
        if df_lower is not None and len(df_lower) > 20:
            # Calculate simple trend
            df_lower['sma_10'] = df_lower['close'].rolling(10).mean()
            df_lower['sma_20'] = df_lower['close'].rolling(20).mean()
            
            # Current trend
            last_close = df_lower['close'].iloc[-1]
            last_sma10 = df_lower['sma_10'].iloc[-1]
            last_sma20 = df_lower['sma_20'].iloc[-1]
            
            if last_close > last_sma10 > last_sma20:
                trend_signals.append(1)  # Bullish
            elif last_close < last_sma10 < last_sma20:
                trend_signals.append(-1)  # Bearish
            else:
                trend_signals.append(0)  # Neutral
    
    if not trend_signals:
        return None
    
    # Calculate consensus
    avg_signal = np.mean(trend_signals)
    
    return {
        'signal': avg_signal,
        'bullish_count': sum(1 for s in trend_signals if s > 0),
        'bearish_count': sum(1 for s in trend_signals if s < 0),
        'neutral_count': sum(1 for s in trend_signals if s == 0),
        'total_timeframes': len(trend_signals),
        'consensus': 'BULLISH' if avg_signal > 0.3 else 'BEARISH' if avg_signal < -0.3 else 'MIXED'
    }

# ==================== TECHNICAL INDICATORS ====================

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    df['rsi_12'] = compute_rsi(df['close'], 12)
    df['rsi_16'] = compute_rsi(df['close'], 16)
    df['rsi_24'] = compute_rsi(df['close'], 24)
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Additional features for ML
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 0.0001)
    
    # Price position
    df['price_to_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 0.0001)
    df['price_to_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 0.0001)
    
    return df

def compute_rsi(series, period):
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==================== ML MODEL ====================

class ImprovedEnsemble:
    """Ensemble model optimized for trading"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                min_samples_split=15,
                subsample=0.8,
                random_state=42
            )
        }
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        """Fit all models"""
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
    def predict(self, X):
        """Predict with ensemble"""
        X_scaled = self.scaler.transform(X)
        
        pred_rf = self.models['rf'].predict(X_scaled)
        pred_gb = self.models['gb'].predict(X_scaled)
        
        # Weighted average
        ensemble_pred = pred_rf * 0.45 + pred_gb * 0.55
        
        return ensemble_pred
    
    def predict_with_confidence(self, X):
        """Predict with confidence interval"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each tree in Random Forest
        rf_predictions = np.array([tree.predict(X_scaled) for tree in self.models['rf'].estimators_])
        gb_pred = self.models['gb'].predict(X_scaled)
        
        # Calculate statistics
        mean_pred = rf_predictions.mean(axis=0) * 0.45 + gb_pred * 0.55
        std_pred = rf_predictions.std(axis=0)
        
        # Confidence score (0-100)
        confidence_score = 100 * (1 - np.clip(std_pred / (np.abs(mean_pred) + 0.0001), 0, 1))
        
        return mean_pred, confidence_score

# ==================== VALIDATION ====================

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
    
    # MAPE
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 0.0001))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

# ==================== MAIN APPLICATION ====================

if symbol:
    
    with st.spinner(f"🔄 Fetching {symbol} data for {timeframe_name}..."):
        df, data_source = fetch_data_with_fallback(
            symbol,
            timeframe_config.get('okx', '1H'),
            timeframe_config.get('limit', 200)
        )
    
    if df is not None and len(df) > 50:
        st.success(f"✅ Loaded {len(df)} data points from {data_source}")
        
        # Multi-Timeframe Analysis
        mtf_analysis = None
        if enable_multi_timeframe and timeframe_name in ["1 Hour", "15 Minutes", "30 Minutes", "4 Hours", "1 Day"]:
            with st.spinner("🔍 Analyzing lower timeframes..."):
                mtf_analysis = get_lower_timeframes_trend(symbol, timeframe_name)
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Current metrics
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
        
        # Multi-Timeframe Analysis Display
        if mtf_analysis:
            st.markdown("### 🔍 Multi-Timeframe Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                consensus_color = "🟢" if mtf_analysis['consensus'] == 'BULLISH' else "🔴" if mtf_analysis['consensus'] == 'BEARISH' else "🟡"
                st.metric(f"{consensus_color} Consensus", mtf_analysis['consensus'])
            
            with col2:
                st.metric("📊 Bullish Timeframes", f"{mtf_analysis['bullish_count']}/{mtf_analysis['total_timeframes']}")
            
            with col3:
                st.metric("📉 Bearish Timeframes", f"{mtf_analysis['bearish_count']}/{mtf_analysis['total_timeframes']}")
            
            st.info(f"✅ Multi-timeframe confirmation: {mtf_analysis['consensus']} across {mtf_analysis['total_timeframes']} lower timeframes")
            st.markdown("---")
        
        # Prepare features for ML
        feature_cols = ['returns', 'volatility', 'momentum', 'volume_ratio', 
                       'price_to_sma20', 'price_to_sma50']
        
        # Add indicator features if they exist
        for col in ['rsi_12', 'rsi_16', 'macd_hist']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Create target (future returns)
        df['target'] = df['close'].shift(-prediction_periods) / df['close'] - 1
        
        # Remove NaN
        df_model = df[feature_cols + ['target', 'timestamp', 'close']].dropna()
        
        if len(df_model) > 80:
            st.info(f"✅ Training AI model with {len(df_model)} data points")
            
            # Train model
            X_train = df_model[feature_cols].iloc[:-prediction_periods]
            y_train = df_model['target'].iloc[:-prediction_periods]
            
            model = ImprovedEnsemble()
            model.fit(X_train, y_train)
            
            # Validation
            if enable_validation and len(X_train) > 100:
                split_point = int(len(X_train) * 0.7)
                X_val = X_train.iloc[split_point:]
                y_val = y_train.iloc[split_point:]
                
                val_pred = model.predict(X_val)
                metrics = calculate_performance_metrics(val_pred, y_val.values)
                
                st.markdown("### 📊 AI Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    acc_score = metrics['Directional_Accuracy']
                    color = "🟢" if acc_score > 55 else "🟡" if acc_score > 50 else "🔴"
                    st.metric(f"{color} Directional Accuracy", f"{acc_score:.1f}%")
                
                with col2:
                    r2_val = metrics['R2']
                    color = "🟢" if r2_val > 0.2 else "🟡" if r2_val > 0.1 else "🔴"
                    st.metric(f"{color} R² Score", f"{r2_val:.3f}")
                
                with col3:
                    st.metric("📏 MAE", f"{metrics['MAE']:.4f}")
                
                with col4:
                    mape_val = metrics['MAPE']
                    color = "🟢" if mape_val < 5 else "🟡" if mape_val < 10 else "🔴"
                    st.metric(f"{color} MAPE", f"{mape_val:.2f}%")
                
                # Accuracy rating
                accuracy_rating = min(10, max(1, int(acc_score / 10)))
                st.markdown(f"### 🎯 Prediction Accuracy Rating: **{accuracy_rating}/10**")
                
                if acc_score >= 58:
                    st.success("✅ **Excellent** - Model shows strong predictive power for trading")
                elif acc_score >= 54:
                    st.info("✓ **Good** - Model shows good edge over random")
                elif acc_score >= 50:
                    st.warning("⚠️ **Fair** - Model shows slight edge")
                else:
                    st.error("❌ **Poor** - Model not reliable for trading")
                
                st.markdown("---")
            
            # Live Prediction
            st.markdown("### 🔮 AI Prediction & Trading Setup")
            
            X_current = df_model[feature_cols].iloc[-1:]
            pred, confidence = model.predict_with_confidence(X_current)
            
            predicted_return = pred[0] * 100
            confidence_score = confidence[0]
            predicted_price = current_price * (1 + pred[0])
            
            # Apply multi-timeframe filter
            mtf_filter_passed = True
            if mtf_analysis:
                if predicted_return > 0 and mtf_analysis['consensus'] == 'BEARISH':
                    mtf_filter_passed = False
                    st.warning("⚠️ **Multi-Timeframe Conflict:** AI predicts UP but lower timeframes show BEARISH trend")
                elif predicted_return < 0 and mtf_analysis['consensus'] == 'BULLISH':
                    mtf_filter_passed = False
                    st.warning("⚠️ **Multi-Timeframe Conflict:** AI predicts DOWN but lower timeframes show BULLISH trend")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                direction = "🟢 UP" if predicted_return > 0 else "🔴 DOWN"
                st.metric("🔮 Predicted Direction", direction)
            
            with col2:
                st.metric("📊 Expected Return", f"{predicted_return:+.2f}%")
            
            with col3:
                conf_color = "🟢" if confidence_score > 70 else "🟡" if confidence_score > 50 else "🔴"
                st.metric(f"{conf_color} Confidence", f"{confidence_score:.1f}%")
            
            # Trading Recommendation
            st.markdown("### 💡 Trading Recommendation")
            
            # Check all conditions
            signal_strength = confidence_score >= confidence_threshold and mtf_filter_passed
            
            if signal_strength:
                if predicted_return > 0:
                    st.success(f"""
                    ### 🟢 STRONG BUY SIGNAL
                    
                    **📍 Entry:** ${current_price:,.2f}
                    **🎯 Target:** ${predicted_price:,.2f} (+{predicted_return:.2f}%)
                    **🛡️ Stop Loss:** ${current_price * 0.98:,.2f} (-2%)
                    **🔒 Confidence:** {confidence_score:.1f}%
                    **⏰ Recommended Hold Time:** {timeframe_config['hold_time']}
                    
                    ✅ Signal meets confidence threshold ({confidence_threshold}%)
                    {"✅ Multi-timeframe confirmation: " + mtf_analysis['consensus'] if mtf_analysis else ""}
                    """)
                else:
                    st.error(f"""
                    ### 🔴 STRONG SELL SIGNAL
                    
                    **📍 Entry:** ${current_price:,.2f}
                    **🎯 Target:** ${predicted_price:,.2f} ({predicted_return:.2f}%)
                    **🛡️ Stop Loss:** ${current_price * 1.02:,.2f} (+2%)
                    **🔒 Confidence:** {confidence_score:.1f}%
                    **⏰ Recommended Hold Time:** {timeframe_config['hold_time']}
                    
                    ✅ Signal meets confidence threshold ({confidence_threshold}%)
                    {"✅ Multi-timeframe confirmation: " + mtf_analysis['consensus'] if mtf_analysis else ""}
                    """)
            else:
                st.warning(f"""
                ### ⚠️ WAIT - CONDITIONS NOT MET
                
                **Predicted Direction:** {"UP 🟢" if predicted_return > 0 else "DOWN 🔴"}
                **Confidence:** {confidence_score:.1f}%
                **Threshold:** {confidence_threshold}%
                {"**Multi-Timeframe:** Conflict detected" if not mtf_filter_passed else ""}
                
                ⚠️ Wait for better setup. Avoid trading when confidence is low or timeframes conflict.
                """)
            
            # Chart
            st.markdown("---")
            st.markdown("### 📊 Price Chart with Technical Analysis")
            
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'),
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
            if use_sma:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
            
            if use_ema:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', dash='dot')), row=1, col=1)
            
            if use_bb:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # Volume
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
            
            # RSI
            if use_rsi:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_12'], name='RSI-12', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_16'], name='RSI-16', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            if use_macd:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=4, col=1)
                colors_macd = ['green' if val > 0 else 'red' for val in df['macd_hist']]
                fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], marker_color=colors_macd, showlegend=False), row=4, col=1)
            
            fig.update_layout(height=1000, showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Insufficient data for modeling (need at least 80 points)")
        
        st.markdown("---")
        st.warning("""
        **⚠️ IMPORTANT NOTES:**
        - **Recommended Hold Time:** {hold_time}
        - Always use stop-loss orders (2% maximum loss)
        - Position sizing: 1-2% of capital per trade
        - Multi-timeframe analysis helps confirm trend direction
        - This is educational - NOT financial advice
        """.format(hold_time=timeframe_config['hold_time']))
    
    else:
        st.error("❌ Unable to fetch data. Try different asset or timeframe.")

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>🚀 AI Trading Platform with Multi-Timeframe Analysis</b></p>
    <p><b>✨ Features:</b> Multi-Timeframe Confirmation | AI Predictions | Position Hold Times | Validation</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
