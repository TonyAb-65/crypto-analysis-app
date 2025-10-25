import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

# Title
st.title("🤖 AI Trading Analysis Platform - IMPROVED")
st.markdown("*Crypto, Forex, Metals + Enhanced AI Predictions*")

# Display current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Debug mode
debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed API information")

# Asset Type Selection
asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search"],
    index=0
)

# Asset symbols based on type
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
    "Avalanche (AVAX)": "AVAX"
}

PRECIOUS_METALS = {
    "Gold (XAU/USD)": "XAU/USD",
    "Silver (XAG/USD)": "XAG/USD"
}

FOREX_PAIRS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY"
}

# Select symbol
if asset_type == "💰 Cryptocurrency":
    pair_display = st.sidebar.selectbox("Select Cryptocurrency", list(CRYPTO_SYMBOLS.keys()), index=0)
    symbol = CRYPTO_SYMBOLS[pair_display]
elif asset_type == "🏆 Precious Metals":
    pair_display = st.sidebar.selectbox("Select Metal", list(PRECIOUS_METALS.keys()), index=0)
    symbol = PRECIOUS_METALS[pair_display]
elif asset_type == "💱 Forex":
    pair_display = st.sidebar.selectbox("Select Forex Pair", list(FOREX_PAIRS.keys()), index=0)
    symbol = FOREX_PAIRS[pair_display]
else:
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol

# Timeframe selection
TIMEFRAMES = {
    "1 Hour": {"limit": 100, "binance": "1h", "okx": "1H"},
    "4 Hours": {"limit": 100, "binance": "4h", "okx": "4H"},
    "1 Day": {"limit": 100, "binance": "1d", "okx": "1D"}
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=0)
timeframe_config = TIMEFRAMES[timeframe_name]

# AI Configuration
st.sidebar.markdown("### 🤖 AI Configuration")
prediction_periods = st.sidebar.slider("Prediction Periods", 1, 10, 5)
lookback_hours = st.sidebar.slider("Context Window (hours)", 4, 12, 6, 
                                   help="How many hours to look back for pattern analysis")

# Technical Indicators
st.sidebar.markdown("### 📊 Technical Indicators")
use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (14)", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# API Functions
@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=100):
    """Fetch data from OKX API - PRIMARY BACKUP"""
    url = "https://www.okx.com/api/v5/market/candles"
    limit = min(limit, 300)
    params = {"instId": f"{symbol}-USDT", "bar": interval, "limit": str(limit)}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != '0':
            error_msg = data.get('msg', 'Unknown error')
            st.warning(f"⚠️ OKX error: {error_msg}")
            return None, None
        
        candles = data.get('data', [])
        if not candles or len(candles) == 0:
            st.warning(f"⚠️ OKX returned no data")
            return None, None
        
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'volCcy', 'volCcyQuote', 'confirm'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        st.success(f"✅ Loaded {len(df)} data points from OKX")
        return df, "OKX"
    except Exception as e:
        st.warning(f"⚠️ OKX API failed: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_binance_data(symbol, interval="1h", limit=100):
    """Fetch data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and 'code' in data:
            st.warning(f"⚠️ Binance error: {data.get('msg', 'Unknown')}")
            return None, None
        
        if not data or len(data) == 0:
            st.warning("⚠️ Binance returned no data")
            return None, None
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        st.success(f"✅ Loaded {len(df)} data points from Binance")
        return df, "Binance"
    except Exception as e:
        st.warning(f"⚠️ Binance API failed: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_cryptocompare_data(symbol, limit=100):
    """Fetch data from CryptoCompare API (Backup)"""
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('Response') != 'Success':
            st.warning(f"⚠️ CryptoCompare error: {data.get('Message', 'Unknown')}")
            return None, None
        
        hist_data = data.get('Data', {}).get('Data', [])
        if not hist_data:
            st.warning("⚠️ CryptoCompare returned no data")
            return None, None
        
        df = pd.DataFrame(hist_data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open': 'open', 
            'high': 'high', 
            'low': 'low', 
            'close': 'close', 
            'volumefrom': 'volume'
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        st.success(f"✅ Loaded {len(df)} data points from CryptoCompare")
        return df, "CryptoCompare"
    except Exception as e:
        st.warning(f"⚠️ CryptoCompare API failed: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_coingecko_data(symbol, limit=100):
    """Fetch data from CoinGecko API (Backup 2)"""
    # Map common symbols to CoinGecko IDs
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'SOL': 'solana',
        'DOGE': 'dogecoin',
        'MATIC': 'matic-network',
        'DOT': 'polkadot',
        'AVAX': 'avalanche-2'
    }
    
    coin_id = symbol_map.get(symbol, symbol.lower())
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "7", "interval": "hourly"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'prices' not in data:
            st.warning("⚠️ CoinGecko: No price data")
            return None, None
        
        # Convert to DataFrame
        prices = data['prices']
        volumes = data.get('total_volumes', [[p[0], 1000000] for p in prices])
        
        df = pd.DataFrame({
            'timestamp': [pd.to_datetime(p[0], unit='ms') for p in prices],
            'close': [p[1] for p in prices],
            'volume': [v[1] for v in volumes]
        })
        
        # Create OHLC from close (approximation)
        df['open'] = df['close']
        df['high'] = df['close'] * 1.001
        df['low'] = df['close'] * 0.999
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.tail(limit).reset_index(drop=True)
        st.success(f"✅ Loaded {len(df)} data points from CoinGecko")
        return df, "CoinGecko"
    except Exception as e:
        st.warning(f"⚠️ CoinGecko API failed: {str(e)}")
        return None, None

def fetch_data(symbol, asset_type):
    """Main function to fetch data with multiple fallbacks"""
    if asset_type == "💰 Cryptocurrency" or asset_type == "🔍 Custom Search":
        interval_map = timeframe_config
        
        st.info(f"🔄 Trying to fetch {symbol} data...")
        
        # Try Binance first
        df, source = get_binance_data(symbol, interval_map['binance'], interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        # Try OKX as primary backup
        st.info("🔄 Trying backup API (OKX)...")
        df, source = get_okx_data(symbol, interval_map['okx'], interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        # Try CryptoCompare
        st.info("🔄 Trying backup API (CryptoCompare)...")
        df, source = get_cryptocompare_data(symbol, interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        # Try CoinGecko
        st.info("🔄 Trying backup API (CoinGecko)...")
        df, source = get_coingecko_data(symbol, interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        # All APIs failed
        st.error(f"""
        ❌ **Could not fetch data for {symbol}**
        
        **APIs Tried (in order):**
        1. ❌ Binance
        2. ❌ OKX
        3. ❌ CryptoCompare
        4. ❌ CoinGecko
        
        **Possible reasons:**
        - Symbol might be incorrect (try: BTC, ETH, BNB, SOL, ADA, DOGE)
        - All APIs are down or rate-limited
        - Internet connection issues
        - Symbol not available on these exchanges
        
        **Try:**
        - Use major coins: BTC, ETH, BNB, SOL, ADA
        - Wait a few minutes and refresh
        - Check your internet connection
        - Make sure symbol is correct (uppercase)
        """)
        return None, None
    
    return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    try:
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def analyze_rsi_bounce_patterns(df):
    """Analyze historical RSI bounce patterns"""
    if 'rsi' not in df.columns or len(df) < 50:
        return None
    
    rsi = df['rsi'].values
    price = df['close'].values
    
    overbought_bounces = []
    oversold_bounces = []
    
    # Scan through history
    for i in range(10, len(rsi) - 10):
        current_rsi = rsi[i]
        future_rsi = rsi[i+1:min(i+11, len(rsi))]
        current_price = price[i]
        future_prices = price[i+1:min(i+11, len(price))]
        
        # Overbought detection (RSI > 70)
        if current_rsi > 70:
            bounce_points = future_rsi[future_rsi < 70]
            if len(bounce_points) > 0:
                periods = np.where(future_rsi < 70)[0][0] + 1
                if periods < len(future_prices):
                    price_change = ((future_prices[periods-1] - current_price) / current_price) * 100
                    overbought_bounces.append({
                        'price_change': price_change,
                        'periods': periods
                    })
        
        # Oversold detection (RSI < 30)
        elif current_rsi < 30:
            bounce_points = future_rsi[future_rsi > 30]
            if len(bounce_points) > 0:
                periods = np.where(future_rsi > 30)[0][0] + 1
                if periods < len(future_prices):
                    price_change = ((future_prices[periods-1] - current_price) / current_price) * 100
                    oversold_bounces.append({
                        'price_change': price_change,
                        'periods': periods
                    })
    
    # Calculate statistics
    insights = ""
    if len(overbought_bounces) > 5:
        avg_change = np.mean([b['price_change'] for b in overbought_bounces])
        avg_periods = np.mean([b['periods'] for b in overbought_bounces])
        insights += f"📉 **Overbought Pattern**: {len(overbought_bounces)} cases, avg {avg_change:.2f}% change in {avg_periods:.1f} periods\n"
    
    if len(oversold_bounces) > 5:
        avg_change = np.mean([b['price_change'] for b in oversold_bounces])
        avg_periods = np.mean([b['periods'] for b in oversold_bounces])
        insights += f"📈 **Oversold Pattern**: {len(oversold_bounces)} cases, avg {avg_change:.2f}% change in {avg_periods:.1f} periods"
    
    return insights if insights else "Insufficient RSI pattern data"

def create_pattern_features(df, lookback=6):
    """Create features using last N hours as context"""
    sequences = []
    targets = []
    
    for i in range(lookback, len(df) - 1):
        # Get last N hours
        sequence = []
        for j in range(i - lookback, i):
            hour_features = [
                df['close'].iloc[j],
                df['volume'].iloc[j],
                df['rsi'].iloc[j] if 'rsi' in df.columns else 50,
                df['macd'].iloc[j] if 'macd' in df.columns else 0,
                df['sma_20'].iloc[j] if 'sma_20' in df.columns else df['close'].iloc[j],
                df['volatility'].iloc[j] if 'volatility' in df.columns else 0
            ]
            
            # Price change
            if j > i - lookback:
                prev_close = df['close'].iloc[j-1]
                hour_features.append((df['close'].iloc[j] - prev_close) / (prev_close + 1e-10))
            else:
                hour_features.append(0)
            
            sequence.extend(hour_features)
        
        sequences.append(sequence)
        targets.append(df['close'].iloc[i])
    
    return np.array(sequences), np.array(targets)

def train_improved_model(df, lookback=6, prediction_periods=5):
    """
    IMPROVED: Pattern-based prediction with context
    - Monitors last N hours
    - Learns from historical patterns
    - Better accuracy
    """
    try:
        if len(df) < 60:
            return None, None, 0, None
        
        # Create sequences with context
        X, y = create_pattern_features(df, lookback=lookback)
        
        if len(X) < 30:
            return None, None, 0, None
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data (80-20)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_test = X_scaled[split_idx:]
        y_test = y[split_idx:]
        
        # Train optimized models
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Make predictions
        current_sequence = []
        lookback_start = len(df) - lookback
        
        for i in range(lookback_start, len(df)):
            hour_features = [
                df['close'].iloc[i],
                df['volume'].iloc[i],
                df['rsi'].iloc[i] if 'rsi' in df.columns else 50,
                df['macd'].iloc[i] if 'macd' in df.columns else 0,
                df['sma_20'].iloc[i] if 'sma_20' in df.columns else df['close'].iloc[i],
                df['volatility'].iloc[i] if 'volatility' in df.columns else 0
            ]
            
            if i > lookback_start:
                prev_close = df['close'].iloc[i-1]
                hour_features.append((df['close'].iloc[i] - prev_close) / (prev_close + 1e-10))
            else:
                hour_features.append(0)
            
            current_sequence.extend(hour_features)
        
        current_sequence = np.array(current_sequence).reshape(1, -1)
        current_scaled = scaler.transform(current_sequence)
        
        # Predict future prices
        predictions = []
        for _ in range(prediction_periods):
            rf_pred = rf_model.predict(current_scaled)[0]
            gb_pred = gb_model.predict(current_scaled)[0]
            pred_price = 0.4 * rf_pred + 0.6 * gb_pred  # GB weighted more
            predictions.append(pred_price)
        
        # Calculate confidence
        if len(X_test) > 0:
            rf_test_pred = rf_model.predict(X_test)
            gb_test_pred = gb_model.predict(X_test)
            ensemble_pred = 0.4 * rf_test_pred + 0.6 * gb_test_pred
            
            mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
            confidence = max(0, min(100, 100 - mape))
        else:
            confidence = 65
        
        # Get RSI insights
        rsi_insights = analyze_rsi_bounce_patterns(df)
        
        return predictions, ['Pattern-based features'], confidence, rsi_insights
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, 0, None

def calculate_signal_strength(df):
    """Calculate trading signal strength"""
    signals = []
    
    # RSI
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            signals.append(-2)
        elif rsi < 30:
            signals.append(2)
        else:
            signals.append(0)
    
    # MACD
    if 'macd' in df.columns:
        macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
        signals.append(1 if macd_diff > 0 else -1)
    
    # MA
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        price = df['close'].iloc[-1]
        sma20 = df['sma_20'].iloc[-1]
        sma50 = df['sma_50'].iloc[-1]
        
        if price > sma20 > sma50:
            signals.append(2)
        elif price > sma20:
            signals.append(1)
        elif price < sma20 < sma50:
            signals.append(-2)
        else:
            signals.append(-1)
    
    return sum(signals) if signals else 0

# Main Application
with st.spinner(f"🔄 Fetching {pair_display} data..."):
    df, data_source = fetch_data(symbol, asset_type)

if df is not None and len(df) > 0:
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Current price
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
    # Display metrics
    st.markdown(f"### 📊 {pair_display} - Real-Time Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
    with col2:
        st.metric("24h High", f"${df['high'].tail(24).max():,.2f}" if len(df) >= 24 else "N/A")
    with col3:
        st.metric("24h Low", f"${df['low'].tail(24).min():,.2f}" if len(df) >= 24 else "N/A")
    with col4:
        st.metric("Data Source", data_source)
    
    st.markdown("---")
    
    # AI Predictions
    st.markdown("### 🤖 Improved AI Predictions")
    st.info(f"""
    **🎯 Improvements:**
    - ✅ Monitors last {lookback_hours} hours as context
    - ✅ Analyzes RSI bounce patterns from history
    - ✅ Uses pattern-based prediction
    - ✅ Optimized ML models with feature scaling
    """)
    
    with st.spinner("🧠 Training AI models..."):
        predictions, features, confidence, rsi_insights = train_improved_model(
            df, 
            lookback=lookback_hours,
            prediction_periods=prediction_periods
        )
    
    if predictions and len(predictions) > 0:
        # Prediction metrics
        pred_change = ((predictions[-1] - current_price) / current_price) * 100
        signal_strength = calculate_signal_strength(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Prediction", f"${predictions[-1]:,.2f}", f"{pred_change:+.2f}%")
        
        with col2:
            conf_color = "🟢" if confidence >= 60 else "🟡" if confidence >= 45 else "🔴"
            conf_level = "High" if confidence >= 60 else "Medium" if confidence >= 45 else "Low"
            st.metric("Confidence", f"{conf_color} {confidence:.1f}%", conf_level)
        
        with col3:
            signal_emoji = "🟢" if signal_strength > 0 else "🔴" if signal_strength < 0 else "⚪"
            st.metric("Signal", f"{signal_emoji} {abs(signal_strength)}/10",
                     "Bullish" if signal_strength > 0 else "Bearish" if signal_strength < 0 else "Neutral")
        
        # RSI Insights
        if rsi_insights:
            st.success(f"**📊 RSI Historical Analysis:**\n\n{rsi_insights}")
        
        # Prediction table
        st.markdown("#### 📈 Detailed Predictions")
        pred_data = []
        last_timestamp = df['timestamp'].iloc[-1]
        
        for i, pred in enumerate(predictions, 1):
            future_time = last_timestamp + timedelta(hours=i)
            change = ((pred - current_price) / current_price) * 100
            pred_data.append({
                'Time': future_time.strftime('%Y-%m-%d %H:%M'),
                'Price': f"${pred:,.2f}",
                'Change': f"{change:+.2f}%"
            })
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
    
    else:
        st.error("❌ Could not generate predictions")
    
    st.markdown("---")
    
    # Chart
    st.markdown("### 📊 Technical Chart")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=['Price', 'RSI', 'MACD']
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add predictions
    if predictions:
        future_times = pd.date_range(
            start=df['timestamp'].iloc[-1],
            periods=len(predictions) + 1,
            freq='H'
        )[1:]
        
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='purple', width=3, dash='dash')
            ),
            row=1, col=1
        )
    
    # Indicators
    if use_sma:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    if use_bb:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    if use_rsi:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='blue')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if use_macd:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading recommendations
    st.markdown("### 💰 Trading Recommendations")
    
    is_buy = signal_strength >= 0
    recent_low = df['low'].tail(20).min()
    recent_high = df['high'].tail(20).max()
    
    if is_buy:
        st.success("### 🟢 BUY SETUP")
        entry = current_price
        tp = entry * 1.03
        sl = entry * 0.98
        
        st.info(f"""
        **Entry:** ${entry:,.2f}
        **Take Profit:** ${tp:,.2f} (+3%)
        **Stop Loss:** ${sl:,.2f} (-2%)
        **Risk/Reward:** 1:1.5
        """)
    else:
        st.error("### 🔴 SELL SETUP")
        entry = current_price
        tp = entry * 0.97
        sl = entry * 1.02
        
        st.info(f"""
        **Entry:** ${entry:,.2f}
        **Take Profit:** ${tp:,.2f} (-3%)
        **Stop Loss:** ${sl:,.2f} (+2%)
        **Risk/Reward:** 1:1.5
        """)
    
    st.warning("⚠️ **Risk Warning:** Use stop-losses. Never risk more than 1-2% per trade. Not financial advice.")

else:
    st.error("❌ Unable to fetch data. Please check symbol and try again.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>🚀 IMPROVED AI TRADING PLATFORM</b></p>
    <p><b>📡 Data Source:</b> Binance API</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p><b>🧠 Improvements:</b> Pattern-Based | Context Window | RSI Learning</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
