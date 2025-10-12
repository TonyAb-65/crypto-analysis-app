import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Live Crypto AI Analysis", layout="wide", page_icon="🤖")

# Title
st.title("🤖 Live Cryptocurrency AI Analysis & Trading Signals")
st.markdown("*Real-time data from multiple exchanges with ML predictions*")

# Display current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Exchange selection
exchange = st.sidebar.selectbox(
    "Select Exchange",
    ["Binance", "Multi-Source (Binance + CoinGecko)"],
    index=0
)

# Popular trading pairs
POPULAR_PAIRS = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USDT": "ADAUSDT",
    "SOL/USDT": "SOLUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "MATIC/USDT": "MATICUSDT",
    "DOT/USDT": "DOTUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "LINK/USDT": "LINKUSDT",
    "UNI/USDT": "UNIUSDT",
    "ATOM/USDT": "ATOMUSDT",
    "LTC/USDT": "LTCUSDT",
    "TRX/USDT": "TRXUSDT"
}

pair_display = st.sidebar.selectbox("Select Trading Pair", list(POPULAR_PAIRS.keys()), index=0)
symbol = POPULAR_PAIRS[pair_display]

# Timeframe selection
TIMEFRAMES = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
    "1 week": "1w"
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=4)
interval = TIMEFRAMES[timeframe_name]

# Number of candles
limit = st.sidebar.slider("Number of Candles", 100, 500, 300)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)

# AI Model Selection
st.sidebar.markdown("### 🤖 AI Configuration")
ai_model = st.sidebar.selectbox(
    "Prediction Model",
    ["Ensemble (Recommended)", "Random Forest", "Gradient Boosting"],
    index=0
)

prediction_periods = st.sidebar.slider("Prediction Periods", 1, 20, 5)

# Technical Indicators
st.sidebar.markdown("### 📊 Technical Indicators")
use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
use_ema = st.sidebar.checkbox("EMA (20, 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (14)", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# API Functions
def get_binance_data(symbol, interval, limit):
    """Fetch live data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['source'] = 'Binance'
        return df
    except Exception as e:
        st.error(f"❌ Binance API Error: {e}")
        return None

def get_binance_ticker(symbol):
    """Get current ticker info from Binance"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_coingecko_price(coin_id):
    """Get current price from CoinGecko API"""
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
        "include_market_cap": "true"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_coingecko_market_data(coin_id):
    """Get detailed market data from CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None

# Coin ID mapping for CoinGecko
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "SOLUSDT": "solana",
    "DOGEUSDT": "dogecoin",
    "MATICUSDT": "matic-network",
    "DOTUSDT": "polkadot",
    "AVAXUSDT": "avalanche-2",
    "LINKUSDT": "chainlink",
    "UNIUSDT": "uniswap",
    "ATOMUSDT": "cosmos",
    "LTCUSDT": "litecoin",
    "TRXUSDT": "tron"
}

# Technical Indicators
def calculate_sma(df, period=20):
    return df['close'].rolling(window=period).mean()

def calculate_ema(df, period=20):
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, period=20, std=2):
    sma = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def create_features(df):
    """Create ML features"""
    df_feat = df.copy()
    
    df_feat['price_change'] = df_feat['close'].pct_change()
    df_feat['high_low_diff'] = df_feat['high'] - df_feat['low']
    df_feat['price_momentum'] = df_feat['close'] - df_feat['close'].shift(5)
    
    for period in [5, 10, 20, 50]:
        df_feat[f'sma_{period}'] = df_feat['close'].rolling(window=period).mean()
        df_feat[f'ema_{period}'] = df_feat['close'].ewm(span=period, adjust=False).mean()
    
    df_feat['rsi_14'] = calculate_rsi(df_feat, 14)
    
    macd, signal, hist = calculate_macd(df_feat)
    df_feat['macd'] = macd
    df_feat['macd_signal'] = signal
    df_feat['macd_hist'] = hist
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df_feat)
    df_feat['bb_upper'] = bb_upper
    df_feat['bb_middle'] = bb_middle
    df_feat['bb_lower'] = bb_lower
    df_feat['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    df_feat['volume_sma'] = df_feat['volume'].rolling(window=20).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / df_feat['volume_sma']
    df_feat['volatility'] = df_feat['close'].rolling(window=20).std()
    
    for i in [1, 2, 3, 5, 10]:
        df_feat[f'close_lag_{i}'] = df_feat['close'].shift(i)
    
    return df_feat

def train_ml_model(df, model_type='Ensemble (Recommended)', periods_ahead=5):
    """Train ML model"""
    df_ml = create_features(df)
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 50:
        return None, None, None
    
    df_ml['target'] = df_ml['close'].shift(-periods_ahead)
    df_ml = df_ml.dropna()
    
    feature_cols = [col for col in df_ml.columns if col not in 
                   ['timestamp', 'close_time', 'quote_volume', 'trades', 
                    'taker_buy_base', 'taker_buy_quote', 'ignore', 'target', 
                    'open', 'high', 'low', 'close', 'source']]
    
    X = df_ml[feature_cols]
    y = df_ml['target']
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values)
        return model, X.columns.tolist(), score
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values)
        return model, X.columns.tolist(), score
    else:  # Ensemble
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        predictions = (rf_pred + gb_pred) / 2
        
        score = 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values)
        
        return (rf_model, gb_model), X.columns.tolist(), score

def predict_future_prices(df, model, feature_cols, model_type, periods=5):
    """Generate predictions"""
    df_pred = create_features(df)
    df_pred = df_pred.dropna()
    
    predictions = []
    current_data = df_pred.iloc[-1:].copy()
    
    for i in range(periods):
        X_pred = current_data[feature_cols]
        
        if model_type == 'Ensemble (Recommended)':
            rf_model, gb_model = model
            pred = (rf_model.predict(X_pred)[0] + gb_model.predict(X_pred)[0]) / 2
        else:
            pred = model.predict(X_pred)[0]
        
        predictions.append(pred)
        
        new_row = current_data.iloc[-1].copy()
        new_row['close'] = pred
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        current_data = create_features(current_data).iloc[-1:]
    
    return predictions

def generate_signals(df):
    """Generate trading signals"""
    signals = []
    signal_strength = 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    if 'rsi' in df.columns:
        if latest['rsi'] < 30:
            signals.append("🟢 RSI Oversold (<30) - Strong BUY")
            signal_strength += 2
        elif latest['rsi'] > 70:
            signals.append("🔴 RSI Overbought (>70) - Strong SELL")
            signal_strength -= 2
        elif 30 <= latest['rsi'] <= 45:
            signals.append("🟡 RSI Neutral-Bullish")
            signal_strength += 1
        elif 55 <= latest['rsi'] <= 70:
            signals.append("🟡 RSI Neutral-Bearish")
            signal_strength -= 1
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            signals.append("🟢 MACD Bullish Crossover - BUY")
            signal_strength += 3
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            signals.append("🔴 MACD Bearish Crossover - SELL")
            signal_strength -= 3
        elif latest['macd'] > latest['macd_signal']:
            signals.append("🟢 MACD Above Signal - Bullish")
            signal_strength += 1
        else:
            signals.append("🔴 MACD Below Signal - Bearish")
            signal_strength -= 1
    
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
            signals.append("🟢 Golden Cross - Strong BUY")
            signal_strength += 3
        elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
            signals.append("🔴 Death Cross - Strong SELL")
            signal_strength -= 3
    
    if 'ema_20' in df.columns:
        if latest['close'] > latest['ema_20']:
            signals.append("🟢 Price Above EMA20 - Bullish")
            signal_strength += 1
        else:
            signals.append("🔴 Price Below EMA20 - Bearish")
            signal_strength -= 1
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        if latest['close'] <= latest['bb_lower']:
            signals.append("🟢 At Lower BB - Potential BUY")
            signal_strength += 2
        elif latest['close'] >= latest['bb_upper']:
            signals.append("🔴 At Upper BB - Potential SELL")
            signal_strength -= 2
    
    if 'volume' in df.columns:
        avg_volume = df['volume'].tail(20).mean()
        if latest['volume'] > avg_volume * 1.5:
            signals.append("📊 High Volume - Strong Movement")
            signal_strength += 1
    
    return signals, signal_strength

# Main App
if st.sidebar.button("🔄 Refresh Now", type="primary"):
    st.rerun()

# Live Multi-Source Data
st.markdown("### 📡 Live Market Data from Multiple Sources")

col1, col2, col3 = st.columns(3)

# Fetch live data
with st.spinner("🔄 Fetching live data..."):
    df = get_binance_data(symbol, interval, limit)
    ticker_data = get_binance_ticker(symbol)
    
    if symbol in COINGECKO_IDS:
        coingecko_data = get_coingecko_price(COINGECKO_IDS[symbol])
        coingecko_market = get_coingecko_market_data(COINGECKO_IDS[symbol])
    else:
        coingecko_data = None
        coingecko_market = None

# Display live prices from multiple sources
if df is not None and len(df) > 50:
    current_price = df['close'].iloc[-1]
    
    with col1:
        st.markdown("#### 🟢 Binance (Live)")
        if ticker_data:
            change_24h = float(ticker_data['priceChangePercent'])
            st.metric(
                "Current Price",
                f"${current_price:.4f}",
                f"{change_24h:+.2f}%"
            )
            st.write(f"**Volume (24h):** {float(ticker_data['volume']):.0f}")
        else:
            st.metric("Current Price", f"${current_price:.4f}")
    
    with col2:
        st.markdown("#### 🔵 CoinGecko (Live)")
        if coingecko_data:
            coin_id = COINGECKO_IDS[symbol]
            price_cg = coingecko_data[coin_id]['usd']
            change_cg = coingecko_data[coin_id].get('usd_24h_change', 0)
            st.metric(
                "Current Price",
                f"${price_cg:.4f}",
                f"{change_cg:+.2f}%"
            )
            if 'usd_24h_vol' in coingecko_data[coin_id]:
                st.write(f"**Volume (24h):** ${coingecko_data[coin_id]['usd_24h_vol']:,.0f}")
        else:
            st.info("CoinGecko data unavailable")
    
    with col3:
        st.markdown("#### 📊 Market Overview")
        if coingecko_market:
            market_data = coingecko_market.get('market_data', {})
            if 'market_cap' in market_data:
                mcap = market_data['market_cap'].get('usd', 0)
                st.write(f"**Market Cap:** ${mcap:,.0f}")
            if 'total_volume' in market_data:
                vol = market_data['total_volume'].get('usd', 0)
                st.write(f"**Total Volume:** ${vol:,.0f}")
            if 'market_cap_rank' in market_data:
                st.write(f"**Rank:** #{market_data.get('market_cap_rank', 'N/A')}")
        else:
            st.write(f"**24h High:** ${df['high'].max():.4f}")
            st.write(f"**24h Low:** ${df['low'].min():.4f}")
    
    st.markdown("---")
    
    # Calculate all indicators
    if use_sma:
        df['sma_20'] = calculate_sma(df, 20)
        df['sma_50'] = calculate_sma(df, 50)
    
    if use_ema:
        df['ema_20'] = calculate_ema(df, 20)
        df['ema_50'] = calculate_ema(df, 50)
    
    if use_rsi:
        df['rsi'] = calculate_rsi(df)
    
    if use_macd:
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    
    if use_bb:
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df)
    
    # AI Model Training
    st.markdown("### 🤖 AI Price Prediction Engine")
    
    with st.spinner("🧠 Training AI model on live data..."):
        model, feature_cols, accuracy = train_ml_model(df, ai_model, prediction_periods)
    
    if model is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AI Model", ai_model)
        with col2:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with col3:
            st.metric("Data Points", len(df))
        with col4:
            st.metric("Timeframe", timeframe_name)
        
        # Generate predictions
        future_prices = predict_future_prices(df, model, feature_cols, ai_model, prediction_periods)
        
        st.markdown("#### 🎯 AI Price Predictions")
        pred_cols = st.columns(min(5, prediction_periods))
        
        for i, pred_price in enumerate(future_prices):
            with pred_cols[i % 5]:
                change_pct = ((pred_price - current_price) / current_price) * 100
                st.metric(
                    f"+{i+1}",
                    f"${pred_price:.4f}",
                    f"{change_pct:+.2f}%",
                    delta_color="normal"
                )
        
        # AI Recommendation
        avg_prediction = np.mean(future_prices)
        prediction_trend = "BULLISH 🚀" if avg_prediction > current_price else "BEARISH 🔻"
        expected_change = ((avg_prediction - current_price) / current_price) * 100
        
        if avg_prediction > current_price:
            st.success(f"### ✅ AI PREDICTION: {prediction_trend}")
            st.success(f"Expected price movement: **+{expected_change:.2f}%** over {prediction_periods} periods")
        else:
            st.error(f"### ⚠️ AI PREDICTION: {prediction_trend}")
            st.error(f"Expected price movement: **{expected_change:.2f}%** over {prediction_periods} periods")
    
    st.markdown("---")
    
    # Trading Signals
    signals, signal_strength = generate_signals(df)
    
    st.markdown("### 🎯 Live Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if signal_strength >= 5:
            st.success("## 🟢 STRONG BUY")
            st.markdown("**Action:** Enter LONG position")
        elif signal_strength >= 2:
            st.success("## 🟢 BUY")
            st.markdown("**Action:** Consider LONG")
        elif signal_strength <= -5:
            st.error("## 🔴 STRONG SELL")
            st.markdown("**Action:** Enter SHORT/Exit")
        elif signal_strength <= -2:
            st.error("## 🔴 SELL")
            st.markdown("**Action:** Consider exit")
        else:
            st.warning("## 🟡 NEUTRAL")
            st.markdown("**Action:** Wait for signals")
        
        st.metric("Signal Strength", f"{signal_strength}/10")
    
    with col2:
        st.markdown("#### 📋 Signal Breakdown:")
        for signal in signals:
            st.markdown(f"- {signal}")
    
    st.markdown("---")
    
    # Interactive Chart
    st.markdown("### 📈 Live Price Chart with Technical Analysis")
    
    # Create comprehensive chart
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(f'{pair_display} - {timeframe_name}', 'Volume', 'RSI (14)', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add predictions to chart
    if model is not None and len(future_prices) > 0:
        last_timestamp = df['timestamp'].iloc[-1]
        time_deltas = {
            '1m': timedelta(minutes=1), '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15), '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1), '4h': timedelta(hours=4),
            '1d': timedelta(days=1), '1w': timedelta(weeks=1)
        }
        delta = time_deltas.get(interval, timedelta(hours=1))
        
        future_timestamps = [last_timestamp + delta * (i + 1) for i in range(len(future_prices))]
        
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=future_prices,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='purple', width=3, dash='dash'),
                marker=dict(size=10, symbol='star', color='purple')
            ),
            row=1, col=1
        )
    
    # Moving Averages
    if use_sma:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
    
    if use_ema:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', width=1.5, dash='dot')), row=1, col=1)
    
    # Bollinger Bands
    if use_bb:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash'), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    
    # Volume bars
    colors = ['#ef5350' if df['close'].iloc[i] < df['open'].iloc[i] else '#26a69a' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # RSI
    if use_rsi:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # MACD
    if use_macd:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red', width=2)), row=4, col=1)
        colors_macd = ['#26a69a' if val > 0 else '#ef5350' for val in df['macd_hist']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], name='Histogram', marker_color=colors_macd, showlegend=False), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Entry/Exit Points
    st.markdown("### 💰 Suggested Entry & Exit Points (Based on Live Data)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("#### 🟢 BUY ZONES (Support Levels)")
        if 'bb_lower' in df.columns:
            st.write(f"✓ Lower Bollinger Band: **${df['bb_lower'].iloc[-1]:.4f}**")
        if 'sma_50' in df.columns:
            st.write(f"✓ SMA 50: **${df['sma_50'].iloc[-1]:.4f}**")
        recent_low = df['low'].tail(20).min()
        st.write(f"✓ Recent Support: **${recent_low:.4f}**")
        if model is not None and len(future_prices) > 0:
            min_pred = min(future_prices)
            st.write(f"✓ AI Predicted Low: **${min_pred:.4f}**")
    
    with col2:
        st.error("#### 🔴 SELL ZONES (Resistance Levels)")
        if 'bb_upper' in df.columns:
            st.write(f"✓ Upper Bollinger Band: **${df['bb_upper'].iloc[-1]:.4f}**")
        recent_high = df['high'].tail(20).max()
        st.write(f"✓ Recent Resistance: **${recent_high:.4f}**")
        if model is not None and len(future_prices) > 0:
            max_pred = max(future_prices)
            st.write(f"✓ AI Predicted High: **${max_pred:.4f}**")
    
    # Risk Management
    st.markdown("---")
    st.markdown("### ⚠️ Risk Management & Disclaimer")
    st.warning("""
    **Important Trading Guidelines:**
    
    ✓ **Stop Loss:** Always set stop-loss orders (recommended 2-3% below entry)
    
    ✓ **Position Sizing:** Never risk more than 1-2% of your portfolio on a single trade
    
    ✓ **Diversification:** Spread investments across multiple assets
    
    ✓ **AI Limitations:** Predictions are probabilistic, not guarantees
    
    ✓ **Market Volatility:** Crypto markets are highly volatile - trade with caution
    
    ✓ **Do Your Research:** This is a tool to assist analysis, not financial advice
    """)

else:
    st.error("❌ Unable to fetch live data. Please check your internet connection and try again.")
    st.info("💡 Tips: Try refreshing the page or selecting a different trading pair.")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px;'>
    <p><b>📡 Live Data Sources:</b> Binance API • CoinGecko API</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ This tool is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
