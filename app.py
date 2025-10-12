import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Live Crypto AI Analysis", layout="wide", page_icon="🤖")

# Title
st.title("🤖 Live Cryptocurrency AI Analysis & Trading Signals")
st.markdown("*Real-time data with ML predictions - Works Globally!*")

# Display current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Popular trading pairs
POPULAR_PAIRS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Binance Coin (BNB)": "binancecoin",
    "XRP": "ripple",
    "Cardano (ADA)": "cardano",
    "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin",
    "Polygon (MATIC)": "matic-network",
    "Polkadot (DOT)": "polkadot",
    "Avalanche (AVAX)": "avalanche-2",
    "Chainlink (LINK)": "chainlink",
    "Uniswap (UNI)": "uniswap",
    "Cosmos (ATOM)": "cosmos",
    "Litecoin (LTC)": "litecoin",
    "Tron (TRX)": "tron"
}

pair_display = st.sidebar.selectbox("Select Cryptocurrency", list(POPULAR_PAIRS.keys()), index=0)
coin_id = POPULAR_PAIRS[pair_display]

# Timeframe selection
TIMEFRAMES = {
    "1 hour": 1,
    "4 hours": 4,
    "12 hours": 12,
    "1 day": 24,
    "7 days": 168,
    "30 days": 720,
    "90 days": 2160
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=3)
hours = TIMEFRAMES[timeframe_name]

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)

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

# API Functions - Using CoinGecko (No restrictions)
@st.cache_data(ttl=300)
def get_coingecko_data(coin_id, days=7):
    """Fetch historical data from CoinGecko API"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly" if days <= 90 else "daily"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        prices = data['prices']
        volumes = data['total_volumes']
        
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = [v[1] for v in volumes]
        
        # Create OHLC from close prices (approximation)
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df[['open', 'close']].max(axis=1) * 1.001
        df['low'] = df[['open', 'close']].min(axis=1) * 0.999
        
        return df
    except Exception as e:
        st.error(f"❌ CoinGecko API Error: {e}")
        return None

def get_coingecko_current_price(coin_id):
    """Get current price and market data"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "true",
        "developer_data": "false"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

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
                   ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
    
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
    
    return signals, signal_strength

# Main App
if st.sidebar.button("🔄 Refresh Now", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Live Data
st.markdown("### 📡 Live Market Data (CoinGecko API)")

# Calculate days for API call
days_map = {1: 1, 4: 2, 12: 3, 24: 7, 168: 30, 720: 90, 2160: 365}
days = days_map.get(hours, 7)

col1, col2, col3 = st.columns(3)

# Fetch live data
with st.spinner("🔄 Fetching live data from CoinGecko..."):
    df = get_coingecko_data(coin_id, days)
    market_data = get_coingecko_current_price(coin_id)

if df is not None and len(df) > 50:
    current_price = df['close'].iloc[-1]
    price_change_24h = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) > 24 else 0
    
    with col1:
        st.markdown("#### 💰 Current Price")
        st.metric(
            pair_display,
            f"${current_price:,.2f}",
            f"{price_change_24h:+.2f}%"
        )
    
    with col2:
        st.markdown("#### 📊 24h Range")
        st.write(f"**High:** ${df['high'].tail(24).max():,.2f}")
        st.write(f"**Low:** ${df['low'].tail(24).min():,.2f}")
    
    with col3:
        st.markdown("#### 📈 Market Data")
        if market_data and 'market_data' in market_data:
            md = market_data['market_data']
            if 'market_cap' in md:
                mcap = md['market_cap'].get('usd', 0)
                st.write(f"**Market Cap:** ${mcap:,.0f}")
            if 'market_cap_rank' in md:
                st.write(f"**Rank:** #{md.get('market_cap_rank', 'N/A')}")
    
    st.markdown("---")
    
    # Calculate indicators
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
                    f"${pred_price:,.2f}",
                    f"{change_pct:+.2f}%"
                )
        
        # AI Recommendation
        avg_prediction = np.mean(future_prices)
        prediction_trend = "BULLISH 🚀" if avg_prediction > current_price else "BEARISH 🔻"
        expected_change = ((avg_prediction - current_price) / current_price) * 100
        
        if avg_prediction > current_price:
            st.success(f"### ✅ AI PREDICTION: {prediction_trend}")
            st.success(f"Expected movement: **+{expected_change:.2f}%** over {prediction_periods} periods")
        else:
            st.error(f"### ⚠️ AI PREDICTION: {prediction_trend}")
            st.error(f"Expected movement: **{expected_change:.2f}%** over {prediction_periods} periods")
    
    st.markdown("---")
    
    # Trading Signals
    signals, signal_strength = generate_signals(df)
    
    st.markdown("### 🎯 Live Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if signal_strength >= 5:
            st.success("## 🟢 STRONG BUY")
            st.markdown("**Action:** Enter LONG")
        elif signal_strength >= 2:
            st.success("## 🟢 BUY")
            st.markdown("**Action:** Consider LONG")
        elif signal_strength <= -5:
            st.error("## 🔴 STRONG SELL")
            st.markdown("**Action:** Exit/SHORT")
        elif signal_strength <= -2:
            st.error("## 🔴 SELL")
            st.markdown("**Action:** Consider exit")
        else:
            st.warning("## 🟡 NEUTRAL")
            st.markdown("**Action:** Wait")
        
        st.metric("Signal Strength", f"{signal_strength}/10")
    
    with col2:
        st.markdown("#### 📋 Signals:")
        for signal in signals:
            st.markdown(f"- {signal}")
    
    st.markdown("---")
    
    # Chart
    st.markdown("### 📈 Live Price Chart")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(f'{pair_display} - {timeframe_name}', 'Volume', 'RSI', 'MACD')
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
    
    # Predictions
    if model and future_prices:
        last_ts = df['timestamp'].iloc[-1]
        future_ts = [last_ts + timedelta(hours=i+1) for i in range(len(future_prices))]
        
        fig.add_trace(
            go.Scatter(
                x=future_ts,
                y=future_prices,
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
    
    if use_ema:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', dash='dot')), row=1, col=1)
    
    if use_bb:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors), row=2, col=1)
    
    # RSI
    if use_rsi:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if use_macd:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=4, col=1)
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], name='Histogram'), row=4, col=1)
    
    fig.update_layout(height=1000, showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Entry/Exit
    st.markdown("### 💰 Entry & Exit Points")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("#### 🟢 BUY ZONES")
        if 'bb_lower' in df.columns:
            st.write(f"Lower BB: **${df['bb_lower'].iloc[-1]:,.2f}**")
        st.write(f"Recent Low: **${df['low'].tail(20).min():,.2f}**")
    
    with col2:
        st.error("#### 🔴 SELL ZONES")
        if 'bb_upper' in df.columns:
            st.write(f"Upper BB: **${df['bb_upper'].iloc[-1]:,.2f}**")
        st.write(f"Recent High: **${df['high'].tail(20).max():,.2f}**")
    
    st.markdown("---")
    st.warning("""
    **⚠️ Risk Management:**
    - Use stop-loss orders (2-3% below entry)
    - Never risk more than 1-2% per trade
    - Diversify your portfolio
    - AI predictions are probabilistic
    - This is NOT financial advice
    """)

else:
    st.error("❌ Unable to fetch data. Please try again in a few moments.")
    st.info("Using CoinGecko API - works globally without restrictions!")

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>📡 Data Source:</b> CoinGecko API (Global - No Restrictions)</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
