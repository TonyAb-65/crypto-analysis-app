import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
import time
import base64
from io import BytesIO
import os

warnings.filterwarnings('ignore')

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

st.title("🤖 AI Trading Analysis Platform")
st.markdown("*Crypto, Forex, Metals + AI Chart Image Analysis*")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")
st.markdown("---")

# =========================================================
# Sidebar configuration
# =========================================================
st.sidebar.header("⚙️ Configuration")

asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search", "📸 Analyze Chart Image"],
    index=0
)

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
    Examples:
    - Crypto: BTC, ETH, DOGE
    - Forex: EUR/USD, GBP/JPY
    - Metals: XAU/USD, XAG/USD
    - Stocks: AAPL, TSLA, GOOGL
    """)
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol
else:
    pair_display = "Chart Analysis"
    symbol = None

# =========================================================
# Timeframes
# =========================================================
TIMEFRAMES = {
    "5 Minutes":  {"limit": 200, "binance": "5m",  "okx": "5m",  "interval_label": "5m"},
    "15 Minutes": {"limit": 200, "binance": "15m", "okx": "15m", "interval_label": "15m"},
    "1 Hour":     {"limit": 200, "binance": "1h",  "okx": "1H",  "interval_label": "1h"},
    "4 Hours":    {"limit": 200, "binance": "4h",  "okx": "4H",  "interval_label": "4h"},
    "1 Day":      {"limit": 200, "binance": "1d",  "okx": "1D",  "interval_label": "1d"},
}

if asset_type != "📸 Analyze Chart Image":
    timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=2)
    timeframe_config = TIMEFRAMES[timeframe_name]
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)

    st.sidebar.markdown("### 🤖 AI Configuration")
    ai_model_choice = st.sidebar.selectbox(
        "Prediction Model",
        ["Ensemble (Recommended)", "Random Forest", "Gradient Boosting"],
        index=0
    )

    prediction_periods = st.sidebar.slider("Forecast Horizon (bars to show visually)", 1, 20, 5)

    st.sidebar.markdown("### 📊 Technical Indicators")
    use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
    use_ema = st.sidebar.checkbox("EMA (20, 50)", value=True)
    use_rsi = st.sidebar.checkbox("RSI (12, 16, 24)", value=True)
    use_macd = st.sidebar.checkbox("MACD", value=True)
    use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
else:
    auto_refresh = False
    timeframe_name = "N/A"
    timeframe_config = {"limit": 0, "binance": "1h", "okx": "1H", "interval_label": "1h"}
    ai_model_choice = "Ensemble (Recommended)"
    prediction_periods = 5
    use_sma = use_ema = use_rsi = use_macd = use_bb = False

# =========================================================
# Data fetchers
# =========================================================
@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=200):
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
    except:
        return None, None

@st.cache_data(ttl=300)
def get_binance_data(symbol, interval="1h", limit=200):
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
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df, "Binance"
    except:
        return None, None

@st.cache_data(ttl=300)
def get_cryptocompare_data(symbol, limit=200):
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') != 'Success':
            return None, None

        hist_data = data.get('Data', {}).get('Data', [])
        if not hist_data:
            return None, None

        df = pd.DataFrame(hist_data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volumefrom': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df, "CryptoCompare"
    except:
        return None, None

@st.cache_data(ttl=300)
def get_metal_data(symbol):
    api_key = "demo"  # TODO: replace with env var
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": 200,
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['timestamp'] = pd.to_datetime(df['datetime'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = df['volume'].fillna(0)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df, "Twelve Data"
    except:
        pass

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo", interval="1h")
        if not data.empty:
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            data = data.sort_values('timestamp').reset_index(drop=True)
            return data, "Yahoo Finance"
    except:
        pass
    return None, None

@st.cache_data(ttl=300)
def get_forex_data(symbol):
    api_key = "demo"  # TODO: replace with env var
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": 200,
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            # true FX volume is not available
            df['volume'] = np.nan
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df, "Twelve Data"
    except:
        pass
    return None, None

def fetch_data(symbol, asset_type, timeframe_cfg):
    if asset_type == "💰 Cryptocurrency" or asset_type == "🔍 Custom Search":
        df, source = get_okx_data(symbol, timeframe_cfg['okx'], timeframe_cfg['limit'])
        if df is not None:
            return df, source

        df, source = get_binance_data(symbol, timeframe_cfg['binance'], timeframe_cfg['limit'])
        if df is not None:
            return df, source

        df, source = get_cryptocompare_data(symbol, timeframe_cfg['limit'])
        if df is not None:
            return df, source

    elif asset_type == "🏆 Precious Metals":
        df, source = get_metal_data(symbol)
        if df is not None:
            return df, source

    elif asset_type == "💱 Forex":
        df, source = get_forex_data(symbol)
        if df is not None:
            return df, source

    return None, None

# =========================================================
# Indicators and features
# =========================================================
def calculate_technical_indicators(df):
    try:
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # RSI variants
        for period in [12, 16, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Price change and volatility
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()

        # Range features
        df['rolling_high_20'] = df['high'].rolling(20).max()
        df['rolling_low_20'] = df['low'].rolling(20).min()
        df['dist_from_high_20'] = (df['close'] / df['rolling_high_20']) - 1
        df['dist_from_low_20'] = (df['close'] / df['rolling_low_20']) - 1

        # EMA slope
        df['ema20_slope'] = df['ema_20'].diff()

        # High/low spread normalized
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def detect_regime(row):
    if pd.isna(row['ema_20']) or pd.isna(row['ema_50']) or pd.isna(row['ema20_slope']):
        return "unknown"
    if row['ema_20'] > row['ema_50'] and row['ema20_slope'] > 0:
        return "up"
    if row['ema_20'] < row['ema_50'] and row['ema20_slope'] < 0:
        return "down"
    return "chop"

def build_feature_matrix(df):
    """
    Build features at time t to predict close at t+1.
    Shift features by 1 bar to avoid leakage.
    """
    base_cols = [
        'close','price_change','volatility',
        'sma_20','sma_50','ema_20','ema_50',
        'rsi_12','rsi_16','rsi_24',
        'macd','macd_signal','macd_hist',
        'bb_upper','bb_lower',
        'dist_from_high_20','dist_from_low_20',
        'ema20_slope','hl_spread'
    ]

    # include volume only if not all NaN
    if not df['volume'].isna().all():
        base_cols.append('volume')

    shifted = df[base_cols].shift(1)
    target = df['close']

    valid = pd.concat([shifted, target], axis=1).dropna()
    X = valid[base_cols].values
    y = valid['close'].values  # 1d already

    df['regime'] = df.apply(detect_regime, axis=1)
    regime_shifted = df['regime'].shift(1).reindex(valid.index)

    return X, y, regime_shifted.values, base_cols, valid.index

def train_models_by_regime(X, y, regimes, model_choice):
    """
    Train models for up / down / chop regimes.
    Handles small sample edge cases safely.
    """
    models = {}
    for label in ["up","down","chop"]:
        mask = (regimes == label)
        X_sub = X[mask]
        y_sub = y[mask]

        # Need enough samples to even try
        if len(X_sub) < 30:
            models[label] = None
            continue

        # split
        split_idx = int(len(X_sub)*0.8)
        if split_idx < 1:
            split_idx = 1
        if split_idx >= len(X_sub):
            split_idx = len(X_sub) - 1

        X_train, X_test = X_sub[:split_idx], X_sub[split_idx:]
        y_train, y_test = y_sub[:split_idx], y_sub[split_idx:]

        # guard against empty after split
        if len(X_train) == 0 or len(y_train) == 0:
            models[label] = None
            continue

        y_train_flat = np.ravel(y_train)
        y_test_flat = np.ravel(y_test) if len(y_test) > 0 else np.array([])

        # choose model
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train_flat)

            if len(X_test) > 0:
                test_pred = model.predict(X_test)
                mape = np.mean(np.abs((y_test_flat - test_pred) / y_test_flat)) * 100
                fit_score = max(0, 100 - mape)
            else:
                fit_score = 50.0

            models[label] = {
                "type": "single",
                "model": model,
                "fit_score": fit_score
            }

        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
            model.fit(X_train, y_train_flat)

            if len(X_test) > 0:
                test_pred = model.predict(X_test)
                mape = np.mean(np.abs((y_test_flat - test_pred) / y_test_flat)) * 100
                fit_score = max(0, 100 - mape)
            else:
                fit_score = 50.0

            models[label] = {
                "type": "single",
                "model": model,
                "fit_score": fit_score
            }

        else:
            # ensemble
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
            gb = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)

            rf.fit(X_train, y_train_flat)
            gb.fit(X_train, y_train_flat)

            if len(X_test) > 0:
                rf_pred = rf.predict(X_test)
                gb_pred = gb.predict(X_test)
                rf_mape = np.mean(np.abs((y_test_flat - rf_pred) / y_test_flat)) * 100
                gb_mape = np.mean(np.abs((y_test_flat - gb_pred) / y_test_flat)) * 100
                fit_score = max(0, 100 - ((rf_mape + gb_mape)/2))
            else:
                fit_score = 50.0

            models[label] = {
                "type": "ensemble",
                "rf": rf,
                "gb": gb,
                "fit_score": fit_score
            }

    return models

def predict_next_close(models, last_features, current_regime):
    """
    Pick model for current regime or fallback.
    """
    order = [current_regime, "up", "down", "chop"]
    for reg in order:
        m = models.get(reg)
        if m is None:
            continue
        if m["type"] == "ensemble":
            rf_pred = m["rf"].predict(last_features)[0]
            gb_pred = m["gb"].predict(last_features)[0]
            pred = 0.6 * rf_pred + 0.4 * gb_pred
            return pred, m["fit_score"]
        else:
            pred = m["model"].predict(last_features)[0]
            return pred, m["fit_score"]
    return None, None

def calculate_signal_strength(df):
    try:
        signals = []

        if 'rsi_12' in df.columns:
            rsi = df['rsi_12'].iloc[-1]
            if rsi > 70:
                signals.append(-2)
            elif rsi < 30:
                signals.append(2)
            else:
                signals.append(0)

        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
            if macd_diff > 0:
                signals.append(1)
            else:
                signals.append(-1)

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

        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            price = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            if price > bb_upper:
                signals.append(-1)
            elif price < bb_lower:
                signals.append(1)
            else:
                signals.append(0)

        total = sum(signals) if signals else 0
        return total
    except:
        return 0

def analyze_chart_image(uploaded_file):
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        _ = base64.b64encode(image_bytes).decode()
        st.image(uploaded_file, caption="Uploaded Chart", use_column_width=True)
        st.info("""
        🤖 AI Chart Analysis (Demo Mode)

        Technical Pattern: Ascending Triangle
        Bias: Bullish scenario
        Suggested breakout confirmation above resistance
        """)
        return True
    return False

# =========================================================
# Main
# =========================================================
if asset_type == "📸 Analyze Chart Image":
    st.markdown("### 📸 Upload Chart for AI Analysis")
    st.info("Upload a trading chart image for AI-powered technical analysis.")
    uploaded_file = st.file_uploader("Choose a chart image...", type=['png', 'jpg', 'jpeg'])
    analyze_chart_image(uploaded_file)

else:
    with st.spinner(f"🔄 Fetching {pair_display} data..."):
        df, data_source = fetch_data(symbol, asset_type, timeframe_config)

    if df is not None and len(df) > 0:
        df = calculate_technical_indicators(df)

        df['regime'] = df.apply(detect_regime, axis=1)

        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

        volume_24h = df['volume'].tail(24).sum(skipna=True) if len(df) >= 24 else df['volume'].sum(skipna=True)
        high_24h = df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()
        low_24h = df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                f"💰 {pair_display}",
                f"${current_price:,.2f}" if current_price < 1000 else f"${current_price:,.0f}",
                f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
            )
        with col2:
            st.metric("📊 Volume (window)", f"{volume_24h:,.0f}" if not np.isnan(volume_24h) else "N/A")
        with col3:
            st.metric("📈 Window High", f"${high_24h:,.2f}" if high_24h < 1000 else f"${high_24h:,.0f}")
        with col4:
            st.metric("📉 Window Low", f"${low_24h:,.2f}" if low_24h < 1000 else f"${low_24h:,.0f}")

        st.markdown("---")

        st.markdown("### 🤖 AI Predictions & Analysis")

        # build features and train regime models
        X, y, regimes, feature_names, valid_index = build_feature_matrix(df)

        future_prices = None
        fit_score_used = None
        predicted_next_close = None

        if len(X) >= 60:
            models = train_models_by_regime(X, y, regimes, ai_model_choice)

            current_regime = df['regime'].iloc[-1] if len(df['regime']) > 0 else "unknown"

            last_features = X[-1:].copy()
            predicted_next_close, fit_score_used = predict_next_close(models, last_features, current_regime)

            if predicted_next_close is not None:
                tmp_next = predicted_next_close

                vis_prices = [tmp_next]

                if len(df['timestamp']) >= 2:
                    last_ts = df['timestamp'].iloc[-1]
                    dt = df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]
                else:
                    last_ts = pd.Timestamp.utcnow()
                    dt = pd.Timedelta(minutes=60)

                vis_timestamps = []
                for i in range(1, prediction_periods):
                    vis_timestamps.append(last_ts + dt * i)
                    vis_prices.append(tmp_next)

                future_prices = {
                    "timestamps": vis_timestamps if len(vis_timestamps)>0 else [last_ts + dt],
                    "prices": vis_prices
                }

        col1, col2, col3 = st.columns(3)

        if predicted_next_close is not None:
            predicted_change = predicted_next_close - current_price
            predicted_change_pct = (predicted_change / current_price) * 100 if current_price != 0 else 0
            with col1:
                st.metric(
                    "📈 Predicted Next Close",
                    f"${predicted_next_close:,.2f}" if predicted_next_close < 1000 else f"${predicted_next_close:,.0f}",
                    f"{predicted_change:+.2f} ({predicted_change_pct:+.1f}%)"
                )
        else:
            with col1:
                st.metric("📈 Predicted Next Close", "N/A", " ")

        with col2:
            if fit_score_used is not None:
                st.metric("🎯 Model Fit Score", f"{fit_score_used:.1f}%")
            else:
                st.metric("🎯 Model Fit Score", "N/A")

        with col3:
            st.metric("⏱️ Forecast Horizon Used", "1 bar ahead")

        signal_strength = calculate_signal_strength(df)

        st.markdown(f"""
        **📊 Technical Bias Summary:**
        - Bias Strength Score: {signal_strength}
        - Market Regime: {df['regime'].iloc[-1] if len(df['regime'])>0 else 'unknown'}
        - Timeframe: {timeframe_name}
        - Data Source: {data_source}
        """)

        st.markdown("---")

        st.markdown("### 📈 Technical Analysis Chart")

        if use_rsi and use_macd:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'),
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
        elif use_rsi or use_macd:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'Volume', 'RSI' if use_rsi else 'MACD'),
                row_heights=[0.6, 0.2, 0.2]
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'Volume'),
                row_heights=[0.7, 0.3]
            )

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

        if use_sma:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)

        if use_ema:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_50'], name='EMA 50', line=dict(color='pink', dash='dot')), row=1, col=1)

        if use_bb:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)

        if future_prices and len(future_prices["timestamps"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=future_prices["timestamps"],
                    y=future_prices["prices"],
                    mode='lines+markers',
                    name='AI Projection (visual)',
                    line=dict(color='purple', width=3, dash='dash')
                ),
                row=1, col=1
            )

        if not df['volume'].isna().all():
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x=df['timestamp'], y=[0]*len(df), marker_color="gray", showlegend=False), row=2, col=1)

        if use_rsi:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_12'], name='RSI-12', line=dict(color='blue', width=2)), row=3 if use_macd else 3, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_16'], name='RSI-16', line=dict(color='purple', width=2)), row=3 if use_macd else 3, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_24'], name='RSI-24', line=dict(color='orange', width=2)), row=3 if use_macd else 3, col=1)

        if use_macd:
            macd_row = 4 if use_rsi else 3
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=macd_row, col=1)
            colors_macd = ['green' if val > 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], marker_color=colors_macd, showlegend=False), row=macd_row, col=1)

        fig.update_layout(height=1000, showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💰 Trade Scenario (Not advice)")

        def format_price(p):
            if p is None or np.isnan(p):
                return "N/A"
            if p >= 1000:
                return f"${p:,.0f}"
            return f"${p:,.2f}"

        recent_vol = df['volatility'].iloc[-1] if 'volatility' in df.columns else np.nan
        if np.isnan(recent_vol) or recent_vol == 0:
            recent_vol = 0.01  # fallback

        k = 1.5
        stop_dist_abs = current_price * recent_vol * k
        tp_mults = [2.0, 3.0]

        signal_strength = calculate_signal_strength(df)
        bullish_bias = signal_strength >= 2
        bearish_bias = signal_strength <= -2

        if bullish_bias or bearish_bias:
            if bullish_bias:
                st.success("#### 🟢 Bullish Scenario")
                entry_price = current_price
                stop_loss = entry_price - stop_dist_abs
                tps = [entry_price + stop_dist_abs * m for m in tp_mults]

                st.info(f"""
Entry (long bias): {format_price(entry_price)}

Take Profit Targets:
- TP1: {format_price(tps[0])}
- TP2: {format_price(tps[1])}

Protective Stop:
- SL: {format_price(stop_loss)}

Volatility (recent): {recent_vol:.4f}
Bias Strength Score: {signal_strength}
""")

            if bearish_bias:
                st.error("#### 🔴 Bearish Scenario")
                entry_price = current_price
                stop_loss = entry_price + stop_dist_abs
                tps = [entry_price - stop_dist_abs * m for m in tp_mults]

                st.warning(f"""
Entry (short bias): {format_price(entry_price)}

Take Profit Targets:
- TP1: {format_price(tps[0])}
- TP2: {format_price(tps[1])}

Protective Stop:
- SL: {format_price(stop_loss)}

Volatility (recent): {recent_vol:.4f}
Bias Strength Score: {signal_strength}
""")
        else:
            st.warning("No high-conviction scenario. Market viewed as neutral or noisy (|Bias Strength Score| < 2).")

        st.markdown("---")
        st.markdown("#### 📊 Key Technical Levels")

        recent_low = df['low'].tail(20).min()
        recent_high = df['high'].tail(20).max()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("💰 Current Price", format_price(current_price))
        with c2:
            st.metric("🟢 Local Support (20 bars low)", format_price(recent_low))
        with c3:
            st.metric("🔴 Local Resistance (20 bars high)", format_price(recent_high))

        st.markdown("---")

        st.warning("""
Risk Management:
- This dashboard estimates bias and next-bar move. It is not a guarantee.
- Volatility-based SL and TP sizing is used to normalize risk.
- Model Fit Score is historical fit, not win probability.
- Educational purposes only. Not financial advice.
""")

        if 'predicted_next_close' in locals() and predicted_next_close is not None:
            log_row = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe_name,
                "current_price": current_price,
                "predicted_next_close": predicted_next_close,
                "signal_strength": signal_strength,
                "regime": df['regime'].iloc[-1] if len(df['regime'])>0 else 'unknown',
                "fit_score_used": fit_score_used
            }

            log_path = "prediction_log.csv"
            try:
                if os.path.exists(log_path):
                    pd.DataFrame([log_row]).to_csv(log_path, mode='a', header=False, index=False)
                else:
                    pd.DataFrame([log_row]).to_csv(log_path, mode='w', header=True, index=False)
                st.caption("Prediction logged for later evaluation (prediction_log.csv).")
            except Exception as e:
                st.caption(f"Logging failed: {e}")

    else:
        st.error("❌ Unable to fetch data")

# =========================================================
# Auto-refresh
# =========================================================
if auto_refresh and asset_type != "📸 Analyze Chart Image":
    time.sleep(60)
    st.rerun()

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>📡 Data Sources:</b></p>
    <p>Crypto: OKX → Binance → CryptoCompare</p>
    <p>Metals: Twelve Data → Yahoo Finance</p>
    <p>Forex: Twelve Data API</p>
    <p>Chart Analysis: Vision model placeholder</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
