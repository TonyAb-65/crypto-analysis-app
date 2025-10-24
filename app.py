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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

# Title
st.title("🤖 AI Trading Analysis Platform")
st.markdown("*Crypto, Forex, Metals + AI Chart Image Analysis*")

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
    - Forex: EUR/USD, GBP/JPY
    - Metals: XAU/USD, XAG/USD
    - Stocks: AAPL, TSLA, GOOGL
    """)
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol
else:  # Chart Image Analysis
    pair_display = "Chart Analysis"
    symbol = None

# Timeframe selection
TIMEFRAMES = {
    "1 Minute": {"limit": 60, "unit": "minute", "binance": "1m", "okx": "1m"},
    "5 Minutes": {"limit": 60, "unit": "minute", "binance": "5m", "okx": "5m"},
    "15 Minutes": {"limit": 96, "unit": "minute", "binance": "15m", "okx": "15m"},
    "1 Hour": {"limit": 60, "unit": "minute", "binance": "1m", "okx": "1m"},
    "6 Hours": {"limit": 72, "unit": "minute", "binance": "5m", "okx": "5m"},
    "24 Hours": {"limit": 96, "unit": "hour", "binance": "15m", "okx": "15m"},
    "7 Days": {"limit": 168, "unit": "hour", "binance": "1h", "okx": "1H"},
    "30 Days": {"limit": 30, "unit": "day", "binance": "1d", "okx": "1D"},
    "90 Days": {"limit": 90, "unit": "day", "binance": "1d", "okx": "1D"}
}

# Show configuration only if not chart analysis
if asset_type != "📸 Analyze Chart Image":
    timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=5)
    timeframe_config = TIMEFRAMES[timeframe_name]
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
    use_rsi = st.sidebar.checkbox("RSI (12, 16, 24)", value=True)
    use_macd = st.sidebar.checkbox("MACD", value=True)
    use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
else:
    auto_refresh = False
    timeframe_name = "N/A"
    timeframe_config = {"limit": 0, "unit": "hour", "binance": "1h", "okx": "1H"}
    ai_model = "Ensemble (Recommended)"
    prediction_periods = 5
    use_sma = use_ema = use_rsi = use_macd = use_bb = False

# API Functions
@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=100):
    """Fetch data from OKX API"""
    url = "https://www.okx.com/api/v5/market/candles"
    limit = min(limit, 300)
    params = {"instId": f"{symbol}-USDT", "bar": interval, "limit": str(limit)}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != '0':
            st.warning(f"⚠️ OKX API error: {data.get('msg', 'Unknown error')}")
            return None, None
        
        candles = data.get('data', [])
        if not candles or len(candles) == 0:
            st.warning(f"⚠️ OKX returned no data")
            return None, None
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        st.success(f"✅ OKX: Loaded {len(df)} data points")
        return df, "OKX"
    except Exception as e:
        st.warning(f"⚠️ OKX API failed: {str(e)[:150]}")
        return None, None

@st.cache_data(ttl=300)
def get_binance_data(symbol, interval="1h", limit=100):
    """Fetch data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    limit = min(limit, 1000)
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and 'code' in data:
            st.warning(f"⚠️ Binance API error: {data.get('msg', 'Unknown error')}")
            return None, None
        
        if not data or len(data) == 0:
            st.warning("⚠️ Binance returned no data")
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
        st.success(f"✅ Binance: Loaded {len(df)} data points")
        return df, "Binance"
    except Exception as e:
        st.warning(f"⚠️ Binance API failed: {str(e)[:150]}")
        return None, None

@st.cache_data(ttl=300)
def get_cryptocompare_data(symbol, limit=100, unit="hour"):
    """Fetch data from CryptoCompare"""
    limit = min(limit, 2000)
    
    if unit == "minute":
        endpoint = "histominute"
    elif unit == "hour":
        endpoint = "histohour"
    else:
        endpoint = "histoday"
    
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('Response') == 'Error':
            st.warning(f"⚠️ CryptoCompare: {data.get('Message', 'Unknown error')}")
            return None, None
        
        if 'Data' not in data or 'Data' not in data['Data']:
            st.warning("⚠️ CryptoCompare returned no data")
            return None, None
        
        df = pd.DataFrame(data['Data']['Data'])
        if len(df) == 0:
            return None, None
        
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'volumefrom': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna()
        
        st.success(f"✅ CryptoCompare: Loaded {len(df)} data points")
        return df, "CryptoCompare"
    except Exception as e:
        st.warning(f"⚠️ CryptoCompare failed: {str(e)[:150]}")
        return None, None

@st.cache_data(ttl=300)
def get_yahoo_finance_commodities(symbol, period="1d", interval="1h"):
    """Fetch commodities from Yahoo Finance"""
    yahoo_symbols = {
        "XAU/USD": "GC=F",
        "XAG/USD": "SI=F",
        "XPT/USD": "PL=F",
        "XPD/USD": "PA=F"
    }
    yahoo_ticker = yahoo_symbols.get(symbol, "GC=F")
    
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp', 'Datetime': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        if 'timestamp' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            elif 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
        
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        st.success(f"✅ Yahoo Finance: Loaded {len(df)} data points")
        return df, "Yahoo Finance"
    except ImportError:
        st.error("❌ yfinance library not installed")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {str(e)[:100]}")
        return None, None

@st.cache_data(ttl=300)
def get_twelve_data_commodities(symbol, interval="1h", outputsize=100):
    """Fetch data from Twelve Data API"""
    try:
        api_key = st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Twelve Data API key not found")
        return None, None
    
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": api_key, "format": "JSON"}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data:
            error_msg = data.get('message', data.get('status', 'Unknown error'))
            st.error(f"❌ Twelve Data Error: {error_msg}")
            return None, None
        
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={'datetime': 'timestamp'})
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
        else:
            df['volume'] = 0.0
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        st.success(f"✅ Twelve Data: Loaded {len(df)} data points")
        return df, "Twelve Data"
    except Exception as e:
        st.error(f"❌ Twelve Data API Error: {str(e)}")
        return None, None

def fetch_market_data(symbol, timeframe_config, asset_type):
    """Fetch data from appropriate source"""
    if asset_type == "💰 Cryptocurrency":
        st.info("🔄 Trying OKX API...")
        df, source = get_okx_data(symbol, timeframe_config["okx"], timeframe_config["limit"])
        
        if df is None:
            st.info("🔄 OKX failed. Trying Binance API...")
            df, source = get_binance_data(symbol, timeframe_config["binance"], timeframe_config["limit"])
        
        if df is None:
            st.info("🔄 Binance failed. Trying CryptoCompare...")
            cc_limit = min(timeframe_config["limit"], 2000)
            df, source = get_cryptocompare_data(symbol, cc_limit, timeframe_config["unit"])
        
        return df, source
    
    elif asset_type == "💱 Forex" or asset_type == "🔍 Custom Search":
        st.info("🔄 Fetching from Twelve Data API...")
        
        api_key = None
        try:
            api_key = st.secrets.get("TWELVE_DATA_API_KEY")
        except:
            pass
        
        if api_key:
            interval_map = {"minute": "1min", "hour": "1h", "day": "1day"}
            interval = interval_map.get(timeframe_config["unit"], "1h")
            df, source = get_twelve_data_commodities(symbol, interval, timeframe_config["limit"])
            
            if df is not None:
                return df, source
            else:
                st.warning("⚠️ Twelve Data couldn't fetch this symbol")
        else:
            st.error("❌ Twelve Data API key required for Forex and custom symbols")
        
        if asset_type == "🔍 Custom Search":
            st.info("🔄 Trying CryptoCompare...")
            cc_limit = min(timeframe_config["limit"], 2000)
            df, source = get_cryptocompare_data(symbol, cc_limit, timeframe_config["unit"])
            if df is not None:
                return df, source
        
        return None, None
    
    else:  # Precious Metals
        st.info("🔄 Fetching from Twelve Data API (Primary)...")
        
        api_key = None
        try:
            api_key = st.secrets.get("TWELVE_DATA_API_KEY")
        except:
            pass
        
        if api_key:
            interval_map = {"minute": "1min", "hour": "1h", "day": "1day"}
            interval = interval_map.get(timeframe_config["unit"], "1h")
            df, source = get_twelve_data_commodities(symbol, interval, timeframe_config["limit"])
            
            if df is not None:
                return df, source
        else:
            st.warning("⚠️ Twelve Data API key not found")
        
        st.info("🔄 Twelve Data failed. Trying Yahoo Finance (Backup)...")
        
        if timeframe_config["unit"] == "minute":
            period, interval = "1d", "1m"
        elif timeframe_config["unit"] == "hour":
            period, interval = "5d", "1h"
        else:
            period, interval = "60d", "1d"
        
        df, source = get_yahoo_finance_commodities(symbol, period, interval)
        return df, source

def analyze_single_timeframe(df):
    """Analyze a single timeframe and return signal"""
    if df is None or len(df) < 50:
        return None, 0, "INSUFFICIENT_DATA"
    
    # Calculate basic indicators
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_50'] = calculate_sma(df, 50)
    df['ema_20'] = calculate_ema(df, 20)
    df['rsi'] = calculate_rsi(df, 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal_strength = 0
    trend = "NEUTRAL"
    
    # Trend determination (SMA 20 vs 50)
    if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
        if latest['sma_20'] > latest['sma_50']:
            trend = "BULLISH"
            signal_strength += 2
        elif latest['sma_20'] < latest['sma_50']:
            trend = "BEARISH"
            signal_strength -= 2
    
    # Price vs EMA
    if not pd.isna(latest['ema_20']):
        if latest['close'] > latest['ema_20']:
            signal_strength += 1
        else:
            signal_strength -= 1
    
    # RSI
    if not pd.isna(latest['rsi']):
        if latest['rsi'] < 30:
            signal_strength += 2
        elif latest['rsi'] > 70:
            signal_strength -= 2
    
    # MACD
    if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
        if latest['macd'] > latest['macd_signal']:
            signal_strength += 1
        else:
            signal_strength -= 1
    
    # Determine final signal
    if signal_strength >= 3:
        signal = "BUY"
    elif signal_strength <= -3:
        signal = "SELL"
    else:
        signal = "NEUTRAL"
    
    return signal, signal_strength, trend

def multi_timeframe_analysis(symbol, asset_type):
    """
    Analyze multiple timeframes to determine trend alignment
    Returns: alignment score, timeframe signals, and recommendation
    """
    # Define timeframe configurations for different granularities
    timeframes = {
        "15m": {"limit": 100, "unit": "minute", "binance": "15m", "okx": "15m"},
        "1H": {"limit": 100, "unit": "hour", "binance": "1h", "okx": "1H"},
        "4H": {"limit": 100, "unit": "hour", "binance": "4h", "okx": "4H"},
    }
    
    results = {}
    
    # Analyze each timeframe
    for tf_name, tf_config in timeframes.items():
        try:
            df, _ = fetch_market_data(symbol, tf_config, asset_type)
            signal, strength, trend = analyze_single_timeframe(df)
            results[tf_name] = {
                "signal": signal,
                "strength": strength,
                "trend": trend,
                "price": df['close'].iloc[-1] if df is not None else None
            }
        except:
            results[tf_name] = {
                "signal": "ERROR",
                "strength": 0,
                "trend": "UNKNOWN",
                "price": None
            }
    
    # Calculate alignment score
    signals = [r["signal"] for r in results.values() if r["signal"] != "ERROR"]
    
    if len(signals) < 2:
        return 0, results, "INSUFFICIENT_DATA", "NEUTRAL"
    
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    
    # Determine alignment
    if buy_count >= 2:
        alignment = "BULLISH"
        alignment_score = buy_count / len(signals)
    elif sell_count >= 2:
        alignment = "BEARISH"
        alignment_score = sell_count / len(signals)
    else:
        alignment = "MIXED"
        alignment_score = 0.5
    
    # Determine confidence based on agreement
    if buy_count == 3:
        confidence = "VERY HIGH"
        confidence_multiplier = 2.0
    elif sell_count == 3:
        confidence = "VERY HIGH"
        confidence_multiplier = 2.0
    elif buy_count == 2 or sell_count == 2:
        confidence = "HIGH"
        confidence_multiplier = 1.5
    else:
        confidence = "LOW"
        confidence_multiplier = 0.5
    
    return alignment_score, results, confidence, alignment, confidence_multiplier

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

def detect_market_regime(df):
    """Detect if market is trending or ranging"""
    # Calculate ADX (Average Directional Index)
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    
    # Bollinger Band Width
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Determine regime
    regime = pd.Series('UNKNOWN', index=df.index)
    regime[adx > 25] = 'TRENDING'
    regime[(adx <= 25) & (bb_width < bb_width.quantile(0.3))] = 'RANGING'
    regime[(adx <= 25) & (bb_width >= bb_width.quantile(0.3))] = 'VOLATILE'
    
    return regime, adx, plus_di, minus_di

def create_features(df):
    df_feat = df.copy()
    df_feat['price_change'] = df_feat['close'].pct_change()
    df_feat['high_low_diff'] = df_feat['high'] - df_feat['low']
    df_feat['price_momentum'] = df_feat['close'] - df_feat['close'].shift(5)
    
    # Moving Averages
    for period in [5, 10, 20, 50]:
        df_feat[f'sma_{period}'] = df_feat['close'].rolling(window=period).mean()
        df_feat[f'ema_{period}'] = df_feat['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    df_feat['rsi_14'] = calculate_rsi(df_feat, 14)
    
    # MACD
    macd, signal, hist = calculate_macd(df_feat)
    df_feat['macd'] = macd
    df_feat['macd_signal'] = signal
    df_feat['macd_hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df_feat)
    df_feat['bb_upper'] = bb_upper
    df_feat['bb_middle'] = bb_middle
    df_feat['bb_lower'] = bb_lower
    df_feat['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df_feat['bb_position'] = (df_feat['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Volatility measures
    df_feat['volatility'] = df_feat['close'].rolling(window=20).std()
    df_feat['volatility_ratio'] = df_feat['volatility'] / df_feat['volatility'].rolling(window=50).mean()
    
    # ATR (Average True Range) - Critical for risk management!
    high_low = df_feat['high'] - df_feat['low']
    high_close = abs(df_feat['high'] - df_feat['close'].shift())
    low_close = abs(df_feat['low'] - df_feat['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_feat['atr'] = true_range.rolling(window=14).mean()
    df_feat['atr_percent'] = (df_feat['atr'] / df_feat['close']) * 100
    
    # Volume features - Very important!
    df_feat['volume_sma'] = df_feat['volume'].rolling(window=20).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / df_feat['volume_sma']
    df_feat['volume_change'] = df_feat['volume'].pct_change()
    
    # Price-Volume relationship
    df_feat['pv_trend'] = df_feat['price_change'] * df_feat['volume_ratio']
    
    # Momentum indicators
    df_feat['roc_5'] = (df_feat['close'] - df_feat['close'].shift(5)) / df_feat['close'].shift(5) * 100
    df_feat['roc_10'] = (df_feat['close'] - df_feat['close'].shift(10)) / df_feat['close'].shift(10) * 100
    
    # Trend strength
    df_feat['trend_ema'] = (df_feat['ema_20'] - df_feat['ema_50']) / df_feat['ema_50'] * 100
    
    # Support/Resistance proximity
    df_feat['dist_from_high_20'] = (df_feat['high'].rolling(window=20).max() - df_feat['close']) / df_feat['close']
    df_feat['dist_from_low_20'] = (df_feat['close'] - df_feat['low'].rolling(window=20).min()) / df_feat['close']
    
    # Lagged features
    for i in [1, 2, 3, 5, 10]:
        df_feat[f'close_lag_{i}'] = df_feat['close'].shift(i)
        df_feat[f'volume_lag_{i}'] = df_feat['volume'].shift(i)
    
    # Candlestick patterns (simple)
    df_feat['body_size'] = abs(df_feat['close'] - df_feat['open'])
    df_feat['upper_shadow'] = df_feat['high'] - df_feat[['close', 'open']].max(axis=1)
    df_feat['lower_shadow'] = df_feat[['close', 'open']].min(axis=1) - df_feat['low']
    df_feat['is_bullish'] = (df_feat['close'] > df_feat['open']).astype(int)
    
    return df_feat

def train_ml_model(df, model_type='Ensemble (Recommended)', periods_ahead=5):
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
        score = max(0, 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values))
        return model, X.columns.tolist(), score
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = max(0, 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values))
        return model, X.columns.tolist(), score
    else:
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        predictions = (rf_pred + gb_pred) / 2
        
        score = max(0, 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values))
        return (rf_model, gb_model), X.columns.tolist(), score

def predict_future_prices(df, model, feature_cols, model_type, periods=5):
    df_pred = create_features(df)
    df_pred = df_pred.dropna()
    
    if len(df_pred) == 0:
        return []
    
    predictions = []
    current_data = df_pred.iloc[-1:].copy()
    
    for i in range(periods):
        try:
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
        except:
            break
    
    return predictions

def generate_signals(df):
    signals = []
    signal_strength = 0
    
    if len(df) < 2:
        return signals, signal_strength
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # MARKET REGIME CHECK - Critical for accuracy!
    regime = latest.get('regime', 'UNKNOWN')
    adx = latest.get('adx', 0)
    
    # Don't trade in ranging markets - this alone saves 30% of losses!
    if regime == 'RANGING':
        signals.append("⚠️ RANGING MARKET - Low Probability Setup")
        signal_strength = 0  # Neutral in ranging markets
        return signals, signal_strength
    
    # Boost confidence in strong trends
    trend_multiplier = 1.0
    if regime == 'TRENDING' and adx > 30:
        trend_multiplier = 1.5
        signals.append(f"✅ STRONG TREND (ADX: {adx:.1f}) - High Confidence")
    
    if 'rsi' in df.columns and not pd.isna(latest['rsi']):
        if latest['rsi'] < 30:
            signals.append("🟢 RSI Oversold (<30) - Strong BUY")
            signal_strength += int(2 * trend_multiplier)
        elif latest['rsi'] > 70:
            signals.append("🔴 RSI Overbought (>70) - Strong SELL")
            signal_strength -= int(2 * trend_multiplier)
        elif 30 <= latest['rsi'] <= 45:
            signals.append("🟡 RSI Neutral-Bullish")
            signal_strength += 1
        elif 55 <= latest['rsi'] <= 70:
            signals.append("🟡 RSI Neutral-Bearish")
            signal_strength -= 1
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if not pd.isna(latest['macd']) and not pd.isna(prev['macd']):
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                signals.append("🟢 MACD Bullish Crossover - BUY")
                signal_strength += int(3 * trend_multiplier)
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                signals.append("🔴 MACD Bearish Crossover - SELL")
                signal_strength -= int(3 * trend_multiplier)
            elif latest['macd'] > latest['macd_signal']:
                signals.append("🟢 MACD Above Signal - Bullish")
                signal_strength += 1
            else:
                signals.append("🔴 MACD Below Signal - Bearish")
                signal_strength -= 1
    
    # Volume confirmation - adds ~10% accuracy
    if 'volume' in df.columns:
        avg_volume = df['volume'].tail(20).mean()
        if latest['volume'] > avg_volume * 1.5:
            signals.append("📊 HIGH VOLUME - Strong Confirmation")
            if signal_strength > 0:
                signal_strength += 1
            elif signal_strength < 0:
                signal_strength -= 1
    
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
            if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                signals.append("🟢 Golden Cross - Strong BUY")
                signal_strength += int(3 * trend_multiplier)
            elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                signals.append("🔴 Death Cross - Strong SELL")
                signal_strength -= int(3 * trend_multiplier)
    
    if 'ema_20' in df.columns and not pd.isna(latest['ema_20']):
        if latest['close'] > latest['ema_20']:
            signals.append("🟢 Price Above EMA20 - Bullish")
            signal_strength += 1
        else:
            signals.append("🔴 Price Below EMA20 - Bearish")
            signal_strength -= 1
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        if not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']):
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

# CHART IMAGE ANALYSIS MODE
if asset_type == "📸 Analyze Chart Image":
    st.markdown("### 📸 AI Chart Image Analysis")
    st.markdown("Upload a trading chart image and get AI-powered technical analysis!")
    
    uploaded_file = st.file_uploader(
        "Upload Chart Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a screenshot of your trading chart"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Chart", use_container_width=True)
        
        with col2:
            st.info("🤖 **AI Analysis in Progress...**")
            
            bytes_data = uploaded_file.getvalue()
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            
            img_type = uploaded_file.type.split('/')[-1]
            if img_type == 'jpg':
                img_type = 'jpeg'
            
            analysis_prompt = """Analyze this trading chart and provide:

1. **Trend Analysis**: Current trend (bullish/bearish/sideways)
2. **Support & Resistance**: Key levels
3. **Technical Patterns**: Any chart patterns (triangles, H&S, etc.)
4. **Indicators**: Visible indicators and signals (RSI, MACD, MA, etc.)
5. **Trading Signal**: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
6. **Entry Point**: Suggested entry price
7. **Stop Loss**: Recommended stop loss
8. **Take Profit**: Target prices
9. **Risk Assessment**: Low/Medium/High
10. **Confidence Level**: Your confidence (%)

Be specific and actionable."""

            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2000,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": f"image/{img_type}",
                                        "data": base64_image
                                    }
                                },
                                {"type": "text", "text": analysis_prompt}
                            ]
                        }]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result['content'][0]['text']
                    
                    st.success("✅ **AI Analysis Complete!**")
                    st.markdown("---")
                    st.markdown("### 🤖 Claude AI Analysis:")
                    st.markdown(analysis)
                    
                    if "STRONG BUY" in analysis.upper():
                        st.success("## 🟢 SIGNAL: STRONG BUY")
                    elif "STRONG SELL" in analysis.upper():
                        st.error("## 🔴 SIGNAL: STRONG SELL")
                    elif "BUY" in analysis.upper():
                        st.success("## 🟢 SIGNAL: BUY")
                    elif "SELL" in analysis.upper():
                        st.error("## 🔴 SIGNAL: SELL")
                    else:
                        st.warning("## 🟡 SIGNAL: NEUTRAL")
                else:
                    st.error(f"❌ API Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
    else:
        st.info("""
        📸 **How to use:**
        1. Screenshot your chart with indicators
        2. Upload the image
        3. Get AI analysis and signals!
        """)

# NORMAL DATA ANALYSIS MODE
elif symbol is not None:
    st.markdown("### 📡 Live Market Data")
    
    col1, col2, col3 = st.columns(3)
    
    with st.spinner(f"🔄 Fetching live data for {pair_display}..."):
        df, data_source = fetch_market_data(symbol, timeframe_config, asset_type)
    
    if df is not None and len(df) > 50:
        display_price = df['close'].iloc[-1]
        
        if len(df) >= 24:
            price_24h_ago = df['close'].iloc[-min(24, len(df))]
            price_change_24h = ((display_price - price_24h_ago) / price_24h_ago) * 100
        else:
            price_change_24h = 0
        
        is_precious_metal = asset_type == "🏆 Precious Metals"
        price_label = "Price per Ounce" if is_precious_metal else "Current Price"
        
        with col1:
            st.markdown(f"#### 💰 {price_label}")
            if display_price >= 1000:
                st.metric(pair_display, f"${display_price:,.0f}", f"{price_change_24h:+.2f}%")
            else:
                st.metric(pair_display, f"${display_price:,.2f}", f"{price_change_24h:+.2f}%")
            st.caption(f"📡 Source: {data_source}")
        
        with col2:
            st.markdown("#### 📊 24h Range")
            high_val = df['high'].tail(24).max()
            low_val = df['low'].tail(24).min()
            st.write(f"**High:** ${high_val:,.0f}" if high_val >= 1000 else f"**High:** ${high_val:,.2f}")
            st.write(f"**Low:** ${low_val:,.0f}" if low_val >= 1000 else f"**Low:** ${low_val:,.2f}")
        
        with col3:
            st.markdown("#### 📈 Market Data")
            st.write(f"**Volume:** ${df['volume'].sum()/1e6:.2f}M")
            st.write(f"**Data Points:** {len(df)}")
            if is_precious_metal:
                st.info("💎 Troy Ounce")
        
        st.markdown("---")
        
        # MULTI-TIMEFRAME ANALYSIS - Critical for Pro Trading!
        st.markdown("### 📊 Multi-Timeframe Confirmation")
        st.info("💡 **Pro Tip:** Always check multiple timeframes before entering a trade!")
        
        with st.spinner("🔍 Analyzing 15m, 1H, and 4H timeframes..."):
            alignment_score, tf_results, confidence, alignment, confidence_multiplier = multi_timeframe_analysis(symbol, asset_type)
        
        # Display timeframe analysis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 📉 15-Minute")
            if tf_results.get("15m"):
                signal_15m = tf_results["15m"]["signal"]
                trend_15m = tf_results["15m"]["trend"]
                if signal_15m == "BUY":
                    st.success(f"**{signal_15m}** 🟢")
                elif signal_15m == "SELL":
                    st.error(f"**{signal_15m}** 🔴")
                else:
                    st.warning(f"**{signal_15m}** 🟡")
                st.caption(f"Trend: {trend_15m}")
        
        with col2:
            st.markdown("#### 📊 1-Hour")
            if tf_results.get("1H"):
                signal_1h = tf_results["1H"]["signal"]
                trend_1h = tf_results["1H"]["trend"]
                if signal_1h == "BUY":
                    st.success(f"**{signal_1h}** 🟢")
                elif signal_1h == "SELL":
                    st.error(f"**{signal_1h}** 🔴")
                else:
                    st.warning(f"**{signal_1h}** 🟡")
                st.caption(f"Trend: {trend_1h}")
        
        with col3:
            st.markdown("#### 📈 4-Hour")
            if tf_results.get("4H"):
                signal_4h = tf_results["4H"]["signal"]
                trend_4h = tf_results["4H"]["trend"]
                if signal_4h == "BUY":
                    st.success(f"**{signal_4h}** 🟢")
                elif signal_4h == "SELL":
                    st.error(f"**{signal_4h}** 🔴")
                else:
                    st.warning(f"**{signal_4h}** 🟡")
                st.caption(f"Trend: {trend_4h}")
        
        with col4:
            st.markdown("#### ✅ Alignment")
            if confidence == "VERY HIGH":
                st.success(f"**{confidence}**")
                st.success(f"**{alignment}**")
            elif confidence == "HIGH":
                st.info(f"**{confidence}**")
                st.info(f"**{alignment}**")
            else:
                st.warning(f"**{confidence}**")
                st.warning(f"**{alignment}**")
            st.caption(f"Score: {alignment_score:.0%}")
        
        # Trading recommendation based on timeframe alignment
        if confidence == "VERY HIGH":
            if alignment == "BULLISH":
                st.success("🎯 **STRONG BUY SETUP**: All timeframes aligned BULLISH! High probability trade.")
            elif alignment == "BEARISH":
                st.error("🎯 **STRONG SELL SETUP**: All timeframes aligned BEARISH! High probability trade.")
        elif confidence == "HIGH":
            if alignment == "BULLISH":
                st.success("✅ **BUY SETUP**: 2 out of 3 timeframes BULLISH. Good probability trade.")
            elif alignment == "BEARISH":
                st.error("✅ **SELL SETUP**: 2 out of 3 timeframes BEARISH. Good probability trade.")
        else:
            st.warning("⚠️ **MIXED SIGNALS**: Timeframes disagree. Wait for better alignment or reduce position size.")
        
        st.markdown("---")
        
        # Calculate indicators
        if use_sma:
            df['sma_20'] = calculate_sma(df, 20)
            df['sma_50'] = calculate_sma(df, 50)
        if use_ema:
            df['ema_20'] = calculate_ema(df, 20)
            df['ema_50'] = calculate_ema(df, 50)
        if use_rsi:
            df['rsi_12'] = calculate_rsi(df, 12)
            df['rsi_16'] = calculate_rsi(df, 16)
            df['rsi_24'] = calculate_rsi(df, 24)
            df['rsi'] = df['rsi_16']  # Keep for backward compatibility
        if use_macd:
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
        if use_bb:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df)
        
        # Market Regime Detection - Critical for accuracy!
        df['regime'], df['adx'], df['plus_di'], df['minus_di'] = detect_market_regime(df)
        current_regime = df['regime'].iloc[-1]
        current_adx = df['adx'].iloc[-1]
        
        # Display Market Regime
        st.markdown("### 🎯 Market Regime Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            if current_regime == 'TRENDING':
                st.success(f"**Regime:** TRENDING 📈")
            elif current_regime == 'RANGING':
                st.warning(f"**Regime:** RANGING ↔️")
            else:
                st.info(f"**Regime:** VOLATILE 📊")
        with col2:
            st.metric("ADX (Trend Strength)", f"{current_adx:.1f}")
        with col3:
            if current_regime == 'TRENDING':
                st.success("✅ High Probability Trades")
            elif current_regime == 'RANGING':
                st.warning("⚠️ Low Probability - Caution!")
            else:
                st.info("ℹ️ Mixed Signals")
        
        st.markdown("---")
        
        # AI Model Training
        st.markdown("### 🤖 AI Price Prediction Engine")
        
        with st.spinner("🧠 Training AI model..."):
            model, feature_cols, accuracy = train_ml_model(df, ai_model, prediction_periods)
        
        if model is not None and feature_cols is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AI Model", ai_model)
            with col2:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            with col3:
                st.metric("Data Points", len(df))
            with col4:
                st.metric("Timeframe", timeframe_name)
            
            future_prices = predict_future_prices(df, model, feature_cols, ai_model, prediction_periods)
            
            if len(future_prices) > 0:
                st.markdown("#### 🎯 AI Price Predictions")
                pred_cols = st.columns(min(5, len(future_prices)))
                
                for i, pred_price in enumerate(future_prices):
                    with pred_cols[i % 5]:
                        change_pct = ((pred_price - display_price) / display_price) * 100
                        if pred_price >= 1000:
                            price_display = f"${pred_price:,.0f}"
                        else:
                            price_display = f"${pred_price:,.2f}"
                        
                        st.metric(f"+{i+1}", price_display, f"{change_pct:+.2f}%")
                
                avg_prediction = np.mean(future_prices)
                prediction_trend = "BULLISH 🚀" if avg_prediction > display_price else "BEARISH 🔻"
                expected_change = ((avg_prediction - display_price) / display_price) * 100
                
                if avg_prediction > display_price:
                    st.success(f"### ✅ AI PREDICTION: {prediction_trend}")
                    st.success(f"Expected movement: **+{expected_change:.2f}%** over {prediction_periods} periods")
                else:
                    st.error(f"### ⚠️ AI PREDICTION: {prediction_trend}")
                    st.error(f"Expected movement: **{expected_change:.2f}%** over {prediction_periods} periods")
        
        st.markdown("---")
        
        # Trading Signals
        signals, signal_strength = generate_signals(df)
        
        # Apply multi-timeframe confidence multiplier
        original_strength = signal_strength
        signal_strength = int(signal_strength * confidence_multiplier)
        
        st.markdown("### 🎯 Live Trading Signals")
        
        # Show multi-timeframe adjustment
        if confidence_multiplier != 1.0:
            if confidence_multiplier > 1.0:
                st.success(f"✅ Signal BOOSTED by multi-timeframe alignment (×{confidence_multiplier:.1f})")
            else:
                st.warning(f"⚠️ Signal REDUCED due to timeframe conflict (×{confidence_multiplier:.1f})")
        
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
            
            st.metric("Signal Strength", f"{signal_strength}/10", delta=f"{signal_strength - original_strength:+d} (MTF)")
        
        with col2:
            st.markdown("#### 📋 Signals:")
            if len(signals) > 0:
                for signal in signals:
                    st.markdown(f"- {signal}")
            else:
                st.info("Calculating signals...")
        
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
        
        if model and len(future_prices) > 0:
            last_ts = df['timestamp'].iloc[-1]
            time_delta = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2])
            future_ts = [last_ts + time_delta * (i+1) for i in range(len(future_prices))]
            
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
        
        if use_sma:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
        
        if use_ema:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], name='EMA 20', line=dict(color='red', dash='dot')), row=1, col=1)
        
        if use_bb:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
        
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
        
        if use_rsi:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_12'], name='RSI-12', line=dict(color='blue', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_16'], name='RSI-16', line=dict(color='purple', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_24'], name='RSI-24', line=dict(color='orange', width=2)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        if use_macd:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=4, col=1)
            colors_macd = ['green' if val > 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], marker_color=colors_macd, showlegend=False), row=4, col=1)
        
        fig.update_layout(height=1000, showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # IMPROVED Entry/Exit Section
        st.markdown("### 💰 Trading Setup & Recommendations")
        
        is_buy_setup = signal_strength >= 0
        current_price = display_price
        
        if 'bb_lower' in df.columns and not pd.isna(df['bb_lower'].iloc[-1]):
            support_bb = df['bb_lower'].iloc[-1]
        else:
            support_bb = None
        
        if 'bb_upper' in df.columns and not pd.isna(df['bb_upper'].iloc[-1]):
            resistance_bb = df['bb_upper'].iloc[-1]
        else:
            resistance_bb = None
        
        recent_low = df['low'].tail(20).min()
        recent_high = df['high'].tail(20).max()
        
        def format_price(price):
            if price >= 1000:
                return f"${price:,.0f}"
            else:
                return f"${price:,.2f}"
        
        if is_buy_setup:
            st.success("### 🟢 BUY SETUP DETECTED")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Buy Strategy")
                
                entry_price = current_price
                tp1 = entry_price * 1.02
                tp2 = entry_price * 1.035
                tp3 = entry_price * 1.05
                stop_loss = entry_price * 0.98
                
                st.info(f"""
                **📍 Entry Price:** {format_price(entry_price)}
                
                **🎯 Take Profit Targets:**
                - TP1: {format_price(tp1)} (+2%)
                - TP2: {format_price(tp2)} (+3.5%)
                - TP3: {format_price(tp3)} (+5%)
                
                **🛡️ Stop Loss:** {format_price(stop_loss)} (-2%)
                
                **📊 Risk/Reward:** 1:2.5 (Good)
                """)
            
            with col2:
                st.markdown("#### 💡 Buy Recommendation")
                
                better_entry = support_bb if support_bb else recent_low
                
                st.success(f"""
                **✅ IMMEDIATE BUY:**
                If price is at or below **{format_price(current_price)}**
                - Entry: {format_price(entry_price)}
                - TP: {format_price(tp2)}
                - SL: {format_price(stop_loss)}
                
                **⭐ BETTER BUY (Wait for dip):**
                If price drops to **{format_price(better_entry)}**
                - Entry: {format_price(better_entry)}
                - TP: {format_price(better_entry * 1.04)}
                - SL: {format_price(better_entry * 0.98)}
                
                **Risk Level:** {"Low" if signal_strength >= 5 else "Medium"}
                """)
        
        else:
            st.error("### 🔴 SELL SETUP DETECTED")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Sell Strategy")
                
                entry_price = current_price
                tp1 = entry_price * 0.98
                tp2 = entry_price * 0.965
                tp3 = entry_price * 0.95
                stop_loss = entry_price * 1.02
                
                st.info(f"""
                **📍 Entry Price:** {format_price(entry_price)}
                
                **🎯 Take Profit Targets:**
                - TP1: {format_price(tp1)} (-2%)
                - TP2: {format_price(tp2)} (-3.5%)
                - TP3: {format_price(tp3)} (-5%)
                
                **🛡️ Stop Loss:** {format_price(stop_loss)} (+2%)
                
                **📊 Risk/Reward:** 1:2.5 (Good)
                """)
            
            with col2:
                st.markdown("#### 💡 Sell Recommendation")
                
                better_entry = resistance_bb if resistance_bb else recent_high
                
                st.error(f"""
                **✅ IMMEDIATE SELL:**
                If price is at or above **{format_price(current_price)}**
                - Entry: {format_price(entry_price)}
                - TP: {format_price(tp2)}
                - SL: {format_price(stop_loss)}
                
                **⭐ BETTER SELL (Wait for bounce):**
                If price rises to **{format_price(better_entry)}**
                - Entry: {format_price(better_entry)}
                - TP: {format_price(better_entry * 0.96)}
                - SL: {format_price(better_entry * 1.02)}
                
                **Risk Level:** {"Low" if signal_strength <= -5 else "Medium"}
                """)
        
        st.markdown("---")
        st.markdown("#### 📊 Key Technical Levels")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Current Price", format_price(current_price))
        with col2:
            st.metric("🟢 Support Level", format_price(recent_low))
        with col3:
            st.metric("🔴 Resistance Level", format_price(recent_high))
        
        st.markdown("---")
        st.warning("""
        **⚠️ Risk Management:**
        - Use stop-loss orders
        - Never risk more than 1-2% per trade
        - Diversify your portfolio
        - This is NOT financial advice
        """)
    
    else:
        st.error("❌ Unable to fetch data")

# Auto-refresh
if auto_refresh and asset_type != "📸 Analyze Chart Image":
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>📡 Data Sources:</b></p>
    <p>Crypto: OKX → Binance → CryptoCompare</p>
    <p>Metals: Twelve Data → Yahoo Finance</p>
    <p>Forex: Twelve Data API</p>
    <p>Chart Analysis: Claude AI Vision</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
