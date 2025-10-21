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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Live Market AI Analysis", layout="wide", page_icon="🤖")

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

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=5)
timeframe_config = TIMEFRAMES[timeframe_name]

# Show timeframe only if not analyzing chart image
if asset_type != "📸 Analyze Chart Image":
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)
else:
    auto_refresh = False

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

# 1. OKX API (Primary for Crypto)
@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=100):
    """Fetch data from OKX API"""
    url = "https://www.okx.com/api/v5/market/candles"
    
    # Ensure limit doesn't exceed OKX max (300)
    limit = min(limit, 300)
    
    params = {
        "instId": f"{symbol}-USDT",
        "bar": interval,
        "limit": str(limit)
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != '0':
            error_msg = data.get('msg', 'Unknown error')
            st.warning(f"⚠️ OKX API error: {error_msg}")
            st.write(f"Debug - Tried: {params}")
            return None, None
        
        # OKX returns: [timestamp, open, high, low, close, volume, ...]
        candles = data.get('data', [])
        
        if not candles or len(candles) == 0:
            st.warning(f"⚠️ OKX returned no data for {symbol}")
            return None, None
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        
        # Convert timestamp from milliseconds
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # OKX returns newest first, so reverse it
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        st.success(f"✅ OKX: Loaded {len(df)} data points")
        return df, "OKX"
        
    except requests.exceptions.Timeout:
        st.warning("⚠️ OKX API timeout")
        return None, None
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ OKX API network error: {str(e)[:100]}")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ OKX API failed: {str(e)[:150]}")
        return None, None

# 2. BINANCE API (Backup for Crypto)
@st.cache_data(ttl=300)
def get_binance_data(symbol, interval="1h", limit=100):
    """Fetch data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    
    # Binance max limit is 1000
    limit = min(limit, 1000)
    
    params = {
        "symbol": f"{symbol}USDT",
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Check if response is an error
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
        
    except requests.exceptions.Timeout:
        st.warning("⚠️ Binance API timeout")
        return None, None
    except requests.exceptions.HTTPError as e:
        st.warning(f"⚠️ Binance HTTP error: {e.response.status_code}")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Binance API failed: {str(e)[:150]}")
        return None, None

# 2b. CRYPTOCOMPARE API (Backup for Crypto)
@st.cache_data(ttl=300)
def get_cryptocompare_data(symbol, limit=100, unit="hour"):
    """Fetch data from CryptoCompare"""
    
    # CryptoCompare has a max limit of 2000
    limit = min(limit, 2000)
    
    if unit == "minute":
        endpoint = "histominute"
    elif unit == "hour":
        endpoint = "histohour"
    else:
        endpoint = "histoday"
    
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit
    }
    
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
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume'
        })
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna()
        
        st.success(f"✅ CryptoCompare: Loaded {len(df)} data points")
        return df, "CryptoCompare"
        
    except requests.exceptions.Timeout:
        st.warning("⚠️ CryptoCompare timeout (slow response)")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ CryptoCompare failed: {str(e)[:150]}")
        return None, None

# 3. YAHOO FINANCE (for Gold, Silver - NO API KEY NEEDED!)
@st.cache_data(ttl=300)
def get_yahoo_finance_commodities(symbol, period="1d", interval="1h"):
    """Fetch commodities from Yahoo Finance - NO AUTH NEEDED"""
    
    # Map symbols to Yahoo Finance tickers
    yahoo_symbols = {
        "XAU/USD": "GC=F",  # Gold Futures
        "XAG/USD": "SI=F",  # Silver Futures
        "XPT/USD": "PL=F",  # Platinum Futures
        "XPD/USD": "PA=F"   # Palladium Futures
    }
    
    yahoo_ticker = yahoo_symbols.get(symbol, "GC=F")
    
    # Period mapping
    period_map = {
        "minute": "1d", "hour": "5d", "day": "60d"
    }
    
    # Interval mapping
    interval_map = {
        "minute": "1m", "hour": "1h", "day": "1d"
    }
    
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, None
        
        # Rename columns to match our format
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Make sure timestamp is datetime
        if 'timestamp' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            elif 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
        
        # Ensure volume exists (commodities might not have it)
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna(subset=['open', 'high', 'low', 'close'])  # Only drop if price data is missing
        
        st.success(f"✅ Yahoo Finance: Loaded {len(df)} data points")
        return df, "Yahoo Finance"
        
    except ImportError:
        st.error("❌ yfinance library not installed. Add 'yfinance' to requirements.txt")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {str(e)[:100]}")
        return None, None

# 4. TWELVE DATA API (Backup for Gold, Silver, Commodities)
@st.cache_data(ttl=300)
def get_twelve_data_commodities(symbol, interval="1h", outputsize=100):
    """Fetch commodities data from Twelve Data API"""
    
    # Check if API key exists
    try:
        api_key = st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Twelve Data API key not found. Please add it to Streamlit Secrets.")
        st.info("📝 Go to Settings → Secrets and add: TWELVE_DATA_API_KEY = 'your_key_here'")
        return None, None
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON"
    }
    
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
        
        # Handle volume - might not exist for commodities
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
        else:
            # Create placeholder volume for commodities (they don't have volume)
            df['volume'] = 0.0
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        st.success(f"✅ Twelve Data: Loaded {len(df)} data points")
        return df, "Twelve Data"
        
    except Exception as e:
        st.error(f"❌ Twelve Data API Error: {str(e)}")
        return None, None

# Main data fetching function
def fetch_market_data(symbol, timeframe_config, asset_type):
    """Fetch data from appropriate source based on asset type"""
    
    if asset_type == "💰 Cryptocurrency":
        # Try OKX first
        st.info("🔄 Trying OKX API...")
        df, source = get_okx_data(symbol, timeframe_config["okx"], timeframe_config["limit"])
        
        # Fallback to Binance if OKX fails
        if df is None:
            st.info("🔄 OKX failed. Trying Binance API...")
            df, source = get_binance_data(symbol, timeframe_config["binance"], timeframe_config["limit"])
        
        # Fallback to CryptoCompare if both fail
        if df is None:
            st.info("🔄 Binance failed. Trying CryptoCompare...")
            # Reduce limit for CryptoCompare to avoid issues
            cc_limit = min(timeframe_config["limit"], 2000)
            df, source = get_cryptocompare_data(symbol, cc_limit, timeframe_config["unit"])
        
        return df, source
    
    elif asset_type == "💱 Forex" or asset_type == "🔍 Custom Search":
        # Try Twelve Data for Forex and custom symbols
        st.info("🔄 Fetching from Twelve Data API...")
        
        # Check if API key exists
        api_key = None
        try:
            api_key = st.secrets.get("TWELVE_DATA_API_KEY")
        except:
            pass
        
        if api_key:
            interval_map = {
                "minute": "1min",
                "hour": "1h",
                "day": "1day"
            }
            interval = interval_map.get(timeframe_config["unit"], "1h")
            df, source = get_twelve_data_commodities(symbol, interval, timeframe_config["limit"])
            
            if df is not None:
                return df, source
            else:
                st.warning("⚠️ Twelve Data couldn't fetch this symbol")
        else:
            st.error("❌ Twelve Data API key required for Forex and custom symbols")
            st.info("📝 Add your API key to Settings → Secrets: TWELVE_DATA_API_KEY = 'your_key'")
        
        # Try CryptoCompare as fallback for custom crypto symbols
        if asset_type == "🔍 Custom Search":
            st.info("🔄 Trying CryptoCompare for custom symbol...")
            cc_limit = min(timeframe_config["limit"], 2000)
            df, source = get_cryptocompare_data(symbol, cc_limit, timeframe_config["unit"])
            if df is not None:
                return df, source
        
        return None, None
    
    else:  # Precious Metals
        # Try Twelve Data first (Better quality with API key!)
        st.info("🔄 Fetching from Twelve Data API (Primary)...")
        
        # Check if API key exists
        api_key = None
        try:
            api_key = st.secrets.get("TWELVE_DATA_API_KEY")
        except:
            pass
        
        if api_key:
            # Map interval
            interval_map = {
                "minute": "1min",
                "hour": "1h",
                "day": "1day"
            }
            interval = interval_map.get(timeframe_config["unit"], "1h")
            df, source = get_twelve_data_commodities(symbol, interval, timeframe_config["limit"])
            
            # If Twelve Data succeeds, return immediately
            if df is not None:
                return df, source
        else:
            st.warning("⚠️ Twelve Data API key not found")
        
        # Fallback to Yahoo Finance if Twelve Data fails or no API key
        st.info("🔄 Twelve Data failed. Trying Yahoo Finance (Backup)...")
        
        # Map period and interval for Yahoo Finance
        if timeframe_config["unit"] == "minute":
            period, interval = "1d", "1m"
        elif timeframe_config["unit"] == "hour":
            period, interval = "5d", "1h"
        else:
            period, interval = "60d", "1d"
        
        df, source = get_yahoo_finance_commodities(symbol, period, interval)
        
        return df, source

@st.cache_data(ttl=60)
def get_current_price_crypto(symbol):
    """Get current crypto price from multiple sources"""
    # Try OKX first
    try:
        url = "https://www.okx.com/api/v5/market/ticker"
        response = requests.get(url, params={"instId": f"{symbol}-USDT"}, timeout=5)
        data = response.json()
        if data.get('code') == '0' and data.get('data'):
            return float(data['data'][0]['last']), "OKX"
    except:
        pass
    
    # Try Binance
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, params={"symbol": f"{symbol}USDT"}, timeout=5)
        data = response.json()
        return float(data['price']), "Binance"
    except:
        pass
    
    # Fallback to CryptoCompare
    try:
        url = "https://min-api.cryptocompare.com/data/price"
        response = requests.get(url, params={"fsym": symbol, "tsyms": "USD"}, timeout=5)
        data = response.json()
        return data.get('USD', None), "CryptoCompare"
    except:
        return None, None

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
        score = max(0, 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values))
        return model, X.columns.tolist(), score
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = max(0, 1 - np.mean(np.abs(predictions - y_test.values) / y_test.values))
        return model, X.columns.tolist(), score
    else:  # Ensemble
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
    """Generate predictions"""
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
    """Generate trading signals"""
    signals = []
    signal_strength = 0
    
    if len(df) < 2:
        return signals, signal_strength
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    if 'rsi' in df.columns and not pd.isna(latest['rsi']):
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
        if not pd.isna(latest['macd']) and not pd.isna(prev['macd']):
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
        if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
            if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                signals.append("🟢 Golden Cross - Strong BUY")
                signal_strength += 3
            elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                signals.append("🔴 Death Cross - Strong SELL")
                signal_strength -= 3
    
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
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Chart", use_container_width=True)
        
        with col2:
            st.info("🤖 **AI Analysis in Progress...**")
            
            # Convert image to base64
            import base64
            from io import BytesIO
            
            bytes_data = uploaded_file.getvalue()
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            
            # Determine image type
            img_type = uploaded_file.type.split('/')[-1]
            if img_type == 'jpg':
                img_type = 'jpeg'
            
            # Prepare the prompt for Claude API
            analysis_prompt = """Analyze this trading chart image and provide:

1. **Trend Analysis**: Current trend (bullish/bearish/sideways)
2. **Support & Resistance**: Key levels
3. **Technical Patterns**: Any chart patterns visible (head & shoulders, triangles, etc.)
4. **Indicators**: Visible indicators and their signals (RSI, MACD, Moving Averages, etc.)
5. **Trading Signal**: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
6. **Entry Point**: Suggested entry price (if visible)
7. **Stop Loss**: Recommended stop loss level
8. **Take Profit**: Target price levels
9. **Risk Assessment**: Low/Medium/High risk
10. **Confidence Level**: Your confidence in this analysis (%)

Be specific, professional, and actionable. Focus on what you can clearly see in the chart."""

            try:
                # Call Claude API for image analysis
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
                                {
                                    "type": "text",
                                    "text": analysis_prompt
                                }
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
                    st.markdown("### 🤖 Claude AI Analysis Results:")
                    st.markdown(analysis)
                    
                    # Extract signal if possible
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
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                st.info("💡 Make sure the image is clear and shows technical indicators.")
    
    else:
        st.info("""
        📸 **How to use Chart Image Analysis:**
        
        1. Take a screenshot of your trading chart
        2. Include visible indicators (RSI, MACD, Moving Averages)
        3. Upload the image above
        4. Get AI-powered analysis and trading signals!
        
        **Best Results:**
        - Clear, high-resolution images
        - Include timeframe information
        - Show multiple indicators
        - Include price levels and volume
        """)
    
    st.markdown("---")
    st.warning("""
    **⚠️ Important Notes:**
    - AI analysis is based on visual pattern recognition
    - Not financial advice - always do your own research
    - Best used as a second opinion on your analysis
    - Accuracy depends on image quality and visible information
    """)

# NORMAL DATA ANALYSIS MODE
elif symbol is not None:
    # Live Data
    st.markdown("### 📡 Live Market Data")
    
    col1, col2, col3 = st.columns(3)

# Fetch live data
with st.spinner(f"🔄 Fetching live data for {pair_display}..."):
    df, data_source = fetch_market_data(symbol, timeframe_config, asset_type)

if df is not None and len(df) > 50:
    # Get current price
    display_price = df['close'].iloc[-1]
    
    if asset_type == "💰 Cryptocurrency":
        current_price, price_source = get_current_price_crypto(symbol)
        if current_price:
            display_price = current_price
    
    # Calculate 24h change
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
            st.metric(
                pair_display,
                f"${display_price:,.0f}",
                f"{price_change_24h:+.2f}%"
            )
        else:
            st.metric(
                pair_display,
                f"${display_price:,.2f}",
                f"{price_change_24h:+.2f}%"
            )
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
        
        # Generate predictions
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
                    
                    st.metric(
                        f"+{i+1}",
                        price_display,
                        f"{change_pct:+.2f}%"
                    )
            
            # AI Recommendation
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
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
    
    # RSI
    if use_rsi:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple')), row=3, col=1)
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
    
    # Entry/Exit
    st.markdown("### 💰 Entry & Exit Points")
    
    col1, col2 = st.columns(2)
    
    def format_price(price):
        if price >= 1000:
            return f"${price:,.0f}"
        else:
            return f"${price:,.2f}"
    
    with col1:
        st.success("#### 🟢 BUY ZONES")
        if 'bb_lower' in df.columns and not pd.isna(df['bb_lower'].iloc[-1]):
            st.write(f"Lower BB: **{format_price(df['bb_lower'].iloc[-1])}**")
        st.write(f"Recent Low: **{format_price(df['low'].tail(20).min())}**")
    
    with col2:
        st.error("#### 🔴 SELL ZONES")
        if 'bb_upper' in df.columns and not pd.isna(df['bb_upper'].iloc[-1]):
            st.write(f"Upper BB: **{format_price(df['bb_upper'].iloc[-1])}**")
        st.write(f"Recent High: **{format_price(df['high'].tail(20).max())}**")
    
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
    st.error("❌ Unable to fetch data. All data sources failed.")
    
    if asset_type == "🏆 Precious Metals":
        st.info("""
        💡 **For Gold/Silver Analysis:**
        
        **Primary:** Twelve Data API (Better Quality!) ✅
        - Real-time spot prices
        - You have this configured! ✅
        
        **Backup:** Yahoo Finance (FREE - No API Key)
        - Commodity futures data
        - Falls back automatically if Twelve Data fails
        
        **Both sources failed** - This is unusual. Try:
        - Refresh the page
        - Try a different timeframe
        - Check back in a few minutes
        """)
    elif asset_type == "💱 Forex":
        st.info("""
        💡 **For Forex Analysis:**
        
        **Requires:** Twelve Data API key
        - Get FREE key: https://twelvedata.com/
        - Add to Settings → Secrets
        - Format: `TWELVE_DATA_API_KEY = "your_key"`
        
        **Supported Pairs:**
        - Major: EUR/USD, GBP/USD, USD/JPY, etc.
        - Crosses: EUR/GBP, GBP/JPY, etc.
        - All standard forex pairs
        """)
    elif asset_type == "🔍 Custom Search":
        st.info("""
        💡 **Custom Symbol Search:**
        
        **Searches:**
        1. Twelve Data API (Forex, Stocks, Crypto, Commodities)
        2. CryptoCompare (Cryptocurrencies)
        
        **Tips:**
        - Use standard symbols (BTC, AAPL, EUR/USD)
        - Check symbol format for each exchange
        - Try different variations if it fails
        """)
    else:
        st.info("""
        💡 **Crypto Data Sources:**
        - The app tries: OKX → Binance → CryptoCompare
        - All three sources failed
        - This might be a temporary API issue
        - Try a shorter timeframe or refresh in a few minutes
        """)
    
    # Show debug info
    st.markdown("### 🔍 Debug Information")
    st.write(f"**Asset Type:** {asset_type}")
    st.write(f"**Symbol:** {symbol}")
    st.write(f"**Display Name:** {pair_display}")
    st.write(f"**Timeframe:** {timeframe_name}")
    st.write(f"**Config:** {timeframe_config}")
    
    if asset_type == "💰 Cryptocurrency":
        st.write(f"**OKX Params:** instId={symbol}-USDT, bar={timeframe_config['okx']}, limit={timeframe_config['limit']}")
        st.write(f"**Binance Params:** symbol={symbol}USDT, interval={timeframe_config['binance']}, limit={timeframe_config['limit']}")
    
    st.info("💡 **Tip:** Try selecting '1 Hour' or '6 Hours' timeframe - shorter timeframes usually work better.")

# Auto-refresh (only for normal data mode)
if auto_refresh and asset_type != "📸 Analyze Chart Image":
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>📡 Data Sources:</b></p>
    <p>Crypto: OKX API (Primary) → Binance (Backup) → CryptoCompare (Fallback)</p>
    <p>Precious Metals: Twelve Data API (Primary) → Yahoo Finance (Free Backup)</p>
    <p>Forex: Twelve Data API</p>
    <p>Chart Analysis: Claude AI (Vision)</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
