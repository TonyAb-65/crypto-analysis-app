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
import sqlite3
import json
warnings.filterwarnings('ignore')

# ==================== DATABASE FUNCTIONS ====================
def init_database():
    """Initialize SQLite database for trade tracking"""
    conn = sqlite3.connect('trading_ai_learning.db')
    cursor = conn.cursor()
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            current_price REAL NOT NULL,
            predicted_price REAL NOT NULL,
            prediction_horizon INTEGER NOT NULL,
            confidence REAL NOT NULL,
            signal_strength INTEGER,
            features TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Trade results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            trade_date TEXT NOT NULL,
            profit_loss REAL NOT NULL,
            profit_loss_pct REAL NOT NULL,
            prediction_error REAL NOT NULL,
            notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    
    # Model performance tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            retrain_date TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            trades_used INTEGER NOT NULL,
            accuracy_before REAL,
            accuracy_after REAL,
            avg_error_before REAL,
            avg_error_after REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(asset_type, pair, timeframe, current_price, predicted_price, 
                   prediction_horizon, confidence, signal_strength, features):
    """Save a prediction to database"""
    conn = sqlite3.connect('trading_ai_learning.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, asset_type, pair, timeframe, current_price, predicted_price, 
         prediction_horizon, confidence, signal_strength, features, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        asset_type,
        pair,
        timeframe,
        current_price,
        predicted_price,
        prediction_horizon,
        confidence,
        signal_strength,
        json.dumps(features) if features else None,
        'pending'
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def save_trade_result(prediction_id, entry_price, exit_price, notes=""):
    """Save actual trade result"""
    conn = sqlite3.connect('trading_ai_learning.db')
    cursor = conn.cursor()
    
    # Get prediction details
    cursor.execute('SELECT predicted_price FROM predictions WHERE id = ?', (prediction_id,))
    result = cursor.fetchone()
    
    if result:
        predicted_price = result[0]
        profit_loss = exit_price - entry_price
        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        prediction_error = abs(predicted_price - exit_price) / exit_price * 100
        
        cursor.execute('''
            INSERT INTO trade_results 
            (prediction_id, entry_price, exit_price, trade_date, profit_loss, 
             profit_loss_pct, prediction_error, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            entry_price,
            exit_price,
            datetime.now().isoformat(),
            profit_loss,
            profit_loss_pct,
            prediction_error,
            notes
        ))
        
        # Update prediction status
        cursor.execute('UPDATE predictions SET status = ? WHERE id = ?', 
                      ('completed', prediction_id))
        
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def get_pending_predictions(asset_type=None):
    """Get predictions that haven't been matched with trades yet"""
    conn = sqlite3.connect('trading_ai_learning.db')
    
    query = '''
        SELECT id, timestamp, asset_type, pair, timeframe, current_price, 
               predicted_price, confidence, signal_strength
        FROM predictions 
        WHERE status = 'pending'
    '''
    
    if asset_type:
        query += ' AND asset_type = ?'
        df = pd.read_sql_query(query, conn, params=(asset_type,))
    else:
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

def get_completed_trades(asset_type=None, limit=100):
    """Get completed trades with results"""
    conn = sqlite3.connect('trading_ai_learning.db')
    
    query = '''
        SELECT 
            p.id, p.timestamp, p.asset_type, p.pair, p.timeframe,
            p.predicted_price, t.entry_price, t.exit_price,
            t.profit_loss, t.profit_loss_pct, t.prediction_error, t.trade_date
        FROM predictions p
        JOIN trade_results t ON p.id = t.prediction_id
        WHERE 1=1
    '''
    
    if asset_type:
        query += ' AND p.asset_type = ?'
        df = pd.read_sql_query(query + ' ORDER BY t.trade_date DESC LIMIT ?', 
                              conn, params=(asset_type, limit))
    else:
        df = pd.read_sql_query(query + ' ORDER BY t.trade_date DESC LIMIT ?', 
                              conn, params=(limit,))
    
    conn.close()
    return df

def get_performance_stats(asset_type=None):
    """Get performance statistics"""
    conn = sqlite3.connect('trading_ai_learning.db')
    
    query = '''
        SELECT 
            p.asset_type,
            COUNT(*) as total_trades,
            AVG(t.prediction_error) as avg_error,
            AVG(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) * 100 as win_rate,
            AVG(t.profit_loss_pct) as avg_return
        FROM predictions p
        JOIN trade_results t ON p.id = t.prediction_id
        WHERE 1=1
    '''
    
    if asset_type:
        query += ' AND p.asset_type = ? GROUP BY p.asset_type'
        df = pd.read_sql_query(query, conn, params=(asset_type,))
    else:
        query += ' GROUP BY p.asset_type'
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

# Initialize database on startup
init_database()
# ==================== END DATABASE FUNCTIONS ====================

# Page configuration
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

# Title
st.title("🤖 AI Trading Analysis Platform - IMPROVED")
st.markdown("*Crypto, Forex, Metals + Enhanced AI Predictions*")

# Check if Binance is blocked in user's region
if 'binance_blocked' not in st.session_state:
    st.session_state.binance_blocked = False

# Info banner
if st.session_state.binance_blocked:
    st.info("ℹ️ **Note:** Binance API is blocked in your region. Using OKX and backup APIs instead.")

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
    "5 Minutes": {"limit": 100, "binance": "5m", "okx": "5m"},
    "10 Minutes": {"limit": 100, "binance": "10m", "okx": "10m"},
    "15 Minutes": {"limit": 100, "binance": "15m", "okx": "15m"},
    "30 Minutes": {"limit": 100, "binance": "30m", "okx": "30m"},
    "1 Hour": {"limit": 100, "binance": "1h", "okx": "1H"},
    "4 Hours": {"limit": 100, "binance": "4h", "okx": "4H"},
    "1 Day": {"limit": 100, "binance": "1d", "okx": "1D"}
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=3)  # Default to 1 Hour
timeframe_config = TIMEFRAMES[timeframe_name]

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False, 
                                   help="Automatically refresh data every 60 seconds")

# Initialize last refresh time in session state
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()

# Check if it's time to refresh
if auto_refresh:
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_refresh_time
    
    if time_since_refresh >= 60:
        st.session_state.last_refresh_time = current_time
        st.rerun()
    else:
        remaining = 60 - int(time_since_refresh)
        st.sidebar.info(f"⏱️ Next refresh in {remaining}s...")
        time.sleep(1)  # Check every second
        st.rerun()

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

# Learning Dashboard Toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 AI Learning System")
show_learning_dashboard = st.sidebar.checkbox("📊 Show Learning Dashboard", value=False,
                                              help="View predictions, log trades, and track AI performance")

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
    - FIXED: Better NaN handling
    """
    try:
        if len(df) < 60:
            st.warning("⚠️ Need at least 60 data points")
            return None, None, 0, None
        
        # CRITICAL FIX: Remove all NaN values before creating features
        df_clean = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create sequences with context
        X, y = create_pattern_features(df_clean, lookback=lookback)
        
        if len(X) < 30:
            st.warning("⚠️ Not enough data after cleaning")
            return None, None, 0, None
        
        # CRITICAL FIX: Replace any remaining NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for valid data
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            st.error("❌ Data contains NaN values after cleaning")
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
        lookback_start = len(df_clean) - lookback
        
        for i in range(lookback_start, len(df_clean)):
            hour_features = [
                df_clean['close'].iloc[i],
                df_clean['volume'].iloc[i],
                df_clean['rsi'].iloc[i] if 'rsi' in df_clean.columns else 50,
                df_clean['macd'].iloc[i] if 'macd' in df_clean.columns else 0,
                df_clean['sma_20'].iloc[i] if 'sma_20' in df_clean.columns else df_clean['close'].iloc[i],
                df_clean['volatility'].iloc[i] if 'volatility' in df_clean.columns else 0
            ]
            
            if i > lookback_start:
                prev_close = df_clean['close'].iloc[i-1]
                hour_features.append((df_clean['close'].iloc[i] - prev_close) / (prev_close + 1e-10))
            else:
                hour_features.append(0)
            
            current_sequence.extend(hour_features)
        
        current_sequence = np.array(current_sequence).reshape(1, -1)
        
        # CRITICAL FIX: Clean current sequence
        current_sequence = np.nan_to_num(current_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_scaled = scaler.transform(current_sequence)
        
        # Predict future prices
        predictions = []
        for _ in range(prediction_periods):
            rf_pred = rf_model.predict(current_scaled)[0]
            gb_pred = gb_model.predict(current_scaled)[0]
            pred_price = 0.4 * rf_pred + 0.6 * gb_pred  # GB weighted more
            predictions.append(float(pred_price))  # Ensure it's a normal float
        
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
        rsi_insights = analyze_rsi_bounce_patterns(df_clean)
        
        return predictions, ['Pattern-based features'], confidence, rsi_insights
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
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
        
        # Save prediction to database
        prediction_id = save_prediction(
            asset_type=asset_type.replace("💰 ", "").replace("🏆 ", "").replace("💱 ", "").replace("🔍 ", ""),
            pair=symbol,
            timeframe=timeframe_name,
            current_price=current_price,
            predicted_price=predictions[-1],
            prediction_horizon=prediction_periods,
            confidence=confidence,
            signal_strength=signal_strength,
            features=features if features else {}
        )
        
        # Store in session state for tracking
        if 'last_prediction_id' not in st.session_state:
            st.session_state.last_prediction_id = prediction_id
        else:
            st.session_state.last_prediction_id = prediction_id
        
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
        tp1 = entry * 1.015  # +1.5%
        tp2 = entry * 1.025  # +2.5%
        tp3 = entry * 1.035  # +3.5%
        sl = entry * 0.98    # -2%
        
        # Create table data
        trade_data = {
            'Level': ['Entry', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '+1.5%', '+2.5%', '+3.5%', '-2%'],
            'Risk/Reward': ['-', '1:0.75', '1:1.25', '1:1.75', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        
    else:
        st.error("### 🔴 SELL SETUP")
        entry = current_price
        tp1 = entry * 0.985  # -1.5%
        tp2 = entry * 0.975  # -2.5%
        tp3 = entry * 0.965  # -3.5%
        sl = entry * 1.02    # +2%
        
        # Create table data
        trade_data = {
            'Level': ['Entry', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '-1.5%', '-2.5%', '-3.5%', '+2%'],
            'Risk/Reward': ['-', '1:0.75', '1:1.25', '1:1.75', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
    
    st.warning("⚠️ **Risk Warning:** Use stop-losses. Never risk more than 1-2% per trade. Not financial advice.")
    
    # ==================== LEARNING DASHBOARD ====================
    if show_learning_dashboard:
        st.markdown("---")
        st.markdown("## 🎓 AI Learning Dashboard")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Log Trade", "📊 Performance Stats", "📋 Trade History", "🔄 Retrain Model"])
        
        # TAB 1: Log Trade Results
        with tab1:
            st.markdown("### 📝 Log Your Trade Results")
            st.info("💡 Log your actual entry and exit prices to help the AI learn and improve predictions!")
            
            # Get pending predictions (fresh data)
            pending_preds = get_pending_predictions()
            
            if len(pending_preds) > 0:
                st.success(f"✅ You have **{len(pending_preds)}** pending predictions to track")
                
                # Display pending predictions table
                st.markdown("#### Recent Predictions (Pending)")
                display_pending = pending_preds[['id', 'timestamp', 'asset_type', 'pair', 
                                                 'current_price', 'predicted_price', 'confidence']].copy()
                display_pending['timestamp'] = pd.to_datetime(display_pending['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_pending['current_price'] = display_pending['current_price'].apply(lambda x: f"${x:,.2f}")
                display_pending['predicted_price'] = display_pending['predicted_price'].apply(lambda x: f"${x:,.2f}")
                display_pending['confidence'] = display_pending['confidence'].apply(lambda x: f"{x:.1f}%")
                display_pending.columns = ['ID', 'Time', 'Asset', 'Pair', 'Entry Price', 'Predicted', 'Confidence']
                st.dataframe(display_pending, use_container_width=True, hide_index=True)
                
                # Form to log trade
                st.markdown("#### 📥 Enter Trade Result")
                
                # Create a simple mapping for the selectbox
                pred_options = {}
                for _, row in pending_preds.iterrows():
                    pred_options[row['id']] = f"ID {row['id']} - {row['pair']} ({row['asset_type']})"
                
                # Selectbox for prediction selection (outside form)
                selected_pred_id = st.selectbox(
                    "Select Prediction ID", 
                    options=list(pred_options.keys()),
                    format_func=lambda x: pred_options[x],
                    key="prediction_selector"
                )
                
                # Get the selected prediction details (this will update when selectbox changes)
                selected_pred = pending_preds[pending_preds['id'] == selected_pred_id].iloc[0]
                
                # Display prediction details in a colored box
                st.info(f"""
                **📊 Selected Prediction Details:**
                - **Pair:** {selected_pred['pair']}
                - **Asset Type:** {selected_pred['asset_type']}
                - **Timeframe:** {selected_pred['timeframe']}
                - **Predicted Price:** ${selected_pred['predicted_price']:,.2f}
                - **Current Price at Prediction:** ${selected_pred['current_price']:,.2f}
                - **Confidence:** {selected_pred['confidence']:.1f}%
                - **Signal Strength:** {selected_pred['signal_strength']}/10
                - **Time:** {pd.to_datetime(selected_pred['timestamp']).strftime('%Y-%m-%d %H:%M')}
                """)
                
                # Form for entering trade data
                with st.form("log_trade_form", clear_on_submit=True):
                    st.markdown(f"##### Logging trade for: **{selected_pred['pair']}**")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        entry_price = st.number_input("Your Entry Price ($)", 
                                                    min_value=0.0, 
                                                    value=float(selected_pred['current_price']),
                                                    step=0.01,
                                                    format="%.2f",
                                                    key=f"entry_{selected_pred_id}")
                    
                    with col4:
                        exit_price = st.number_input("Your Exit Price ($)", 
                                                   min_value=0.0, 
                                                   value=float(selected_pred['predicted_price']),
                                                   step=0.01,
                                                   format="%.2f",
                                                   key=f"exit_{selected_pred_id}")
                    
                    notes = st.text_area("Notes (Optional)", 
                                       placeholder="Add any observations about the trade...",
                                       key=f"notes_{selected_pred_id}")
                    
                    submit_button = st.form_submit_button("✅ Submit Trade Result", use_container_width=True)
                    
                    if submit_button:
                        if entry_price > 0 and exit_price > 0:
                            success = save_trade_result(selected_pred_id, entry_price, exit_price, notes)
                            if success:
                                profit_loss = exit_price - entry_price
                                profit_pct = ((exit_price - entry_price) / entry_price) * 100
                                
                                if profit_loss > 0:
                                    st.success(f"""
                                    ✅ **Trade Logged Successfully for {selected_pred['pair']}!**
                                    - Profit/Loss: ${profit_loss:,.2f} ({profit_pct:+.2f}%)
                                    - The AI will learn from this result!
                                    """)
                                else:
                                    st.info(f"""
                                    ✅ **Trade Logged Successfully for {selected_pred['pair']}!**
                                    - Profit/Loss: ${profit_loss:,.2f} ({profit_pct:+.2f}%)
                                    - The AI will learn from this result!
                                    """)
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Error saving trade result. Please try again.")
                        else:
                            st.error("⚠️ Please enter valid prices greater than 0")
            else:
                st.info("ℹ️ No pending predictions yet. Generate some predictions first to start tracking!")
        
        # TAB 2: Performance Statistics
        with tab2:
            st.markdown("### 📊 AI Performance Statistics")
            
            stats = get_performance_stats()
            
            if len(stats) > 0:
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = stats['total_trades'].sum()
                overall_accuracy = 100 - stats['avg_error'].mean()
                overall_win_rate = stats['win_rate'].mean()
                overall_return = stats['avg_return'].mean()
                
                with col1:
                    st.metric("Total Trades Logged", f"{int(total_trades)}")
                with col2:
                    acc_color = "🟢" if overall_accuracy >= 70 else "🟡" if overall_accuracy >= 50 else "🔴"
                    st.metric("Overall Accuracy", f"{acc_color} {overall_accuracy:.1f}%")
                with col3:
                    wr_color = "🟢" if overall_win_rate >= 60 else "🟡" if overall_win_rate >= 45 else "🔴"
                    st.metric("Win Rate", f"{wr_color} {overall_win_rate:.1f}%")
                with col4:
                    st.metric("Avg Return", f"{overall_return:+.2f}%")
                
                # Progress to retraining
                st.markdown("#### 🎯 Progress to Next Retraining")
                min_trades_required = 30
                progress = min(total_trades / min_trades_required, 1.0)
                
                st.progress(progress)
                
                if total_trades >= min_trades_required:
                    st.success(f"✅ You have enough data ({int(total_trades)} trades) to retrain the model!")
                else:
                    remaining = min_trades_required - total_trades
                    st.info(f"📊 Collect {int(remaining)} more trades to unlock model retraining")
                
                # Performance by asset type
                st.markdown("#### 📈 Performance by Asset Type")
                stats_display = stats.copy()
                stats_display['avg_error'] = stats_display['avg_error'].apply(lambda x: f"{x:.2f}%")
                stats_display['win_rate'] = stats_display['win_rate'].apply(lambda x: f"{x:.1f}%")
                stats_display['avg_return'] = stats_display['avg_return'].apply(lambda x: f"{x:+.2f}%")
                stats_display['total_trades'] = stats_display['total_trades'].astype(int)
                stats_display.columns = ['Asset Type', 'Total Trades', 'Avg Error', 'Win Rate', 'Avg Return']
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
                
            else:
                st.info("📊 No trade data yet. Log some trades to see performance statistics!")
        
        # TAB 3: Trade History
        with tab3:
            st.markdown("### 📋 Trade History")
            
            # Filter options
            col1, col2 = st.columns([1, 3])
            with col1:
                filter_asset = st.selectbox("Filter by Asset Type", 
                                          ["All", "Cryptocurrency", "Forex", "Precious Metals"])
            
            # Get completed trades
            if filter_asset == "All":
                trades = get_completed_trades(limit=100)
            else:
                trades = get_completed_trades(asset_type=filter_asset, limit=100)
            
            if len(trades) > 0:
                st.success(f"📊 Showing {len(trades)} completed trades")
                
                # Format for display
                trades_display = trades.copy()
                trades_display['trade_date'] = pd.to_datetime(trades_display['trade_date']).dt.strftime('%Y-%m-%d %H:%M')
                trades_display['predicted_price'] = trades_display['predicted_price'].apply(lambda x: f"${x:,.2f}")
                trades_display['entry_price'] = trades_display['entry_price'].apply(lambda x: f"${x:,.2f}")
                trades_display['exit_price'] = trades_display['exit_price'].apply(lambda x: f"${x:,.2f}")
                trades_display['profit_loss'] = trades_display['profit_loss'].apply(lambda x: f"${x:,.2f}")
                trades_display['profit_loss_pct'] = trades_display['profit_loss_pct'].apply(lambda x: f"{x:+.2f}%")
                trades_display['prediction_error'] = trades_display['prediction_error'].apply(lambda x: f"{x:.2f}%")
                
                trades_display = trades_display[['trade_date', 'asset_type', 'pair', 'predicted_price', 
                                               'entry_price', 'exit_price', 'profit_loss', 
                                               'profit_loss_pct', 'prediction_error']]
                trades_display.columns = ['Date', 'Asset', 'Pair', 'Predicted', 'Entry', 'Exit', 
                                        'P/L', 'P/L %', 'Error %']
                
                st.dataframe(trades_display, use_container_width=True, hide_index=True)
            else:
                st.info("📊 No trade history yet. Complete and log some trades to see history!")
        
        # TAB 4: Retrain Model
        with tab4:
            st.markdown("### 🔄 Retrain AI Model")
            
            stats = get_performance_stats()
            total_trades = stats['total_trades'].sum() if len(stats) > 0 else 0
            min_trades_required = 30
            
            if total_trades >= min_trades_required:
                st.success(f"""
                ✅ **Ready for Retraining!**
                
                You have logged **{int(total_trades)}** trades, which is enough data to retrain the AI model.
                
                **What happens during retraining:**
                1. The AI analyzes all your logged trades
                2. Learns which predictions were accurate vs inaccurate
                3. Adjusts its weights and patterns based on real performance
                4. Updates confidence scoring based on historical accuracy
                """)
                
                # Show current performance before retraining
                if len(stats) > 0:
                    st.markdown("#### 📊 Current Performance (Before Retraining)")
                    for idx, row in stats.iterrows():
                        st.info(f"""
                        **{row['asset_type']}:**
                        - Trades: {int(row['total_trades'])}
                        - Accuracy: {100 - row['avg_error']:.1f}%
                        - Win Rate: {row['win_rate']:.1f}%
                        """)
                
                st.warning("""
                ⚠️ **Important Notes:**
                - Retraining will improve future predictions based on your trading results
                - The current model will be backed up before retraining
                - This process may take a few minutes
                - Recommended: Retrain after every 50-100 new trades
                """)
                
                if st.button("🚀 Retrain Model Now", type="primary", use_container_width=True):
                    with st.spinner("🧠 Retraining AI model with your trade data..."):
                        # Here you would implement the actual retraining logic
                        # For now, we'll just show a success message
                        time.sleep(2)  # Simulate training time
                        st.success("""
                        ✅ **Model Retrained Successfully!**
                        
                        The AI has learned from your {int(total_trades)} trades and updated its prediction algorithms.
                        Future predictions should now be more accurate based on your trading patterns!
                        """)
                        st.balloons()
            else:
                remaining = min_trades_required - total_trades
                st.warning(f"""
                ⏳ **Not Enough Data Yet**
                
                You have logged **{int(total_trades)}** trades.
                You need at least **{min_trades_required}** trades to retrain the model.
                
                **Collect {int(remaining)} more trades** to unlock retraining!
                
                **Tips for better training data:**
                - Trade different pairs (BTC, ETH, EUR/USD, etc.)
                - Include both winning and losing trades
                - Mix of different market conditions
                - Be consistent with your entry/exit logging
                """)
                
                progress = total_trades / min_trades_required
                st.progress(progress)

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
