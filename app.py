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
from pathlib import Path
import shutil
warnings.filterwarnings('ignore')

# ==================== DATABASE PERSISTENCE FIX ====================
HOME = Path.home()
DB_PATH = HOME / 'trading_ai_learning.db'
print(f"💾 Database location: {DB_PATH}")
# ==================================================================

# ==================== PHASE 1 ENHANCEMENTS ====================
# Documentation moved to separate files to keep UI clean
# See: PHASE1_ENHANCEMENTS_SUMMARY.md for full feature list
# =============================================================

# ==================== PHASE 1: BATCH REQUEST CAPABILITY ====================
# NOTE: Ready for use when analyzing multiple symbols simultaneously
def get_batch_data_binance(symbols_list, interval="1h", limit=100):
    """
    Batch request capability - can fetch multiple symbols at once
    
    Args:
        symbols_list: List of symbols, e.g., ['BTC', 'ETH', 'XRP']
        interval: Timeframe (same as single requests)
        limit: Number of candles per symbol
    
    Returns:
        Dictionary of {symbol: dataframe} for each symbol
    
    Usage Example:
        symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']
        data = get_batch_data_binance(symbols, '1h', 100)
        # Returns data for all 5 symbols in optimized API calls
    """
    results = {}
    
    # Binance allows batch ticker requests
    try:
        # Get all tickers at once
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        tickers = response.json()
        
        # Then get individual klines (can be optimized further with asyncio)
        for symbol in symbols_list:
            try:
                kline_url = "https://api.binance.com/api/v3/klines"
                params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
                kline_response = requests.get(kline_url, params=params, timeout=10)
                
                if kline_response.status_code == 200:
                    data = kline_response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    results[symbol] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
    except Exception as e:
        print(f"Batch request error: {e}")
    
    return results
# =============================================================================


# ==================== DATABASE FUNCTIONS ====================
def init_database():
    """Initialize SQLite database for trade tracking"""
    conn = sqlite3.connect(str(DB_PATH))
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
            status TEXT DEFAULT 'analysis_only',
            actual_entry_price REAL,
            entry_timestamp TEXT
        )
    ''')
    
    # CRITICAL FIX: Add columns if they don't exist (for existing databases)
    # Check if columns exist first
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'actual_entry_price' not in columns:
        print("🔧 Adding actual_entry_price column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_entry_price REAL")
        print("✅ actual_entry_price column added!")
    
    if 'entry_timestamp' not in columns:
        print("🔧 Adding entry_timestamp column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN entry_timestamp TEXT")
        print("✅ entry_timestamp column added!")
    
    # Debug: Show existing tracked trades
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('will_trade', 'completed')")
    count = cursor.fetchone()[0]
    print(f"📊 Database has {count} tracked trades")
    
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
    
    # ==================== ONE-TIME FIX: Update empty/NULL status values ====================
    # Fix existing predictions that have empty or NULL status
    cursor.execute('''
        UPDATE predictions 
        SET status = 'analysis_only' 
        WHERE status IS NULL OR status = '' OR LENGTH(TRIM(status)) = 0
    ''')
    
    fixed_count = cursor.rowcount
    if fixed_count > 0:
        print(f"✅ Database fix: Updated {fixed_count} predictions with empty status to 'analysis_only'")
    # ======================================================================================
    
    conn.commit()
    conn.close()

def save_prediction(asset_type, pair, timeframe, current_price, predicted_price, 
                   prediction_horizon, confidence, signal_strength, features):
    """Save a prediction to database"""
    conn = sqlite3.connect(str(DB_PATH))
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
        'analysis_only'  # Default: just analysis, not traded yet
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def mark_prediction_for_trading(prediction_id, actual_entry_price):
    """Mark a prediction as the one you're actually trading and save entry price"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # First check if prediction exists
        cursor.execute('SELECT id, status FROM predictions WHERE id = ?', (prediction_id,))
        existing = cursor.fetchone()
        
        if not existing:
            print(f"❌ ERROR: Prediction ID {prediction_id} not found in database!")
            conn.close()
            return False
        
        print(f"✅ Found prediction ID {prediction_id}, current status: {existing[1]}")
        
        # Mark this prediction as "will_trade" and save actual entry price
        cursor.execute('''
            UPDATE predictions 
            SET status = 'will_trade',
                actual_entry_price = ?,
                entry_timestamp = ?
            WHERE id = ?
        ''', (actual_entry_price, datetime.now().isoformat(), prediction_id))
        
        rows_updated = cursor.rowcount
        print(f"📝 UPDATE query affected {rows_updated} row(s)")
        
        if rows_updated == 0:
            print(f"⚠️ WARNING: No rows were updated for prediction ID {prediction_id}")
            conn.close()
            return False
        
        conn.commit()
        
        # Verify the update
        cursor.execute('SELECT status, actual_entry_price, entry_timestamp FROM predictions WHERE id = ?', (prediction_id,))
        verify = cursor.fetchone()
        
        if verify:
            print(f"✅ Verified: status='{verify[0]}', entry=${verify[1]}, timestamp={verify[2]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ ERROR in mark_prediction_for_trading: {e}")
        return False

def get_all_recent_predictions(limit=20):
    """Get all recent predictions marked for trading (tracked trades only)"""
    conn = sqlite3.connect(str(DB_PATH))
    
    query = '''
        SELECT id, timestamp, asset_type, pair, timeframe, current_price, 
               predicted_price, confidence, signal_strength, status,
               actual_entry_price, entry_timestamp
        FROM predictions 
        WHERE status IN ('will_trade', 'completed')
        ORDER BY 
            CASE status 
                WHEN 'will_trade' THEN 1 
                WHEN 'completed' THEN 2 
            END,
            timestamp DESC
        LIMIT ?
    '''
    
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

def save_trade_result(prediction_id, entry_price, exit_price, notes=""):
    """Save actual trade result"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get prediction details
    cursor.execute('SELECT predicted_price FROM predictions WHERE id = ?', (prediction_id,))
    result = cursor.fetchone()
    
    if result:
        predicted_price = result[0]
        profit_loss = exit_price - entry_price
        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # AI Error: (Predicted Exit - Actual Exit) / Actual Exit × 100
        # Negative = AI under-predicted, Positive = AI over-predicted
        prediction_error = ((predicted_price - exit_price) / exit_price) * 100
        
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
    """Get predictions that you marked for trading (will_trade status)"""
    conn = sqlite3.connect(str(DB_PATH))
    
    query = '''
        SELECT id, timestamp, asset_type, pair, timeframe, current_price, 
               predicted_price, confidence, signal_strength, actual_entry_price, entry_timestamp
        FROM predictions 
        WHERE status = 'will_trade'
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
    conn = sqlite3.connect(str(DB_PATH))
    
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
    conn = sqlite3.connect(str(DB_PATH))
    
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

def backup_database():
    """Backup database to downloads folder"""
    if DB_PATH.exists():
        try:
            backup_dir = Path.home() / 'Downloads'
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f'trading_db_backup_{datetime.now():%Y%m%d_%H%M%S}.db'
            shutil.copy(DB_PATH, backup_path)
            return backup_path
        except Exception as e:
            return None
    return None

def export_trades_to_csv():
    """Export all trades to CSV"""
    try:
        trades = get_completed_trades(limit=1000)
        if len(trades) > 0:
            csv_dir = Path.home() / 'Downloads'
            csv_dir.mkdir(exist_ok=True)
            csv_path = csv_dir / f'trades_export_{datetime.now():%Y%m%d_%H%M%S}.csv'
            trades.to_csv(csv_path, index=False)
            return csv_path
        return None
    except Exception as e:
        return None

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

# Display database location (for troubleshooting)
with st.expander("💾 Database Information", expanded=False):
    st.info(f"""
    **Database Location:** `{DB_PATH}`
    
    **File Exists:** {'✅ Yes' if DB_PATH.exists() else '❌ No'}
    
    **Note:** All your trade history and predictions are saved to this database file.
    If you move or delete this file, your history will be lost.
    
    **Tip:** Use the backup feature in the Trade Tracking tab to save your data.
    """)

st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Debug mode
debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed API information")

# Database Status (always visible for troubleshooting)
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Database Status")
try:
    db_exists = DB_PATH.exists()
    if db_exists:
        # Count records
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trade_results")
        trade_count = cursor.fetchone()[0]
        
        # Get most recent entry
        cursor.execute("SELECT MAX(timestamp) FROM predictions")
        last_pred = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(trade_date) FROM trade_results")
        last_trade = cursor.fetchone()[0]
        
        conn.close()
        
        st.sidebar.success(f"✅ Connected")
        st.sidebar.caption(f"📍 `{DB_PATH.name}`")
        st.sidebar.caption(f"📊 Predictions: {pred_count}")
        st.sidebar.caption(f"💰 Trades: {trade_count}")
        if last_pred:
            st.sidebar.caption(f"🕐 Last prediction: {last_pred[:16]}")
        if last_trade:
            st.sidebar.caption(f"🕐 Last trade: {last_trade[:16]}")
        
        # Show full path in expander for troubleshooting
        with st.sidebar.expander("🔍 Full Path", expanded=False):
            st.code(str(DB_PATH))
            if st.button("📋 Copy Path"):
                st.info("Path shown above - copy manually")
    else:
        st.sidebar.warning(f"⚠️ Database not found")
        st.sidebar.caption(f"Creating at: `{DB_PATH}`")
        # Try to create it
        init_database()
except Exception as e:
    st.sidebar.error(f"❌ Error")
    with st.sidebar.expander("Details", expanded=False):
        st.code(str(e))
st.sidebar.markdown("---")

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
    "USD/JPY": "USD/JPY",
    "USD/CHF": "USD/CHF",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "NZD/USD": "NZD/USD",
    "EUR/GBP": "EUR/GBP",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "AUD/JPY": "AUD/JPY",
    "EUR/CHF": "EUR/CHF",
    "AUD/CAD": "AUD/CAD",
    "AUD/NZD": "AUD/NZD",
    "CAD/JPY": "CAD/JPY"
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

# Implement auto-refresh with countdown
if auto_refresh:
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    elapsed = time.time() - st.session_state.last_refresh_time
    
    if elapsed >= 60:
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    else:
        remaining = int(60 - elapsed)
        st.sidebar.info(f"⏱️ Next refresh in {remaining}s")
        time.sleep(1)
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

# ==================== PHASE 1: NEW INDICATORS ====================
st.sidebar.markdown("#### 🆕 Advanced Indicators")
use_obv = st.sidebar.checkbox("OBV (Volume)", value=False, help="On-Balance Volume - tracks volume flow")
use_mfi = st.sidebar.checkbox("MFI (14)", value=False, help="Money Flow Index - volume-weighted RSI")
use_adx = st.sidebar.checkbox("ADX (14)", value=False, help="Average Directional Index - trend strength")
use_stoch = st.sidebar.checkbox("Stochastic", value=False, help="Stochastic Oscillator - momentum indicator")
use_cci = st.sidebar.checkbox("CCI (20)", value=False, help="Commodity Channel Index - cyclical trends")

# Learning Dashboard Toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 AI Learning System")
show_learning_dashboard = st.sidebar.checkbox("📊 Show Trades Table on Page", value=False,
                                              help="✅ Enable to see your tracked trades table and close trade form on each pair page")

# ==================== PHASE 1: MARKET MOVERS ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 Market Movers")
show_market_movers = st.sidebar.checkbox("📈 Show Top Movers", value=False,
                                        help="Display today's top gainers and losers")

# Function to get market movers
@st.cache_data(ttl=300)
def get_market_movers():
    """Get top movers from popular cryptocurrencies with OKX fallback"""
    popular_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']
    movers = []
    
    # Try Binance first
    binance_failed = False
    for symbol in popular_symbols:
        try:
            # Get 24h data from Binance
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": f"{symbol}USDT"}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                price_change_pct = float(data['priceChangePercent'])
                current_price = float(data['lastPrice'])
                volume = float(data['volume'])
                
                movers.append({
                    'Symbol': symbol,
                    'Price': current_price,
                    'Change %': price_change_pct,
                    'Volume': volume
                })
            else:
                binance_failed = True
                break
        except:
            binance_failed = True
            break
    
    # If Binance failed, try OKX
    if binance_failed or len(movers) == 0:
        movers = []
        for symbol in popular_symbols:
            try:
                # Get 24h data from OKX
                url = "https://www.okx.com/api/v5/market/ticker"
                params = {"instId": f"{symbol}-USDT"}
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == '0' and len(data.get('data', [])) > 0:
                        ticker = data['data'][0]
                        current_price = float(ticker['last'])
                        
                        # Calculate 24h change
                        open_24h = float(ticker['open24h'])
                        price_change_pct = ((current_price - open_24h) / open_24h) * 100
                        volume = float(ticker['vol24h'])
                        
                        movers.append({
                            'Symbol': symbol,
                            'Price': current_price,
                            'Change %': price_change_pct,
                            'Volume': volume
                        })
            except:
                continue
    
    if movers:
        df = pd.DataFrame(movers)
        df = df.sort_values('Change %', ascending=False)
        return df
    return None

if show_market_movers:
    with st.sidebar:
        movers_df = get_market_movers()
        
        if movers_df is not None and len(movers_df) > 0:
            st.markdown("#### 📈 Top Gainers")
            top_gainers = movers_df.head(3)
            for _, row in top_gainers.iterrows():
                delta = f"+{row['Change %']:.2f}%"
                st.metric(row['Symbol'], f"${row['Price']:,.2f}", delta)
            
            st.markdown("#### 📉 Top Losers")
            top_losers = movers_df.tail(3).sort_values('Change %')
            for _, row in top_losers.iterrows():
                delta = f"{row['Change %']:.2f}%"
                st.metric(row['Symbol'], f"${row['Price']:,.2f}", delta)
        else:
            st.warning("Unable to load market movers")


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

@st.cache_data(ttl=300)
def get_forex_metals_data(symbol, interval="60min", limit=100):
    """Fetch forex and precious metals data using Twelve Data API with API key"""
    # Map interval to Twelve Data format
    interval_map = {
        "5m": "5min",
        "10m": "15min",  # Twelve Data doesn't have 10min, use 15min
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day"
    }
    
    mapped_interval = interval_map.get(interval, "1h")
    
    # Get API key from secrets
    try:
        api_key = st.secrets["TWELVE_DATA_API_KEY"]
    except Exception as e:
        st.warning(f"⚠️ Twelve Data API key not found in secrets. Using free tier.")
        api_key = None
    
    # Try Twelve Data API
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": mapped_interval,
        "outputsize": min(limit, 100),
        "format": "JSON"
    }
    
    # Add API key if available
    if api_key:
        params["apikey"] = api_key
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'status' in data and data['status'] == 'error':
            st.warning(f"⚠️ Twelve Data error: {data.get('message', 'Unknown error')}")
            return None, None
        
        if 'values' not in data or not data['values']:
            st.warning(f"⚠️ No data returned for {symbol}")
            return None, None
        
        # Convert to DataFrame
        values = data['values']
        df = pd.DataFrame(values)
        
        # Rename columns and convert types
        df = df.rename(columns={
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        # Handle volume (may be 0 for forex)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        else:
            df['volume'] = 0
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        api_status = "Twelve Data (API Key)" if api_key else "Twelve Data (Free)"
        st.success(f"✅ Loaded {len(df)} data points from {api_status}")
        return df, api_status
        
    except Exception as e:
        st.warning(f"⚠️ Twelve Data API failed: {str(e)}")
        
    # Fallback: Generate sample data (for testing/demo purposes)
    try:
        st.info("📊 Using sample data for demonstration...")
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
        
        # Generate realistic price movements
        base_price = 1.0900 if 'EUR' in symbol else 110.50 if 'JPY' in symbol else 1800 if 'XAU' in symbol else 1.2500
        
        prices = []
        current_price = base_price
        for i in range(limit):
            change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility
            current_price += change
            prices.append(current_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': [p + np.random.normal(0, p * 0.001) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in range(limit)]
        })
        
        st.warning("⚠️ Using sample data. Real data unavailable.")
        return df, "Sample Data"
        
    except Exception as e:
        st.error(f"❌ Error generating sample data: {str(e)}")
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
    
    elif asset_type == "💱 Forex" or asset_type == "🏆 Precious Metals":
        # Fetch forex or precious metals data
        interval_map = timeframe_config
        
        st.info(f"🔄 Fetching {symbol} data...")
        
        # Map timeframe to forex API format
        interval = interval_map['binance']  # Use binance format as reference
        df, source = get_forex_metals_data(symbol, interval, interval_map['limit'])
        
        if df is not None and len(df) > 0:
            return df, source
        
        st.error(f"""
        ❌ **Could not fetch data for {symbol}**
        
        **Possible reasons:**
        - API rate limit reached
        - Symbol format incorrect
        - Data not available for this pair
        
        **Try:**
        - Wait a few minutes and refresh
        - Select a different pair
        """)
        return None, None
    
    return None, None

# ==================== PHASE 1: NEW TECHNICAL INDICATORS ====================
def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def calculate_mfi(df, period=14):
    """Calculate Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    positive_flow[df['close'] > df['close'].shift(1)] = money_flow[df['close'] > df['close'].shift(1)]
    negative_flow[df['close'] < df['close'].shift(1)] = money_flow[df['close'] < df['close'].shift(1)]
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    return mfi.fillna(50)

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (Trend Strength)"""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    pos_di = 100 * (pos_dm.rolling(window=period).mean() / (atr + 1e-10))
    neg_di = 100 * (neg_dm.rolling(window=period).mean() / (atr + 1e-10))
    
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0), pos_di.fillna(0), neg_di.fillna(0)

def calculate_stochastic(df, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    
    return k.fillna(50), d.fillna(50)

def calculate_cci(df, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
    return cci.fillna(0)

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
        
        # ==================== PHASE 1: ADD NEW INDICATORS ====================
        # OBV - On-Balance Volume
        df['obv'] = calculate_obv(df)
        
        # MFI - Money Flow Index
        df['mfi'] = calculate_mfi(df, 14)
        
        # ADX - Average Directional Index (Trend Strength)
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
        
        # CCI - Commodity Channel Index
        df['cci'] = calculate_cci(df, 20)
        
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
    
    # ==================== PHASE 1: NEW INDICATOR SIGNALS ====================
    # MFI - Money Flow Index
    if 'mfi' in df.columns:
        mfi = df['mfi'].iloc[-1]
        if mfi > 80:
            signals.append(-2)  # Overbought
        elif mfi < 20:
            signals.append(2)   # Oversold
        else:
            signals.append(0)
    
    # ADX - Trend Strength
    if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                signals.append(1)  # Uptrend
            else:
                signals.append(-1)  # Downtrend
    
    # Stochastic
    if 'stoch_k' in df.columns:
        stoch_k = df['stoch_k'].iloc[-1]
        if stoch_k > 80:
            signals.append(-1)  # Overbought
        elif stoch_k < 20:
            signals.append(1)   # Oversold
    
    # CCI
    if 'cci' in df.columns:
        cci = df['cci'].iloc[-1]
        if cci > 100:
            signals.append(-1)  # Overbought
        elif cci < -100:
            signals.append(1)   # Oversold
    
    return sum(signals) if signals else 0

# ==================== NEW: 3-PART WARNING SYSTEM ====================

def analyze_price_action(df, for_bullish=True):
    """
    Analyze candlestick patterns for warnings
    Returns: (has_warning, warning_details)
    """
    if len(df) < 3:
        return False, "Insufficient data"
    
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    open_price = last_candle['open']
    close_price = last_candle['close']
    high_price = last_candle['high']
    low_price = last_candle['low']
    
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price
    
    if total_range == 0:
        return False, "No range"
    
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    warnings = []
    
    if for_bullish:
        # Check for bearish reversal signals (exit warnings for longs)
        
        # Long upper wick = rejection at highs
        if upper_wick > body_size * 2 and body_size > 0:
            warnings.append(f"Long upper wick (${high_price:.2f} rejected)")
        
        # Shooting star pattern
        if upper_wick > body_size * 2.5 and lower_wick < body_size * 0.3:
            warnings.append("Shooting star pattern")
        
        # Doji at highs = indecision
        if body_size < total_range * 0.15 and close_price > df['close'].rolling(10).mean().iloc[-1]:
            warnings.append("Doji at elevated levels")
        
        # Check for weakening green candles
        if len(df) >= 3:
            last_3_bodies = []
            for i in range(-3, 0):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:  # Green candle
                    last_3_bodies.append(abs(candle['close'] - candle['open']))
            
            if len(last_3_bodies) >= 2:
                if last_3_bodies[-1] < last_3_bodies[-2] * 0.7:
                    warnings.append("Bullish momentum weakening")
    
    else:
        # Check for bullish reversal signals (entry signals for longs)
        
        # Long lower wick = support/rejection of lows
        if lower_wick > body_size * 2 and body_size > 0:
            warnings.append(f"Hammer/support at ${low_price:.2f}")
        
        # Hammer pattern
        if lower_wick > body_size * 2.5 and upper_wick < body_size * 0.3:
            warnings.append("Hammer reversal pattern")
        
        # Bullish engulfing check
        if close_price > open_price and prev_candle['close'] < prev_candle['open']:
            if open_price < prev_candle['close'] and close_price > prev_candle['open']:
                warnings.append("Bullish engulfing pattern")
    
    has_warning = len(warnings) > 0
    warning_details = " | ".join(warnings) if warnings else "Clean price action"
    
    return has_warning, warning_details

def get_obv_warning(df, for_bullish=True):
    """
    Analyze OBV for volume warnings
    Returns: (has_warning, warning_details, obv_status)
    """
    if 'obv' not in df.columns or len(df) < 5:
        return False, "OBV not available", "Unknown"
    
    obv_current = df['obv'].iloc[-1]
    obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
    obv_change = obv_current - obv_prev
    
    # Determine pressure type and momentum (from our earlier fix)
    if obv_current < 0:
        pressure_type = "Selling"
    else:
        pressure_type = "Buying"
    
    if obv_change > 0:
        if obv_current < 0:
            momentum = "Decreasing"
        else:
            momentum = "Increasing"
    elif obv_change < 0:
        if obv_current < 0:
            momentum = "Increasing"
        else:
            momentum = "Decreasing"
    else:
        momentum = "Flat"
    
    obv_status = f"{pressure_type} - {momentum}"
    
    if for_bullish:
        # For bullish signals, check for divergence
        if "Buying - Decreasing" in obv_status:
            return True, "Volume declining (Divergence warning!)", obv_status
        elif "Selling - Increasing" in obv_status:
            return True, "Selling pressure increasing", obv_status
        elif "Buying - Flat" in obv_status:
            return True, "Volume stalling", obv_status
        else:
            return False, "Volume confirming", obv_status
    else:
        # For bearish signals, check for exhaustion
        if "Selling - Decreasing" in obv_status:
            return True, "Selling pressure easing (Reversal signal)", obv_status
        elif "Buying - Increasing" in obv_status:
            return True, "Buying pressure returning", obv_status
        else:
            return False, "Selling continues", obv_status
    
    return False, obv_status, obv_status

def analyze_di_balance(df, for_bullish=True):
    """
    Analyze +DI vs -DI balance for momentum warnings
    Returns: (has_warning, warning_details, di_gap)
    """
    if 'plus_di' not in df.columns or 'minus_di' not in df.columns:
        return False, "DI not available", 0
    
    plus_di = df['plus_di'].iloc[-1]
    minus_di = df['minus_di'].iloc[-1]
    di_gap = abs(plus_di - minus_di)
    
    if for_bullish:
        # For bullish signals, check if sellers catching up
        if plus_di > minus_di:
            if di_gap < 5:
                return True, f"Buyers barely ahead (gap: {di_gap:.1f})", di_gap
            elif di_gap < 10:
                return True, f"Sellers catching up (gap: {di_gap:.1f})", di_gap
            else:
                return False, f"Buyers dominating (gap: {di_gap:.1f})", di_gap
        else:
            return True, "Sellers now in control", di_gap
    else:
        # For bearish signals, check if buyers catching up
        if minus_di > plus_di:
            if di_gap < 5:
                return True, f"Sellers barely ahead (gap: {di_gap:.1f})", di_gap
            elif di_gap < 10:
                return True, f"Buyers catching up (gap: {di_gap:.1f})", di_gap
            else:
                return False, f"Sellers dominating (gap: {di_gap:.1f})", di_gap
        else:
            return True, "Buyers now in control", di_gap
    
    return False, "Balanced", di_gap

def calculate_warning_signs(df, signal_strength):
    """
    Calculate 3-part warning system
    Returns: (warning_count, details_dict)
    """
    is_bullish = signal_strength > 0
    
    # Part 1: Price Action
    price_warning, price_details = analyze_price_action(df, for_bullish=is_bullish)
    
    # Part 2: Volume
    volume_warning, volume_details, obv_status = get_obv_warning(df, for_bullish=is_bullish)
    
    # Part 3: Momentum
    momentum_warning, momentum_details, di_gap = analyze_di_balance(df, for_bullish=is_bullish)
    
    warning_count = sum([price_warning, volume_warning, momentum_warning])
    
    details = {
        'price_warning': price_warning,
        'price_details': price_details,
        'volume_warning': volume_warning,
        'volume_details': volume_details,
        'obv_status': obv_status,
        'momentum_warning': momentum_warning,
        'momentum_details': momentum_details,
        'di_gap': di_gap,
        'warning_count': warning_count
    }
    
    return warning_count, details

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
        
        # ==================== SURGICAL FIX: SESSION STATE PREDICTION TRACKING ====================
        # CRITICAL FIX: Only save NEW prediction if we don't have one for this page load
        # Use a unique key based on symbol and current price to avoid duplicate predictions
        page_key = f"{symbol}_{current_price:.2f}_{timeframe_name}"

        if 'current_page_key' not in st.session_state or st.session_state.current_page_key != page_key:
            # New page/symbol/price - save new prediction
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
            
            # Store in session state
            st.session_state.current_page_key = page_key
            st.session_state.current_prediction_id = prediction_id
            st.session_state.last_prediction_id = prediction_id
        else:
            # Same page - use existing prediction_id
            prediction_id = st.session_state.current_prediction_id
        # ==================== END SURGICAL FIX ====================
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Prediction", f"${predictions[-1]:,.2f}", f"{pred_change:+.2f}%")
        
        with col2:
            # Confidence is already 0-100 range from model, no need to multiply
            confidence_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
            st.metric("Confidence", f"{confidence_color} {confidence:.1f}%", 
                     "High" if confidence > 80 else "Medium" if confidence > 60 else "Low")
        
        with col3:
            signal_emoji = "🟢" if signal_strength > 0 else "🔴" if signal_strength < 0 else "⚪"
            st.metric("Signal", f"{signal_emoji} {abs(signal_strength)}/10",
                     "Bullish" if signal_strength > 0 else "Bearish" if signal_strength < 0 else "Neutral")
        
        # ==================== NEW: TRACK THIS TRADE BUTTON ====================
        st.markdown("---")
        
        # Check if already tracked
        conn_check = sqlite3.connect(str(DB_PATH))
        cursor_check = conn_check.cursor()
        cursor_check.execute("SELECT status, actual_entry_price FROM predictions WHERE id = ?", (prediction_id,))
        result = cursor_check.fetchone()
        conn_check.close()
        
        is_tracked = result and result[0] == 'will_trade'
        
        if not is_tracked:
            st.info("💡 **Want to track this trade for AI learning?** Enter your actual entry price and click 'Save Trade Entry' (like Excel). Enable 'Show Learning Dashboard' in sidebar to see your trades table below!")
            
            # Show entry form when button clicked
            with st.form(key=f"track_form_{prediction_id}"):
                st.markdown(f"### 📊 Save Trade: {asset_type}")
                
                # Show Prediction ID for debugging
                st.caption(f"🔢 Prediction ID: {prediction_id}")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Predicted Entry", f"${current_price:,.2f}")
                with col_info2:
                    st.metric("Predicted Exit", f"${predictions[0]:,.2f}")
                
                st.markdown("---")
                actual_entry = st.number_input(
                    "💵 What price did YOU actually enter at?",
                    min_value=0.0,
                    value=float(current_price),
                    step=0.01,
                    format="%.2f",
                    help="Enter your real entry price from your exchange/broker",
                    key=f"entry_input_{prediction_id}"
                )
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    submit_track = st.form_submit_button("✅ Save Trade Entry", type="primary", use_container_width=True)
                with col_btn2:
                    st.caption("Entry saved immediately like Excel ✨")
                
                if submit_track:
                    if actual_entry > 0:
                        # Show what we're trying to save
                        st.info(f"""
                        💾 **Attempting to save trade:**
                        - Prediction ID: {prediction_id}
                        - Pair: {asset_type}
                        - Actual Entry: ${actual_entry:,.2f}
                        """)
                        
                        # Check if prediction exists first
                        conn_check_save = sqlite3.connect(str(DB_PATH))
                        cursor_check_save = conn_check_save.cursor()
                        cursor_check_save.execute('SELECT id, pair, status FROM predictions WHERE id = ?', (prediction_id,))
                        check_result = cursor_check_save.fetchone()
                        conn_check_save.close()
                        
                        if not check_result:
                            st.error(f"""
                            ❌ **PROBLEM FOUND!**
                            
                            Prediction ID {prediction_id} does NOT exist in database!
                            
                            **Possible causes:**
                            1. Database was reset/cleared
                            2. Prediction wasn't saved properly
                            3. Wrong prediction ID
                            
                            **Solution:** Go back to main page, generate a NEW prediction, then try tracking again.
                            """)
                        else:
                            st.info(f"✅ Found prediction: ID={check_result[0]}, Pair={check_result[1]}, Current Status={check_result[2]}")
                            
                            # Save to database
                            success = mark_prediction_for_trading(prediction_id, actual_entry)
                            
                            if success:
                                st.success(f"""
                                ✅ **Trade Saved Successfully!**
                                
                                **Pair:** {asset_type}  
                                **Your Entry:** ${actual_entry:,.2f}  
                                **Predicted Exit:** ${predictions[0]:,.2f}  
                                **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                                
                                🎯 **Next Step:** {"Scroll down to see your trade in the table below!" if show_learning_dashboard else "Check 'Show Learning Dashboard' in sidebar to see your trades table!"}
                                """)
                                
                                # Verify save
                                conn_verify = sqlite3.connect(str(DB_PATH))
                                cursor_verify = conn_verify.cursor()
                                cursor_verify.execute("SELECT status, actual_entry_price FROM predictions WHERE id = ?", (prediction_id,))
                                verify_result = cursor_verify.fetchone()
                                conn_verify.close()
                                
                                if verify_result:
                                    st.info(f"✅ Verified in database: Status='{verify_result[0]}', Entry=${verify_result[1]:,.2f}")
                                    
                                    if verify_result[0] != 'will_trade':
                                        st.error(f"⚠️ WARNING: Status is '{verify_result[0]}' instead of 'will_trade'! Something went wrong.")
                                else:
                                    st.error(f"⚠️ Could not verify save for Prediction ID {prediction_id}")
                                
                                time.sleep(2)  # Give user time to see message
                                st.rerun()
                            else:
                                st.error(f"""
                                ❌ **Failed to save trade!**
                                
                                **Prediction ID:** {prediction_id}
                                
                                **Troubleshooting:**
                                1. Check the console/terminal for error messages
                                2. Prediction ID might not exist in database
                                3. Database might be locked
                                
                                **Check console output for details!**
                                """)
                    else:
                        st.error("⚠️ Please enter a valid entry price greater than 0")
        else:
            actual_entry = result[1] if result and result[1] else current_price
            st.success(f"✅ **Trade Tracked** - Entry: ${actual_entry:,.2f} | Scroll down to see your trades table")
        
        st.markdown("---")
        # ==================== END TRACK BUTTON ====================
        
        # ==================== AI LEARNING TABLE ON PAIR PAGE ====================
        if show_learning_dashboard:
            st.markdown("---")
            st.markdown("## 📊 Your Tracked Trades (AI Learning)")
            st.info("✅ AI Learning enabled - Your tracked trades appear below")
            
            # Get tracked trades
            all_predictions = get_all_recent_predictions(limit=50)
            
            if len(all_predictions) > 0:
                # Summary metrics
                open_count = len([p for _, p in all_predictions.iterrows() if p['status'] == 'will_trade'])
                closed_count = len([p for _, p in all_predictions.iterrows() if p['status'] == 'completed'])
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("🟢 Open Trades", open_count)
                col_m2.metric("✅ Closed Trades", closed_count)
                
                st.markdown("---")
                
                # Excel-Like Table
                st.markdown("### 📋 Trades Table")
                
                # Build table
                table_data = []
                for _, row in all_predictions.iterrows():
                    entry_price = row['actual_entry_price'] if pd.notna(row['actual_entry_price']) else row['current_price']
                    entry_time = pd.to_datetime(row['entry_timestamp']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['entry_timestamp']) else pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                    
                    if row['status'] == 'will_trade':
                        status_emoji = "🟢 OPEN"
                        exit_val = "—"
                        pl_val = "—"
                        pl_pct_val = "—"
                        ai_error_val = "—"
                    else:
                        status_emoji = "✅ CLOSED"
                        
                        # Get trade results
                        conn_result = sqlite3.connect(str(DB_PATH))
                        cursor_result = conn_result.cursor()
                        cursor_result.execute('''
                            SELECT exit_price, profit_loss, profit_loss_pct 
                            FROM trade_results 
                            WHERE prediction_id = ?
                        ''', (int(row['id']),))
                        result = cursor_result.fetchone()
                        conn_result.close()
                        
                        if result:
                            exit_val = f"{result[0]:,.2f}"
                            pl_val = f"{result[1]:,.2f}"
                            pl_pct_val = f"{result[2]:+.2f}%"
                            
                            # Calculate AI Error
                            ai_error = ((row['predicted_price'] - result[0]) / result[0]) * 100
                            ai_error_val = f"{ai_error:+.2f}%"
                        else:
                            exit_val = "—"
                            pl_val = "—"
                            pl_pct_val = "—"
                            ai_error_val = "—"
                    
                    table_data.append({
                        'ID': int(row['id']),
                        'Status': status_emoji,
                        'Date': entry_time,
                        'Pair': row['pair'],
                        'Prediction Entry': f"{row['current_price']:,.2f}",
                        'Prediction Exit': f"{row['predicted_price']:,.2f}",
                        'Actual Entry': f"{entry_price:,.2f}",
                        'Actual Exit': exit_val,
                        'P/L': pl_val,
                        'P/L %': pl_pct_val,
                        'AI Error': ai_error_val,
                        'Signal': f"{row['signal_strength']}/10"
                    })
                
                df_display = pd.DataFrame(table_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Close Trade Form on pair page
                open_trades_df = all_predictions[all_predictions['status'] == 'will_trade']
                
                if len(open_trades_df) > 0:
                    st.markdown("### 📥 Close a Trade")
                    
                    # Dropdown
                    trade_options = {}
                    for _, row in open_trades_df.iterrows():
                        entry = row['actual_entry_price'] if pd.notna(row['actual_entry_price']) else row['current_price']
                        trade_options[int(row['id'])] = f"ID {int(row['id'])} - {row['pair']} (Entry: ${entry:,.2f})"
                    
                    selected_id = st.selectbox("Select trade to close:", list(trade_options.keys()), format_func=lambda x: trade_options[x], key="close_trade_pair_page")
                    
                    selected_row = open_trades_df[open_trades_df['id'] == selected_id].iloc[0]
                    actual_entry = selected_row['actual_entry_price'] if pd.notna(selected_row['actual_entry_price']) else selected_row['current_price']
                    
                    # Form
                    with st.form("close_trade_form_pair_page"):
                        st.info(f"**{selected_row['pair']}** - Entry: ${actual_entry:,.2f}")
                        
                        col_exit, col_pl = st.columns([2, 1])
                        
                        with col_exit:
                            exit_price = st.number_input(
                                "💵 Your Exit Price",
                                min_value=0.0,
                                value=float(selected_row['predicted_price']),
                                step=0.01,
                                format="%.2f"
                            )
                        
                        with col_pl:
                            est_pl = exit_price - actual_entry
                            est_pl_pct = (est_pl / actual_entry * 100) if actual_entry > 0 else 0
                            st.metric("Est. P/L", f"${est_pl:,.2f}", f"{est_pl_pct:+.2f}%")
                        
                        notes = st.text_area("Notes (Optional)")
                        
                        submit = st.form_submit_button("✅ Close Trade", type="primary", use_container_width=True)
                        
                        if submit and exit_price > 0:
                            success = save_trade_result(selected_id, actual_entry, exit_price, notes)
                            if success:
                                st.success(f"✅ Trade closed! P/L: ${est_pl:,.2f} ({est_pl_pct:+.2f}%)")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Error closing trade")
                else:
                    st.success("✅ All trades are closed!")
            else:
                st.info("ℹ️ No tracked trades yet. Use 'Save Trade Entry' button above to track trades.")
            
            st.markdown("---")
        # ==================== END AI LEARNING TABLE ====================
        
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
    
    # Determine how many subplot rows we need based on enabled indicators
    enabled_subplots = []
    subplot_titles = ['Price']
    
    # Always have price chart
    base_indicators = []
    if use_rsi:
        base_indicators.append(('RSI', 'rsi'))
    if use_macd:
        base_indicators.append(('MACD', 'macd'))
    
    # Phase 1 indicators
    phase1_indicators = []
    if use_mfi:
        phase1_indicators.append(('MFI', 'mfi'))
    if use_stoch:
        phase1_indicators.append(('Stochastic', 'stoch_k'))
    if use_adx:
        phase1_indicators.append(('ADX', 'adx'))
    if use_cci:
        phase1_indicators.append(('CCI', 'cci'))
    if use_obv:
        phase1_indicators.append(('OBV', 'obv'))
    
    # Combine all indicators
    all_indicators = base_indicators + phase1_indicators
    
    # Calculate total rows (1 for price + indicator rows)
    total_rows = 1 + len(all_indicators)
    
    # Calculate row heights dynamically
    if total_rows == 1:
        row_heights = [1.0]
    elif total_rows == 2:
        row_heights = [0.7, 0.3]
    elif total_rows == 3:
        row_heights = [0.6, 0.2, 0.2]
    else:
        # For 4+ rows: price gets 50%, others split remaining 50%
        indicator_height = 0.5 / len(all_indicators)
        row_heights = [0.5] + [indicator_height] * len(all_indicators)
    
    # Build subplot titles
    subplot_titles = ['Price'] + [ind[0] for ind in all_indicators]
    
    # Create figure with dynamic rows
    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Candlestick (always row 1)
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
    
    # Add price indicators to row 1
    if use_sma:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    if use_bb:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Add oscillator indicators to their own rows
    current_row = 2
    
    # RSI
    if use_rsi and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='blue')), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # MACD
    if use_macd and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=current_row, col=1)
        current_row += 1
    
    # ==================== PHASE 1: NEW INDICATOR CHARTS ====================
    
    # MFI
    if use_mfi and 'mfi' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mfi'], name='MFI', line=dict(color='purple')), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # Stochastic
    if use_stoch and 'stoch_k' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_k'], name='%K', line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_d'], name='%D', line=dict(color='red')), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # ADX
    if use_adx and 'adx' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], name='ADX', line=dict(color='black', width=2)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['plus_di'], name='+DI', line=dict(color='green')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['minus_di'], name='-DI', line=dict(color='red')), row=current_row, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=current_row, col=1, annotation_text="Trend Threshold")
        current_row += 1
    
    # CCI
    if use_cci and 'cci' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cci'], name='CCI', line=dict(color='orange')), row=current_row, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
        current_row += 1
    
    # OBV
    if use_obv and 'obv' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['obv'], name='OBV', line=dict(color='teal'), fill='tozeroy'), row=current_row, col=1)
        current_row += 1
    
    # Calculate appropriate height based on number of rows
    chart_height = 400 + (len(all_indicators) * 150)
    
    fig.update_layout(height=chart_height, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading recommendations
    st.markdown("### 💰 Trading Recommendations")
    
    # FIX: Signal logic - neutral zone between -2 and +2
    recent_low = df['low'].tail(20).min()
    recent_high = df['high'].tail(20).max()
    
    # Calculate pullback targets based on current indicators
    stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 50
    mfi = df['mfi'].iloc[-1] if 'mfi' in df.columns else 50
    cci = df['cci'].iloc[-1] if 'cci' in df.columns else 0
    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
    plus_di = df['plus_di'].iloc[-1] if 'plus_di' in df.columns else 25
    minus_di = df['minus_di'].iloc[-1] if 'minus_di' in df.columns else 25
    
    is_overbought = stoch_k > 70 or mfi > 70
    is_oversold = stoch_k < 30 or mfi < 30
    
    # Context-aware detection for extreme market conditions
    is_strong_trend = adx > 40
    is_extreme_overbought = (stoch_k > 85 or cci > 100) and (mfi > 65)
    is_extreme_oversold = (stoch_k < 15 or cci < -100) and (mfi < 35)
    is_bullish_trend = plus_di > minus_di
    is_bearish_trend = minus_di > plus_di
    
    # Determine market condition - CONTEXT-AWARE TIERED SYSTEM (Option B+)
    
    # SPECIAL CASE 1: Strong Uptrend but Extremely Overbought (even with neutral signal)
    if -1 <= signal_strength <= 1 and is_strong_trend and is_extreme_overbought and is_bullish_trend:
        st.warning("### ⚠️ STRONG UPTREND BUT EXTREMELY OVERBOUGHT")
        
        # Calculate pullback targets
        pullback_3pct = current_price * 0.97
        pullback_5pct = current_price * 0.95
        
        st.info(f"""
        **🎯 Market Analysis:**
        - Signal: {signal_strength}/10 (Conflicting: Trend signals vs Overbought oscillators)
        - Trend: **VERY Strong Bullish** (ADX {adx:.1f}, +DI {plus_di:.1f} >> -DI {minus_di:.1f})
        - Condition: **Extremely Overbought**
          - Stochastic: {stoch_k:.1f} {'🔴 Extreme' if stoch_k > 85 else '⚠️ High'}
          - CCI: {cci:.1f} {'🔴 Extreme' if cci > 100 else '⚠️ High'}
          - MFI: {mfi:.1f} {'🔴 Overbought' if mfi > 70 else '⚠️ High'}
        
        **⚠️ Why Signals Conflict:**
        - Trend indicators say: BUY (+2 to +3 points)
        - Oscillators say: OVERBOUGHT (-2 to -3 points)
        - Result: Signal cancelled out to {signal_strength}/10
        
        **💡 Trading Strategy:**
        
        ❌ **DO NOT BUY NOW** - Price overextended, pullback likely
        - Risk of -3% to -5% pullback in next 1-12 hours
        - Historical pattern shows overbought leads to correction
        
        ❌ **DO NOT SHORT** - Trend is too strong (ADX {adx:.1f})
        - Strong trends can stay overbought longer
        - Risk of continued upward momentum
        
        ✅ **RECOMMENDED ACTION: WAIT FOR PULLBACK**
        
        **Entry Strategy:**
        1. **Wait for pullback to:** ${pullback_3pct:,.2f} - ${pullback_5pct:,.2f} (3-5% down)
        2. **Confirm conditions:**
           - Stochastic drops to 20-50 range
           - Volume still strong (OBV rising)
           - ADX stays above 30 (trend intact)
           - Stochastic starts turning up (bounce confirmation)
        3. **Then enter LONG** with the trend
        
        **📊 Ideal Setup After Pullback:**
        """)
        
        # Calculate ideal entry after pullback
        ideal_entry = pullback_3pct
        tp1 = ideal_entry * 1.02
        tp2 = ideal_entry * 1.035
        tp3 = ideal_entry * 1.05
        sl = ideal_entry * 0.98
        
        trade_data = {
            'Level': ['Entry (after pullback)', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${ideal_entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change from Entry': ['0%', '+2%', '+3.5%', '+5%', '-2%'],
            'Risk/Reward': ['-', '1:1', '1:1.75', '1:2.5', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        
        st.caption(f"""
        ⏰ **Expected Timing:** Pullback typically occurs within 1-12 hours on 1H timeframe
        
        🎯 **Current Price:** ${current_price:,.2f} | **Target Entry:** ${pullback_3pct:,.2f} (-3%)
        
        ⚠️ **Risk Level:** MEDIUM (wait for better entry improves risk/reward significantly)
        """)
    
    # SPECIAL CASE 2: Strong Downtrend but Extremely Oversold (even with neutral signal)
    elif -1 <= signal_strength <= 1 and is_strong_trend and is_extreme_oversold and is_bearish_trend:
        st.warning("### ⚠️ STRONG DOWNTREND BUT EXTREMELY OVERSOLD")
        
        # Calculate bounce targets
        bounce_3pct = current_price * 1.03
        bounce_5pct = current_price * 1.05
        
        st.info(f"""
        **🎯 Market Analysis:**
        - Signal: {signal_strength}/10 (Conflicting: Trend signals vs Oversold oscillators)
        - Trend: **VERY Strong Bearish** (ADX {adx:.1f}, -DI {minus_di:.1f} >> +DI {plus_di:.1f})
        - Condition: **Extremely Oversold**
          - Stochastic: {stoch_k:.1f} {'🔴 Extreme' if stoch_k < 15 else '⚠️ Low'}
          - CCI: {cci:.1f} {'🔴 Extreme' if cci < -100 else '⚠️ Low'}
          - MFI: {mfi:.1f} {'🔴 Oversold' if mfi < 30 else '⚠️ Low'}
        
        **⚠️ Why Signals Conflict:**
        - Trend indicators say: SELL (-2 to -3 points)
        - Oscillators say: OVERSOLD (+2 to +3 points)
        - Result: Signal cancelled out to {signal_strength}/10
        
        **💡 Trading Strategy:**
        
        ❌ **DO NOT SELL/SHORT NOW** - Price oversold, bounce likely
        - Risk of +3% to +5% bounce in next 1-12 hours
        - Historical pattern shows oversold leads to relief rally
        
        ❌ **DO NOT BUY** - Trend is too strong down (ADX {adx:.1f})
        - Strong downtrends can stay oversold longer
        - Risk of continued downward momentum
        
        ✅ **RECOMMENDED ACTION: WAIT FOR BOUNCE**
        
        **Entry Strategy (Short):**
        1. **Wait for bounce to:** ${bounce_3pct:,.2f} - ${bounce_5pct:,.2f} (3-5% up)
        2. **Confirm conditions:**
           - Stochastic rises to 50-80 range
           - Volume declining (OBV falling)
           - ADX stays above 30 (trend intact)
           - Stochastic starts turning down (rejection confirmation)
        3. **Then enter SHORT** with the trend
        
        **📊 Ideal Setup After Bounce:**
        """)
        
        # Calculate ideal entry after bounce
        ideal_entry = bounce_3pct
        tp1 = ideal_entry * 0.98
        tp2 = ideal_entry * 0.965
        tp3 = ideal_entry * 0.95
        sl = ideal_entry * 1.02
        
        trade_data = {
            'Level': ['Entry (after bounce)', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${ideal_entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change from Entry': ['0%', '-2%', '-3.5%', '-5%', '+2%'],
            'Risk/Reward': ['-', '1:1', '1:1.75', '1:2.5', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        
        st.caption(f"""
        ⏰ **Expected Timing:** Bounce typically occurs within 1-12 hours on 1H timeframe
        
        🎯 **Current Price:** ${current_price:,.2f} | **Target Entry:** ${bounce_3pct:,.2f} (+3%)
        
        ⚠️ **Risk Level:** MEDIUM (wait for better entry improves risk/reward significantly)
        """)
    
    # SPECIAL CASE 3: Strong Trend + Extreme Oscillator + Unclear Direction
    elif -1 <= signal_strength <= 1 and is_strong_trend:
        di_difference = abs(plus_di - minus_di)
        is_direction_unclear = di_difference < 5  # DIs within 5 points = no clear direction
        
        if is_direction_unclear and (stoch_k > 85 or stoch_k < 15):
            st.warning("### ⚠️ HIGH VOLATILITY - EXTREME OSCILLATOR IN STRONG TREND")
            
            # Get AI prediction if available
            ai_direction = "bullish" if df['close'].iloc[-1] > df['close'].iloc[-5] else "bearish"
            
            st.info(f"""
            **🎯 Market Analysis:**
            - Signal: {signal_strength}/10 (Conflicting indicators)
            - Trend Strength: **VERY Strong** (ADX {adx:.1f})
            - Direction: **UNCLEAR** (+DI {plus_di:.1f} ≈ -DI {minus_di:.1f}, diff: {di_difference:.1f})
            - Stochastic: {stoch_k:.1f} {'🔴 EXTREME Overbought' if stoch_k > 85 else '🔴 EXTREME Oversold'}
            - MFI: {mfi:.1f} ({'Neutral - no volume confirmation' if 40 < mfi < 60 else 'Confirming'})
            - CCI: {cci:.1f}
            
            **⚠️ Why This is Dangerous:**
            - Very strong trend (ADX {adx:.1f}) but direction unclear
            - +DI and -DI nearly equal (within {di_difference:.1f} points)
            - Extreme oscillator reading suggests overextension
            - High risk of sudden reversal
            - Conflicting signals = Low confidence
            
            **💡 Trading Strategy:**
            
            **Option 1: WAIT (RECOMMENDED - Safest)**
            ✅ Best choice for most traders
            - Too much uncertainty despite strong trend
            - Wait for clear direction:
              * +DI > -DI by 5+ points (bullish confirmation)
              * -DI > +DI by 5+ points (bearish confirmation)
            - Wait for Stochastic to normalize (20-80 range)
            - Current risk/reward unfavorable
            
            **Option 2: Small AI-Based Entry (Aggressive)**
            ⚠️ Only for experienced traders
            - Market appears {ai_direction} based on recent price action
            - Consider VERY SMALL position (10-25% of normal size)
            - Use VERY tight stop loss (1-2% maximum)
            - Take profits quickly at first sign of weakness
            - Monitor constantly - this is NOT set-and-forget
            
            **📊 If Taking Aggressive Entry (Option 2):**
            """)
            
            if stoch_k > 85:  # Overbought scenario
                # Micro position with tight stops
                entry = current_price
                tp1 = entry * 1.01   # +1% quick exit
                tp2 = entry * 1.02   # +2% 
                sl = entry * 0.985   # -1.5% tight stop
                
                trade_data = {
                    'Level': ['Entry (HIGH RISK)', 'TP1 (Quick)', 'TP2', 'Stop Loss'],
                    'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${sl:,.2f}"],
                    'Change': ['0%', '+1%', '+2%', '-1.5%'],
                    'Risk/Reward': ['-', '1:0.67', '1:1.33', '-']
                }
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
                
                st.error("""
                ⚠️ **CRITICAL WARNING:**
                - This is a HIGH RISK setup
                - Position size: MAX 10-25% of normal
                - Exit immediately if Stochastic starts dropping
                - Direction unclear means reversal can happen anytime
                - Consider this "gambling" not "trading"
                """)
                
            else:  # Oversold scenario (stoch < 15)
                entry = current_price
                tp1 = entry * 0.99   # -1% quick exit
                tp2 = entry * 0.98   # -2%
                sl = entry * 1.015   # +1.5% tight stop
                
                trade_data = {
                    'Level': ['Entry (HIGH RISK)', 'TP1 (Quick)', 'TP2', 'Stop Loss'],
                    'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${sl:,.2f}"],
                    'Change': ['0%', '-1%', '-2%', '+1.5%'],
                    'Risk/Reward': ['-', '1:0.67', '1:1.33', '-']
                }
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
                
                st.error("""
                ⚠️ **CRITICAL WARNING:**
                - This is a HIGH RISK setup
                - Position size: MAX 10-25% of normal
                - Exit immediately if Stochastic starts rising
                - Direction unclear means reversal can happen anytime
                - Consider this "gambling" not "trading"
                """)
            
            st.caption(f"""
            🎯 **Current Price:** ${current_price:,.2f}
            
            📊 **Wait for Confirmation:**
            - Clear DI separation (±5 points difference)
            - Stochastic normalization (20-80 range)
            - Volume confirmation (OBV trending)
            
            ⚠️ **Risk Level:** VERY HIGH (extreme uncertainty)
            
            💡 **Recommendation:** 90% of traders should choose Option 1 (WAIT)
            """)
    
    # NORMAL TIERED SYSTEM: Handle all other cases
    elif signal_strength >= 3:
        # STRONG BULLISH SIGNAL
        
        # ==================== NEW: 3-PART WARNING SYSTEM ====================
        warning_count, warning_details = calculate_warning_signs(df, signal_strength)
        
        # Display warning analysis
        st.markdown("### 🎯 3-Part Behavioral Analysis")
        
        col_price, col_volume, col_momentum = st.columns(3)
        
        with col_price:
            if warning_details['price_warning']:
                st.metric("📊 Price Action", "⚠️ Warning", 
                         warning_details['price_details'],
                         delta_color="inverse")
            else:
                st.metric("📊 Price Action", "✅ Strong",
                         warning_details['price_details'],
                         delta_color="normal")
        
        with col_volume:
            if warning_details['volume_warning']:
                st.metric("💰 Volume Flow", "⚠️ Warning",
                         warning_details['volume_details'],
                         delta_color="inverse")
            else:
                st.metric("💰 Volume Flow", "✅ Confirming",
                         warning_details['volume_details'],
                         delta_color="normal")
        
        with col_momentum:
            if warning_details['momentum_warning']:
                st.metric("⚡ Momentum", "⚠️ Warning",
                         warning_details['momentum_details'],
                         delta_color="inverse")
            else:
                st.metric("⚡ Momentum", "✅ Strong",
                         warning_details['momentum_details'],
                         delta_color="normal")
        
        # Decision based on warning count
        if warning_count == 0:
            # PERFECT SETUP - All systems go
            st.success("### 🟢 STRONG BUY - ALL SYSTEMS CONFIRM")
            st.info(f"""
            **🎯 Signal Analysis:**
            - Signal Strength: {signal_strength}/10 (Strong Bullish)
            - Warning Signs: **0/3** ✅ All Clear
            - Price Action: Clean uptrend, no resistance
            - Volume: {warning_details['obv_status']}
            - Momentum: Buyers dominating (+DI/-DI gap: {warning_details['di_gap']:.1f})
            
            **💡 Recommendation: ENTER OR HOLD**
            All three behavioral indicators confirm the bullish signal.
            This is a high-probability setup.
            """)
            
            entry = current_price
            tp1 = entry * 1.015
            tp2 = entry * 1.025
            tp3 = entry * 1.035
            sl = entry * 0.98
            
            trade_data = {
                'Level': ['Entry', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
                'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
                'Change': ['0%', '+1.5%', '+2.5%', '+3.5%', '-2%'],
                'Risk/Reward': ['-', '1:0.75', '1:1.25', '1:1.75', '-']
            }
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            
            st.caption(f"""
            ✅ **Position Size:** 75-100% of normal
            📈 **Trend Context:** ADX {adx:.1f} ({('Strong' if adx > 40 else 'Moderate') if adx > 25 else 'Weak'})
            🎯 **Risk Level:** LOW-MEDIUM (all confirmations present)
            """)
        
        elif warning_count == 1:
            # CAUTION - One warning present
            st.warning("### 🟡 STRONG BULLISH - CAUTION (1 Warning)")
            st.info(f"""
            **🎯 Signal Analysis:**
            - Signal Strength: {signal_strength}/10 (Strong Bullish)
            - Warning Signs: **1/3** ⚠️ Early Warning
            
            **Warnings Detected:**
            {'🔴 Price: ' + warning_details['price_details'] if warning_details['price_warning'] else ''}
            {'🔴 Volume: ' + warning_details['volume_details'] if warning_details['volume_warning'] else ''}
            {'🔴 Momentum: ' + warning_details['momentum_details'] if warning_details['momentum_warning'] else ''}
            
            **Still Confirming:**
            {'✅ Price: ' + warning_details['price_details'] if not warning_details['price_warning'] else ''}
            {'✅ Volume: ' + warning_details['obv_status'] if not warning_details['volume_warning'] else ''}
            {'✅ Momentum: ' + warning_details['momentum_details'] if not warning_details['momentum_warning'] else ''}
            
            **💡 Recommendation: PROCEED WITH CAUTION**
            Signal is still bullish but watch the warning closely.
            Consider smaller position or tighter stops.
            """)
            
            entry = current_price
            tp1 = entry * 1.015
            tp2 = entry * 1.025
            sl = entry * 0.985
            
            trade_data = {
                'Level': ['Entry', 'TP1', 'TP2', 'Stop Loss'],
                'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${sl:,.2f}"],
                'Change': ['0%', '+1.5%', '+2.5%', '-1.5%'],
                'Risk/Reward': ['-', '1:1', '1:1.67', '-']
            }
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            
            st.caption(f"""
            ⚠️ **Position Size:** 50-75% of normal (reduce due to warning)
            📊 **Action:** Tighten stop loss, watch next candles closely
            🎯 **Risk Level:** MEDIUM
            """)
        
        elif warning_count == 2:
            # WARNING - Two warnings present
            st.warning("### 🟠 STRONG BULLISH BUT WARNINGS PRESENT (2/3)")
            
            resistance_level = None
            if warning_details['price_warning'] and 'rejected' in warning_details['price_details'].lower():
                try:
                    resistance_level = df['high'].iloc[-3:].max()
                except:
                    pass
            
            st.info(f"""
            **🎯 Signal Analysis:**
            - Signal Strength: {signal_strength}/10 (Strong Bullish)
            - Warning Signs: **2/3** 🟠 Multiple Warnings
            
            **⚠️ Warnings Detected:**
            {'🔴 Price: ' + warning_details['price_details'] if warning_details['price_warning'] else ''}
            {'🔴 Volume: ' + warning_details['volume_details'] if warning_details['volume_warning'] else ''}
            {'🔴 Momentum: ' + warning_details['momentum_details'] if warning_details['momentum_warning'] else ''}
            
            **Still Confirming:**
            {'✅ Price: ' + warning_details['price_details'] if not warning_details['price_warning'] else ''}
            {'✅ Volume: ' + warning_details['obv_status'] if not warning_details['volume_warning'] else ''}
            {'✅ Momentum: ' + warning_details['momentum_details'] if not warning_details['momentum_warning'] else ''}
            
            {f'**Resistance Detected:** ${resistance_level:,.2f}' if resistance_level else ''}
            
            **💡 Recommendation:**
            - **If NOT in position:** Wait or very small position (25%)
            - **If IN position:** Take 50% profit, tighten stops on remainder
            
            Two warnings suggest resistance forming or momentum weakening.
            """)
            
            # Show both scenarios
            st.markdown("#### For New Entry (if taking risk):")
            entry = current_price
            tp1 = entry * 1.01
            sl = entry * 0.985
            
            trade_data = {
                'Level': ['Entry', 'TP1', 'Stop Loss'],
                'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${sl:,.2f}"],
                'Change': ['0%', '+1%', '-1.5%'],
                'Risk/Reward': ['-', '1:0.67', '-']
            }
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            
            st.markdown("#### For Existing Position:")
            st.info("""
            - Take 50% profit at current price
            - Move stop on remaining 50% to break-even or -1%
            - Exit fully if 3rd warning appears
            """)
            
            st.caption(f"""
            ⚠️ **Position Size:** 25% max (high risk)
            🚨 **Alert:** Watch for 3rd warning - full exit signal
            🎯 **Risk Level:** HIGH
            """)
        
        else:  # warning_count == 3
            # DANGER - All three warnings present
            st.error("### 🔴 RESISTANCE CONFIRMED - EXIT SIGNAL (3/3 Warnings)")
            
            resistance_level = df['high'].iloc[-5:].max()
            
            st.info(f"""
            **🎯 Signal Analysis:**
            - Signal Strength: {signal_strength}/10 (Bullish but...)
            - Warning Signs: **3/3** 🔴🔴🔴 ALL WARNING
            
            **🚨 All Three Systems Warning:**
            🔴 **Price:** {warning_details['price_details']}
            🔴 **Volume:** {warning_details['volume_details']}
            🔴 **Momentum:** {warning_details['momentum_details']}
            
            **Resistance Level:** ${resistance_level:,.2f}
            **Current Price:** ${current_price:,.2f}
            
            **💡 Strong Recommendation:**
            - **If NOT in position:** ❌ DO NOT ENTER
            - **If IN position:** 🚨 EXIT 75-100% NOW
            
            **Why Exit:**
            All three behavioral indicators show:
            - Price is being rejected at resistance
            - Volume not supporting continuation (real money leaving)
            - Sellers catching up to buyers (momentum shifting)
            
            This is a classic top formation pattern.
            Better to exit early and miss 5% than ride down 15-20%.
            """)
            
            st.markdown("#### Exit Strategy:")
            current = current_price
            support1 = current * 0.97
            support2 = current * 0.95
            
            exit_data = {
                'Action': ['Exit Now', 'If held, stop at:', 'Likely support:'],
                'Price': [f"${current:,.2f}", f"${current * 0.99:,.2f}", f"${support1:,.2f} - ${support2:,.2f}"],
                'Reason': ['Avoid larger loss', 'Protect capital', 'Re-entry zone if reverses']
            }
            st.dataframe(pd.DataFrame(exit_data), use_container_width=True, hide_index=True)
            
            st.caption(f"""
            🚨 **Action Required:** Exit position
            📉 **Expected:** Pullback to ${support1:,.2f} - ${support2:,.2f} (3-5%)
            🔄 **Re-entry:** Wait for reversal signals (0/3 warnings) at support
            🎯 **Risk Level:** EXTREME (holding is gambling)
            """)
            
        # Show ADX context
        st.markdown("---")
        st.markdown("#### 📊 Trend Context:")
        if adx > 40:
            st.info(f"""
            **Strong Trend (ADX: {adx:.1f}):**
            - Strong trends can override oscillator warnings
            - But volume and price action MUST confirm
            - Current warnings: {warning_count}/3
            - Even in strong trends, respect 3/3 warnings
            """)
        elif adx > 25:
            st.info(f"""
            **Moderate Trend (ADX: {adx:.1f}):**
            - Pay attention to all warning signs
            - 2+ warnings = significant caution needed
            """)
        else:
            st.warning(f"""
            **Weak Trend (ADX: {adx:.1f}):**
            - Ranging market - mean reversion likely
            - Warning signs more reliable in weak trends
            - Consider waiting for stronger trend
            """)
        # ==================== END NEW WARNING SYSTEM ====================
    
    elif signal_strength >= 1:
        # WEAK BULLISH SIGNAL (1 to 2.99)
        st.warning("### 🟡 WEAK BUY SIGNAL")
        st.info(f"""
        **📊 Signal Strength:** {signal_strength}/10 (🟡 Weak Bullish)
        **⚠️ Confidence:** Moderate (50-79%)
        
        **💡 Recommended Strategy:**
        - Consider SMALLER position size (50% of normal)
        - Wait for confirmation if patient
        - Watch for strengthening to 3+ for full position
        
        **✅ If Taking Trade:**
        - Use tighter stop loss
        - Take profits earlier (TP1-TP2)
        - Monitor closely for signal weakening
        """)
        
        entry = current_price
        tp1 = entry * 1.01   # +1% (conservative)
        tp2 = entry * 1.02   # +2%
        tp3 = entry * 1.03   # +3%
        sl = entry * 0.985   # -1.5% (tighter)
        
        trade_data = {
            'Level': ['Entry', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '+1%', '+2%', '+3%', '-1.5%'],
            'Risk/Reward': ['-', '1:0.67', '1:1.33', '1:2', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        st.caption("⚠️ Weak signal - Consider reduced position size or wait for stronger confirmation")
        
    elif signal_strength <= -3:
        # STRONG BEARISH SIGNAL
        if is_oversold:
            # Strong bearish but oversold - recommend waiting for bounce
            st.warning("### ⚠️ STRONG BEARISH BUT OVERSOLD - WAIT FOR BOUNCE")
            
            # Calculate ideal entry zones
            bounce_conservative = current_price * 1.03  # 3% bounce
            bounce_moderate = current_price * 1.015     # 1.5% bounce
            
            st.info(f"""
            **🎯 Market Analysis:**
            - Signal: {signal_strength}/10 (🔴 Strong Bearish)
            - Confidence: High (80-100%)
            - Stochastic: {stoch_k:.1f} {'(Oversold ⚠️)' if stoch_k < 30 else ''}
            - MFI: {mfi:.1f} {'(Oversold ⚠️)' if mfi < 30 else ''}
            
            **💡 Recommended Strategy:**
            ❌ **Don't sell/short now** - Risk/reward unfavorable at current levels
            
            ✅ **Wait for bounce to:**
            - **Conservative Entry:** ${bounce_conservative:.2f} (when Stochastic rises to 70-80)
            - **Moderate Entry:** ${bounce_moderate:.2f} (when Stochastic rises to 50-60)
            
            **📊 Entry Checklist:**
            1. Price bounces to target zone
            2. Stochastic overbought (70-80) or neutral (50-60)
            3. Volume declining (OBV falling)
            4. Trend down (ADX > 25, -DI > +DI)
            5. Stochastic starts turning down (rejection confirmation)
            """)
        else:
            # Strong bearish and not oversold - STRONG SELL
            st.error("### 🔴 STRONG SELL SETUP")
            st.info(f"**Signal Strength:** {signal_strength}/10 | **Confidence:** High (80-100%)")
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
    
    elif signal_strength <= -1:
        # WEAK BEARISH SIGNAL (-1 to -2.99)
        st.warning("### 🟡 WEAK SELL SIGNAL")
        st.info(f"""
        **📊 Signal Strength:** {signal_strength}/10 (🟡 Weak Bearish)
        **⚠️ Confidence:** Moderate (50-79%)
        
        **💡 Recommended Strategy:**
        - Consider SMALLER position size (50% of normal)
        - Wait for confirmation if patient
        - Watch for weakening to -3 or below for full position
        
        **✅ If Taking Trade:**
        - Use tighter stop loss
        - Take profits earlier (TP1-TP2)
        - Monitor closely for signal strengthening
        """)
        
        entry = current_price
        tp1 = entry * 0.99   # -1% (conservative)
        tp2 = entry * 0.98   # -2%
        tp3 = entry * 0.97   # -3%
        sl = entry * 1.015   # +1.5% (tighter)
        
        trade_data = {
            'Level': ['Entry', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '-1%', '-2%', '-3%', '+1.5%'],
            'Risk/Reward': ['-', '1:0.67', '1:1.33', '1:2', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        st.caption("⚠️ Weak signal - Consider reduced position size or wait for stronger confirmation")
    
    else:
        # NEUTRAL SIGNAL (-0.99 to +0.99)
        st.info("### ⚪ NEUTRAL - NO CLEAR DIRECTION")
        
        st.warning(f"""
        **📊 Current Signal Strength: {signal_strength}/10 (Neutral)**
        **⚠️ Confidence:** Low (<50%)
        
        **Why You Should Wait:**
        - Indicators are giving conflicting signals
        - No clear directional bias
        - Risk/reward is unfavorable
        - High chance of false moves
        
        **💡 Recommended Action:**
        🚫 **Do NOT trade** - Stay on the sidelines
        
        **⏰ Wait for:**
        1. Signal strength ≥ 3 (Strong Bullish) or ≤ -3 (Strong Bearish)
        2. Or signal ≥ 1 or ≤ -1 (Weak but tradeable with caution)
        3. Multiple indicators aligned in same direction
        4. Clear trend confirmation (ADX > 25)
        5. Volume confirmation (OBV trending)
        
        **📈 Current Market Conditions:**
        - Stochastic: {stoch_k:.1f}
        - MFI: {mfi:.1f}
        - Price: ${current_price:,.2f}
        - 20-period Range: ${recent_low:,.2f} - ${recent_high:,.2f}
        
        **🎯 Possible Scenarios:**
        - If price breaks above ${recent_high:,.2f} with volume → Watch for bullish signal
        - If price breaks below ${recent_low:,.2f} with volume → Watch for bearish signal
        - If ranging continues → Keep waiting for clarity
        """)
    
    st.warning("⚠️ **Risk Warning:** Use stop-losses. Never risk more than 1-2% per trade. Not financial advice.")
    
    # ==================== PHASE 1: ADVANCED INDICATORS DASHBOARD ====================
    if any([use_obv, use_mfi, use_adx, use_stoch, use_cci]):
        st.markdown("---")
        st.markdown("### 🆕 Advanced Technical Indicators")
        
        indicator_cols = st.columns(3)
        col_idx = 0
        
        # OBV - On-Balance Volume
        if use_obv and 'obv' in df.columns:
            with indicator_cols[col_idx % 3]:
                obv_current = df['obv'].iloc[-1]
                obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
                obv_change = obv_current - obv_prev
                
                # Determine pressure type based on sign
                if obv_current < 0:
                    pressure_type = "Selling"
                    base_color = "inverse"
                else:
                    pressure_type = "Buying"
                    base_color = "normal"
                
                # Determine momentum based on change
                if obv_change > 0:
                    # Value is increasing (going up)
                    if obv_current < 0:
                        # Negative becoming less negative = Selling pressure DECREASING
                        momentum = "Decreasing"
                        momentum_emoji = "📊"  # Neutral/improving
                        trend_color = "normal"  # Green (good - selling easing)
                    else:
                        # Positive increasing = Buying pressure INCREASING
                        momentum = "Increasing"
                        momentum_emoji = "📈"
                        trend_color = "normal"  # Green (good)
                elif obv_change < 0:
                    # Value is decreasing (going down)
                    if obv_current < 0:
                        # Negative becoming more negative = Selling pressure INCREASING
                        momentum = "Increasing"
                        momentum_emoji = "📉"
                        trend_color = "inverse"  # Red (bad - more selling)
                    else:
                        # Positive decreasing = Buying pressure DECREASING
                        momentum = "Decreasing"
                        momentum_emoji = "📊"  # Neutral/warning
                        trend_color = "inverse"  # Red (bad)
                else:
                    # No change
                    momentum = "Flat"
                    momentum_emoji = "➡️"
                    trend_color = "off"
                
                # Construct clear status message
                obv_status = f"{momentum_emoji} {pressure_type} - {momentum}"
                
                st.metric("OBV (Volume Flow)", 
                         f"{obv_current:,.0f}",
                         obv_status,
                         delta_color=trend_color)
                st.caption("Tracks cumulative buying/selling pressure")
            col_idx += 1
        
        # MFI - Money Flow Index
        if use_mfi and 'mfi' in df.columns:
            with indicator_cols[col_idx % 3]:
                mfi_current = df['mfi'].iloc[-1]
                mfi_status = "🔴 Overbought" if mfi_current > 80 else "🟢 Oversold" if mfi_current < 20 else "⚪ Neutral"
                
                st.metric("MFI (Money Flow)", 
                         f"{mfi_current:.1f}",
                         mfi_status)
                st.caption("Volume-weighted RSI")
            col_idx += 1
        
        # ADX - Average Directional Index
        if use_adx and 'adx' in df.columns:
            with indicator_cols[col_idx % 3]:
                adx_current = df['adx'].iloc[-1]
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]
                
                trend_strength = "💪 Strong" if adx_current > 25 else "😐 Weak"
                trend_dir = "🟢 Up" if plus_di > minus_di else "🔴 Down"
                
                st.metric("ADX (Trend Strength)", 
                         f"{adx_current:.1f}",
                         f"{trend_strength} | {trend_dir}")
                st.caption(f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}")
            col_idx += 1
        
        # Stochastic Oscillator
        if use_stoch and 'stoch_k' in df.columns:
            with indicator_cols[col_idx % 3]:
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                stoch_status = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "⚪ Neutral"
                
                st.metric("Stochastic", 
                         f"{stoch_k:.1f}",
                         stoch_status)
                st.caption(f"%K: {stoch_k:.1f} | %D: {stoch_d:.1f}")
            col_idx += 1
        
        # CCI - Commodity Channel Index
        if use_cci and 'cci' in df.columns:
            with indicator_cols[col_idx % 3]:
                cci_current = df['cci'].iloc[-1]
                cci_status = "🔴 Overbought" if cci_current > 100 else "🟢 Oversold" if cci_current < -100 else "⚪ Neutral"
                
                st.metric("CCI (Cyclical)", 
                         f"{cci_current:.1f}",
                         cci_status)
                st.caption("Commodity Channel Index")
            col_idx += 1
        
        # Add interpretation guide
        with st.expander("📖 How to Read These Indicators"):
            st.markdown("""
            **OBV (On-Balance Volume):**
            - Rising OBV = Accumulation (Buyers stronger)
            - Falling OBV = Distribution (Sellers stronger)
            
            **MFI (Money Flow Index):**
            - >80 = Overbought (potential reversal down)
            - <20 = Oversold (potential reversal up)
            - 40-60 = Neutral zone
            
            **ADX (Trend Strength):**
            - >25 = Strong trend (trust the direction)
            - <20 = Weak/ranging market (avoid trend trades)
            - +DI > -DI = Uptrend | -DI > +DI = Downtrend
            
            **Stochastic:**
            - >80 = Overbought zone
            - <20 = Oversold zone
            - %K crossing %D = Signal change
            
            **CCI (Commodity Channel Index):**
            - >100 = Strong upward movement
            - <-100 = Strong downward movement
            - Between -100 and 100 = Normal range
            """)
    

else:
    st.error("❌ Unable to fetch data. Please check symbol and try again.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>🚀 IMPROVED AI TRADING PLATFORM - PHASE 1 ENHANCED</b></p>
    <p><b>📡 Data Source:</b> Binance API</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p><b>🧠 Core Features:</b> Pattern-Based | Context Window | RSI Learning</p>
    <p><b>🆕 Phase 1:</b> OBV | MFI | ADX | Stochastic | CCI | Market Movers</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
