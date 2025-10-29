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

# ==================== PHASE 1: BATCH REQUEST CAPABILITY ====================
def get_batch_data_binance(symbols_list, interval="1h", limit=100):
    """Batch request capability - can fetch multiple symbols at once"""
    results = {}
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        tickers = response.json()
        
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

# ==================== DATABASE FUNCTIONS ====================
def init_database():
    """Initialize SQLite database for trade tracking with AI learning"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
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
            entry_timestamp TEXT,
            indicator_snapshot TEXT
        )
    ''')
    
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
    
    if 'indicator_snapshot' not in columns:
        print("🔧 Adding indicator_snapshot column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN indicator_snapshot TEXT")
        print("✅ indicator_snapshot column added!")
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('will_trade', 'completed')")
    count = cursor.fetchone()[0]
    print(f"📊 Database has {count} tracked trades")
    
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicator_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name TEXT NOT NULL,
            correct_count INTEGER DEFAULT 0,
            wrong_count INTEGER DEFAULT 0,
            missed_count INTEGER DEFAULT 0,
            accuracy_rate REAL DEFAULT 0,
            weight_multiplier REAL DEFAULT 1.0,
            last_updated TEXT NOT NULL
        )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM indicator_accuracy")
    if cursor.fetchone()[0] == 0:
        indicators = ['OBV', 'ADX', 'Stochastic', 'MFI', 'CCI', 'Hammer', 'Doji', 'Shooting_Star']
        for ind in indicators:
            cursor.execute('''
                INSERT INTO indicator_accuracy 
                (indicator_name, correct_count, wrong_count, missed_count, accuracy_rate, weight_multiplier, last_updated)
                VALUES (?, 0, 0, 0, 0.5, 1.0, ?)
            ''', (ind, datetime.now().isoformat()))
        print("✅ Initialized indicator accuracy tracking")
    
    cursor.execute('''
        UPDATE predictions 
        SET status = 'analysis_only' 
        WHERE status IS NULL OR status = '' OR LENGTH(TRIM(status)) = 0
    ''')
    
    fixed_count = cursor.rowcount
    if fixed_count > 0:
        print(f"✅ Database fix: Updated {fixed_count} predictions with empty status to 'analysis_only'")
    
    conn.commit()
    conn.close()

def save_prediction(asset_type, pair, timeframe, current_price, predicted_price, 
                   prediction_horizon, confidence, signal_strength, features, indicator_snapshot=None):
    """Save a prediction to database with indicator snapshot"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, asset_type, pair, timeframe, current_price, predicted_price, 
         prediction_horizon, confidence, signal_strength, features, status, indicator_snapshot)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        'analysis_only',
        json.dumps(indicator_snapshot) if indicator_snapshot else None
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def mark_prediction_for_trading(prediction_id, actual_entry_price):
    """Mark a prediction as the one you're actually trading and save entry price"""
    try:
        print(f"📥 mark_prediction_for_trading called with:")
        print(f"   prediction_id: {prediction_id}")
        print(f"   actual_entry_price: {actual_entry_price}")
        print(f"   actual_entry_price type: {type(actual_entry_price)}")
        
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, status, current_price FROM predictions WHERE id = ?', (prediction_id,))
        existing = cursor.fetchone()
        
        if not existing:
            print(f"❌ ERROR: Prediction ID {prediction_id} not found in database!")
            conn.close()
            return False
        
        print(f"✅ Found prediction ID {prediction_id}")
        print(f"   Current status: {existing[1]}")
        print(f"   Current price in DB: {existing[2]}")
        print(f"   About to save actual_entry_price: {actual_entry_price}")
        
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
        
        # Verify the save
        cursor.execute('SELECT status, actual_entry_price, entry_timestamp FROM predictions WHERE id = ?', (prediction_id,))
        verify = cursor.fetchone()
        
        if verify:
            print(f"✅ VERIFICATION:")
            print(f"   Status: '{verify[0]}'")
            print(f"   Actual Entry Price: {verify[1]}")
            print(f"   Entry Timestamp: {verify[2]}")
            
            if verify[1] != actual_entry_price:
                print(f"⚠️ WARNING: Saved price {verify[1]} doesn't match input {actual_entry_price}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ ERROR in mark_prediction_for_trading: {e}")
        import traceback
        print(traceback.format_exc())
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

# ==================== SURGICAL FIX 1: UPDATE save_trade_result() ====================
def save_trade_result(prediction_id, entry_price, exit_price, notes="", position_type='LONG'):
    """Save actual trade result and trigger AI learning"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('SELECT predicted_price, indicator_snapshot FROM predictions WHERE id = ?', (prediction_id,))
    result = cursor.fetchone()
    
    if result:
        predicted_price = result[0]
        indicator_snapshot = json.loads(result[1]) if result[1] else None
        
        # Calculate P/L based on position type
        if position_type == 'SHORT':
            profit_loss = entry_price - exit_price  # SHORT: profit when price goes DOWN
            profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
        else:  # LONG
            profit_loss = exit_price - entry_price  # LONG: profit when price goes UP
            profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        
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
        
        cursor.execute('UPDATE predictions SET status = ? WHERE id = ?', 
                      ('completed', prediction_id))
        
        conn.commit()
        
        if indicator_snapshot:
            was_profitable = profit_loss > 0
            analyze_indicator_accuracy(indicator_snapshot, was_profitable, cursor)
        
        cursor.execute("SELECT COUNT(*) FROM trade_results")
        total_trades = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        if should_retrain(total_trades):
            retrain_message = trigger_ai_retraining(total_trades)
            return True, retrain_message
        
        return True, None
    
    conn.close()
    return False, None
# ==================== END SURGICAL FIX 1 ====================

def analyze_indicator_accuracy(indicator_snapshot, was_profitable, cursor):
    """Analyze which indicators were correct and update accuracy scores"""
    try:
        for indicator_name, indicator_data in indicator_snapshot.items():
            signal = indicator_data.get('signal', 'neutral')
            
            if signal == 'bullish' and was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET correct_count = correct_count + 1,
                        last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bearish' and not was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET correct_count = correct_count + 1,
                        last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bullish' and not was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET wrong_count = wrong_count + 1,
                        last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bearish' and was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET wrong_count = wrong_count + 1,
                        last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'neutral':
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET missed_count = missed_count + 1,
                        last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
        
        cursor.execute("SELECT indicator_name, correct_count, wrong_count, missed_count FROM indicator_accuracy")
        for row in cursor.fetchall():
            indicator_name, correct, wrong, missed = row
            total = correct + wrong
            if total > 0:
                accuracy_rate = correct / total
                if accuracy_rate >= 0.8:
                    weight = 2.0
                elif accuracy_rate >= 0.7:
                    weight = 1.5
                elif accuracy_rate >= 0.6:
                    weight = 1.2
                elif accuracy_rate >= 0.5:
                    weight = 1.0
                elif accuracy_rate >= 0.4:
                    weight = 0.7
                else:
                    weight = 0.5
                
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET accuracy_rate = ?, weight_multiplier = ?
                    WHERE indicator_name = ?
                ''', (accuracy_rate, weight, indicator_name))
        
        print(f"✅ Updated indicator accuracy scores")
        
    except Exception as e:
        print(f"❌ Error analyzing indicator accuracy: {e}")

def should_retrain(total_trades):
    """Check if we should trigger retraining at milestone"""
    milestones = [10, 20, 30, 40, 50, 80, 100, 200, 300, 500, 1000]
    return total_trades in milestones

def trigger_ai_retraining(total_trades):
    """Trigger AI retraining and return message"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT indicator_name, accuracy_rate, weight_multiplier 
            FROM indicator_accuracy 
            ORDER BY accuracy_rate DESC
        ''')
        indicators = cursor.fetchall()
        
        if len(indicators) > 0:
            best_indicator = indicators[0]
            worst_indicator = indicators[-1]
            
            message = f"""
            🧠 **AI RETRAINING COMPLETE!**
            
            **Milestone:** {total_trades} completed trades
            
            **Best Indicator:** {best_indicator[0]} ({best_indicator[1]*100:.1f}% accuracy, {best_indicator[2]:.1f}x weight)
            **Worst Indicator:** {worst_indicator[0]} ({worst_indicator[1]*100:.1f}% accuracy, {worst_indicator[2]:.1f}x weight)
            
            **Future predictions will give more weight to accurate indicators!**
            """
            
            conn.close()
            return message
        
        conn.close()
        return f"🧠 AI Retrained on {total_trades} trades!"
        
    except Exception as e:
        print(f"❌ Error in retraining: {e}")
        return f"✅ Trade closed (retraining error)"

def get_indicator_weights():
    """Get current indicator weights for signal calculation"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT indicator_name, weight_multiplier FROM indicator_accuracy")
        weights = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return weights
    except:
        return {
            'OBV': 1.0,
            'ADX': 1.0,
            'Stochastic': 1.0,
            'MFI': 1.0,
            'CCI': 1.0,
            'Hammer': 1.0,
            'Doji': 1.0,
            'Shooting_Star': 1.0
        }

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

def create_indicator_snapshot(df):
    """Create snapshot of non-ML indicators for learning"""
    try:
        snapshot = {}
        
        if 'obv' in df.columns:
            obv_current = df['obv'].iloc[-1]
            obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
            obv_change = obv_current - obv_prev
            
            if obv_change > 0 and obv_current > 0:
                signal = 'bullish'
            elif obv_change < 0 or obv_current < 0:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['OBV'] = {'value': float(obv_current), 'signal': signal}
        
        if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
            adx = df['adx'].iloc[-1]
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            
            if adx > 25 and plus_di > minus_di:
                signal = 'bullish'
            elif adx > 25 and minus_di > plus_di:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['ADX'] = {'value': float(adx), 'signal': signal}
        
        if 'stoch_k' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            
            if stoch_k < 20:
                signal = 'bullish'
            elif stoch_k > 80:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['Stochastic'] = {'value': float(stoch_k), 'signal': signal}
        
        if 'mfi' in df.columns:
            mfi = df['mfi'].iloc[-1]
            
            if mfi < 20:
                signal = 'bullish'
            elif mfi > 80:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['MFI'] = {'value': float(mfi), 'signal': signal}
        
        if 'cci' in df.columns:
            cci = df['cci'].iloc[-1]
            
            if cci < -100:
                signal = 'bullish'
            elif cci > 100:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['CCI'] = {'value': float(cci), 'signal': signal}
        
        if len(df) >= 3:
            last_candle = df.iloc[-1]
            open_price = last_candle['open']
            close_price = last_candle['close']
            high_price = last_candle['high']
            low_price = last_candle['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0:
                upper_wick = high_price - max(open_price, close_price)
                lower_wick = min(open_price, close_price) - low_price
                
                if lower_wick > body_size * 2.5 and upper_wick < body_size * 0.3:
                    snapshot['Hammer'] = {'value': 1.0, 'signal': 'bullish'}
                else:
                    snapshot['Hammer'] = {'value': 0.0, 'signal': 'neutral'}
                
                if upper_wick > body_size * 2.5 and lower_wick < body_size * 0.3:
                    snapshot['Shooting_Star'] = {'value': 1.0, 'signal': 'bearish'}
                else:
                    snapshot['Shooting_Star'] = {'value': 0.0, 'signal': 'neutral'}
                
                if body_size < total_range * 0.15:
                    snapshot['Doji'] = {'value': 1.0, 'signal': 'neutral'}
                else:
                    snapshot['Doji'] = {'value': 0.0, 'signal': 'neutral'}
        
        return snapshot
        
    except Exception as e:
        print(f"Error creating indicator snapshot: {e}")
        return {}

init_database()
st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

st.title("🤖 AI Trading Analysis Platform - ENHANCED")
st.markdown("*Crypto, Forex, Metals + AI ML Predictions + Trading Central Format + AI Learning*")

if 'binance_blocked' not in st.session_state:
    st.session_state.binance_blocked = False

if st.session_state.binance_blocked:
    st.info("ℹ️ **Note:** Binance API is blocked in your region. Using OKX and backup APIs instead.")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")

with st.expander("💾 Database Information", expanded=False):
    st.info(f"""
    **Database Location:** `{DB_PATH}`
    
    **File Exists:** {'✅ Yes' if DB_PATH.exists() else '❌ No'}
    
    **Note:** All your trade history and predictions are saved to this database file.
    """)

st.markdown("---")

st.sidebar.header("⚙️ Configuration")

debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed API information")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Database Status")
try:
    db_exists = DB_PATH.exists()
    if db_exists:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trade_results")
        trade_count = cursor.fetchone()[0]
        
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
        
        with st.sidebar.expander("🔍 Full Path", expanded=False):
            st.code(str(DB_PATH))
            if st.button("📋 Copy Path"):
                st.info("Path shown above - copy manually")
    else:
        st.sidebar.warning(f"⚠️ Database not found")
        st.sidebar.caption(f"Creating at: `{DB_PATH}`")
        init_database()
except Exception as e:
    st.sidebar.error(f"❌ Error")
    with st.sidebar.expander("Details", expanded=False):
        st.code(str(e))
st.sidebar.markdown("---")

asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search"],
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

TIMEFRAMES = {
    "5 Minutes": {"limit": 100, "binance": "5m", "okx": "5m"},
    "10 Minutes": {"limit": 100, "binance": "10m", "okx": "10m"},
    "15 Minutes": {"limit": 100, "binance": "15m", "okx": "15m"},
    "30 Minutes": {"limit": 100, "binance": "30m", "okx": "30m"},
    "1 Hour": {"limit": 100, "binance": "1h", "okx": "1H"},
    "4 Hours": {"limit": 100, "binance": "4h", "okx": "4H"},
    "1 Day": {"limit": 100, "binance": "1d", "okx": "1D"}
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=4)
timeframe_config = TIMEFRAMES[timeframe_name]

auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False, 
                                   help="Automatically refresh data every 60 seconds")

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

st.sidebar.markdown("### 🤖 AI Configuration")
prediction_periods = st.sidebar.slider("Prediction Periods", 1, 10, 5)
lookback_hours = st.sidebar.slider("Context Window (hours)", 4, 12, 6, 
                                   help="How many hours to look back for pattern analysis")

st.sidebar.markdown("### 📊 Technical Indicators")
use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (14)", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

st.sidebar.markdown("#### 🆕 Advanced Indicators")
use_obv = st.sidebar.checkbox("OBV (Volume)", value=False, help="On-Balance Volume - tracks volume flow")
use_mfi = st.sidebar.checkbox("MFI (14)", value=False, help="Money Flow Index - volume-weighted RSI")
use_adx = st.sidebar.checkbox("ADX (14)", value=False, help="Average Directional Index - trend strength")
use_stoch = st.sidebar.checkbox("Stochastic", value=False, help="Stochastic Oscillator - momentum indicator")
use_cci = st.sidebar.checkbox("CCI (20)", value=False, help="Commodity Channel Index - cyclical trends")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 AI Learning System")
show_learning_dashboard = st.sidebar.checkbox("📊 Show Trades Table on Page", value=False,
                                              help="✅ Enable to see your tracked trades table")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 Market Movers")
show_market_movers = st.sidebar.checkbox("📈 Show Top Movers", value=False,
                                        help="Display today's top gainers and losers")

@st.cache_data(ttl=300)
def get_market_movers():
    """Get top movers from popular cryptocurrencies"""
    popular_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']
    movers = []
    
    binance_failed = False
    for symbol in popular_symbols:
        try:
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
    
    if binance_failed or len(movers) == 0:
        movers = []
        for symbol in popular_symbols:
            try:
                url = "https://www.okx.com/api/v5/market/ticker"
                params = {"instId": f"{symbol}-USDT"}
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == '0' and len(data.get('data', [])) > 0:
                        ticker = data['data'][0]
                        current_price = float(ticker['last'])
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

@st.cache_data(ttl=300)
def get_okx_data(symbol, interval="1H", limit=100):
    """Fetch data from OKX API"""
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
    """Fetch data from CryptoCompare API"""
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
    """Fetch data from CoinGecko API"""
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
        
        prices = data['prices']
        volumes = data.get('total_volumes', [[p[0], 1000000] for p in prices])
        
        df = pd.DataFrame({
            'timestamp': [pd.to_datetime(p[0], unit='ms') for p in prices],
            'close': [p[1] for p in prices],
            'volume': [v[1] for v in volumes]
        })
        
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
    """Fetch forex and precious metals data using Twelve Data API"""
    interval_map = {
        "5m": "5min",
        "10m": "15min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day"
    }
    
    mapped_interval = interval_map.get(interval, "1h")
    
    try:
        api_key = st.secrets["TWELVE_DATA_API_KEY"]
    except Exception as e:
        st.warning(f"⚠️ Twelve Data API key not found in secrets. Using free tier.")
        api_key = None
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": mapped_interval,
        "outputsize": min(limit, 100),
        "format": "JSON"
    }
    
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
        
        values = data['values']
        df = pd.DataFrame(values)
        
        df = df.rename(columns={
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        else:
            df['volume'] = 0
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        api_status = "Twelve Data (API Key)" if api_key else "Twelve Data (Free)"
        st.success(f"✅ Loaded {len(df)} data points from {api_status}")
        return df, api_status
        
    except Exception as e:
        st.warning(f"⚠️ Twelve Data API failed: {str(e)}")
        
    try:
        st.info("📊 Using sample data for demonstration...")
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
        
        base_price = 1.0900 if 'EUR' in symbol else 110.50 if 'JPY' in symbol else 1800 if 'XAU' in symbol else 1.2500
        
        prices = []
        current_price = base_price
        for i in range(limit):
            change = np.random.normal(0, base_price * 0.002)
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
        
        df, source = get_binance_data(symbol, interval_map['binance'], interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.info("🔄 Trying backup API (OKX)...")
        df, source = get_okx_data(symbol, interval_map['okx'], interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.info("🔄 Trying backup API (CryptoCompare)...")
        df, source = get_cryptocompare_data(symbol, interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.info("🔄 Trying backup API (CoinGecko)...")
        df, source = get_coingecko_data(symbol, interval_map['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.error(f"❌ Could not fetch data for {symbol}")
        return None, None
    
    elif asset_type == "💱 Forex" or asset_type == "🏆 Precious Metals":
        interval_map = timeframe_config
        
        st.info(f"🔄 Fetching {symbol} data...")
        
        interval = interval_map['binance']
        df, source = get_forex_metals_data(symbol, interval, interval_map['limit'])
        
        if df is not None and len(df) > 0:
            return df, source
        
        st.error(f"❌ Could not fetch data for {symbol}")
        return None, None
    
    return None, None

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
    """Calculate Average Directional Index"""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
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
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        df['obv'] = calculate_obv(df)
        df['mfi'] = calculate_mfi(df, 14)
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
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
    
    for i in range(10, len(rsi) - 10):
        current_rsi = rsi[i]
        future_rsi = rsi[i+1:min(i+11, len(rsi))]
        current_price = price[i]
        future_prices = price[i+1:min(i+11, len(price))]
        
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
    """Pattern-based prediction with context"""
    try:
        if len(df) < 60:
            st.warning("⚠️ Need at least 60 data points")
            return None, None, 0, None
        
        df_clean = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        X, y = create_pattern_features(df_clean, lookback=lookback)
        
        if len(X) < 30:
            st.warning("⚠️ Not enough data after cleaning")
            return None, None, 0, None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            st.error("❌ Data contains NaN values after cleaning")
            return None, None, 0, None
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_test = X_scaled[split_idx:]
        y_test = y[split_idx:]
        
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
        current_sequence = np.nan_to_num(current_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_scaled = scaler.transform(current_sequence)
        
        predictions = []
        for _ in range(prediction_periods):
            rf_pred = rf_model.predict(current_scaled)[0]
            gb_pred = gb_model.predict(current_scaled)[0]
            pred_price = 0.4 * rf_pred + 0.6 * gb_pred
            predictions.append(float(pred_price))
        
        if len(X_test) > 0:
            rf_test_pred = rf_model.predict(X_test)
            gb_test_pred = gb_model.predict(X_test)
            ensemble_pred = 0.4 * rf_test_pred + 0.6 * gb_test_pred
            
            mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
            confidence = max(0, min(100, 100 - mape))
        else:
            confidence = 65
        
        rsi_insights = analyze_rsi_bounce_patterns(df_clean)
        
        return predictions, ['Pattern-based features'], confidence, rsi_insights
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None, 0, None

def calculate_signal_strength(df):
    """Calculate trading signal strength with learned weights"""
    signals = []
    
    weights = get_indicator_weights()
    
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            signals.append(-2)
        elif rsi < 30:
            signals.append(2)
        else:
            signals.append(0)
    
    if 'macd' in df.columns:
        macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
        signals.append(1 if macd_diff > 0 else -1)
    
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
    
    if 'mfi' in df.columns:
        mfi = df['mfi'].iloc[-1]
        weight = weights.get('MFI', 1.0)
        if mfi > 80:
            signals.append(int(-2 * weight))
        elif mfi < 20:
            signals.append(int(2 * weight))
        else:
            signals.append(0)
    
    if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        weight = weights.get('ADX', 1.0)
        
        if adx > 25:
            if plus_di > minus_di:
                signals.append(int(1 * weight))
            else:
                signals.append(int(-1 * weight))
    
    if 'stoch_k' in df.columns:
        stoch_k = df['stoch_k'].iloc[-1]
        weight = weights.get('Stochastic', 1.0)
        if stoch_k > 80:
            signals.append(int(-1 * weight))
        elif stoch_k < 20:
            signals.append(int(1 * weight))
    
    if 'cci' in df.columns:
        cci = df['cci'].iloc[-1]
        weight = weights.get('CCI', 1.0)
        if cci > 100:
            signals.append(int(-1 * weight))
        elif cci < -100:
            signals.append(int(1 * weight))
    
    if 'obv' in df.columns:
        obv_current = df['obv'].iloc[-1]
        obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
        weight = weights.get('OBV', 1.0)
        
        if obv_current > obv_prev and obv_current > 0:
            signals.append(int(2 * weight))
        elif obv_current < obv_prev or obv_current < 0:
            signals.append(int(-1 * weight))
    
    return sum(signals) if signals else 0

def analyze_price_action(df, for_bullish=True):
    """Analyze candlestick patterns for warnings"""
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
        if upper_wick > body_size * 2 and body_size > 0:
            warnings.append(f"Long upper wick (${high_price:.2f} rejected)")
        
        if upper_wick > body_size * 2.5 and lower_wick < body_size * 0.3:
            warnings.append("Shooting star pattern")
        
        if body_size < total_range * 0.15 and close_price > df['close'].rolling(10).mean().iloc[-1]:
            warnings.append("Doji at elevated levels")
        
        if len(df) >= 3:
            last_3_bodies = []
            for i in range(-3, 0):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:
                    last_3_bodies.append(abs(candle['close'] - candle['open']))
            
            if len(last_3_bodies) >= 2:
                if last_3_bodies[-1] < last_3_bodies[-2] * 0.7:
                    warnings.append("Bullish momentum weakening")
    
    else:
        if lower_wick > body_size * 2 and body_size > 0:
            warnings.append(f"Hammer/support at ${low_price:.2f}")
        
        if lower_wick > body_size * 2.5 and upper_wick < body_size * 0.3:
            warnings.append("Hammer reversal pattern")
        
        if close_price > open_price and prev_candle['close'] < prev_candle['open']:
            if open_price < prev_candle['close'] and close_price > prev_candle['open']:
                warnings.append("Bullish engulfing pattern")
    
    has_warning = len(warnings) > 0
    warning_details = " | ".join(warnings) if warnings else "Clean price action"
    
    return has_warning, warning_details

def get_obv_warning(df, for_bullish=True):
    """Analyze OBV for volume warnings"""
    if 'obv' not in df.columns or len(df) < 5:
        return False, "OBV not available", "Unknown"
    
    obv_current = df['obv'].iloc[-1]
    obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
    obv_change = obv_current - obv_prev
    
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
        if "Buying - Decreasing" in obv_status:
            return True, "Volume declining (Divergence warning!)", obv_status
        elif "Selling - Increasing" in obv_status:
            return True, "Selling pressure increasing", obv_status
        elif "Buying - Flat" in obv_status:
            return True, "Volume stalling", obv_status
        else:
            return False, "Volume confirming", obv_status
    else:
        if "Selling - Decreasing" in obv_status:
            return True, "Selling pressure easing (Reversal signal)", obv_status
        elif "Buying - Increasing" in obv_status:
            return True, "Buying pressure returning", obv_status
        else:
            return False, "Selling continues", obv_status
    
    return False, obv_status, obv_status

def analyze_di_balance(df, for_bullish=True):
    """Analyze +DI vs -DI balance for momentum warnings"""
    if 'plus_di' not in df.columns or 'minus_di' not in df.columns:
        return False, "DI not available", 0
    
    plus_di = df['plus_di'].iloc[-1]
    minus_di = df['minus_di'].iloc[-1]
    di_gap = abs(plus_di - minus_di)
    
    if for_bullish:
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
    """Calculate 3-part warning system"""
    is_bullish = signal_strength > 0
    
    price_warning, price_details = analyze_price_action(df, for_bullish=is_bullish)
    volume_warning, volume_details, obv_status = get_obv_warning(df, for_bullish=is_bullish)
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

def calculate_support_resistance_levels(df, current_price):
    """Calculate 7 support and resistance levels using pivot points and technical analysis"""
    high = df['high'].tail(20).max()
    low = df['low'].tail(20).min()
    close = df['close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    all_levels = [r3, r2, r1, pivot, s1, s2, s3]
    
    all_levels.sort(reverse=True)
    
    return all_levels

with st.spinner(f"🔄 Fetching {pair_display} data..."):
    df, data_source = fetch_data(symbol, asset_type)

if df is not None and len(df) > 0:
    df = calculate_technical_indicators(df)
    
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
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
    
    st.markdown("### 🤖 Improved AI Predictions with Learning")
    st.info(f"""
    **🎯 Improvements:**
    - ✅ Monitors last {lookback_hours} hours as context
    - ✅ Analyzes RSI bounce patterns from history
    - ✅ Uses pattern-based prediction
    - ✅ Optimized ML models with feature scaling
    - 🆕 AI learns from your trades automatically!
    """)
    
    with st.spinner("🧠 Training AI models..."):
        predictions, features, confidence, rsi_insights = train_improved_model(
            df, 
            lookback=lookback_hours,
            prediction_periods=prediction_periods
        )
    
    if predictions and len(predictions) > 0:
        pred_change = ((predictions[-1] - current_price) / current_price) * 100
        signal_strength = calculate_signal_strength(df)
        
        indicator_snapshot = create_indicator_snapshot(df)
        
        page_key = f"{symbol}_{current_price:.2f}_{timeframe_name}"

        if 'current_page_key' not in st.session_state or st.session_state.current_page_key != page_key:
            prediction_id = save_prediction(
                asset_type=asset_type.replace("💰 ", "").replace("🏆 ", "").replace("💱 ", "").replace("🔍 ", ""),
                pair=symbol,
                timeframe=timeframe_name,
                current_price=current_price,
                predicted_price=predictions[-1],
                prediction_horizon=prediction_periods,
                confidence=confidence,
                signal_strength=signal_strength,
                features=features if features else {},
                indicator_snapshot=indicator_snapshot
            )
            
            st.session_state.current_page_key = page_key
            st.session_state.current_prediction_id = prediction_id
            st.session_state.last_prediction_id = prediction_id
        else:
            prediction_id = st.session_state.current_prediction_id
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Prediction", f"${predictions[-1]:,.2f}", f"{pred_change:+.2f}%")
        
        with col2:
            confidence_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
            st.metric("Confidence", f"{confidence_color} {confidence:.1f}%", 
                     "High" if confidence > 80 else "Medium" if confidence > 60 else "Low")
        
        with col3:
            signal_emoji = "🟢" if signal_strength > 0 else "🔴" if signal_strength < 0 else "⚪"
            st.metric("Signal", f"{signal_emoji} {abs(signal_strength)}/10",
                     "Bullish" if signal_strength > 0 else "Bearish" if signal_strength < 0 else "Neutral")
        
        st.markdown("---")
        
        conn_check = sqlite3.connect(str(DB_PATH))
        cursor_check = conn_check.cursor()
        cursor_check.execute("SELECT status, actual_entry_price FROM predictions WHERE id = ?", (prediction_id,))
        result = cursor_check.fetchone()
        conn_check.close()
        
        is_tracked = result and result[0] == 'will_trade'
        
        if not is_tracked:
            st.info("💡 **Want to track this trade for AI learning?** Enter your actual entry price and click 'Save Trade Entry'. The AI will learn from every trade you complete!")
            
            with st.form(key=f"track_form_{prediction_id}"):
                st.markdown(f"### 📊 Save Trade: {asset_type}")
                
                st.caption(f"🔢 Prediction ID: {prediction_id}")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("AI Suggested Entry", f"${current_price:,.2f}")
                    st.caption("(Current market price)")
                with col_info2:
                    st.metric("AI Predicted Exit", f"${predictions[0]:,.2f}")
                    st.caption("(Target price)")
                
                st.markdown("---")
                st.warning("⚠️ **Important:** Enter YOUR actual entry price below (can be different from suggested price)")
                
                actual_entry_price = st.number_input(
                    "💵 Your ACTUAL Entry Price (from your exchange):",
                    min_value=0.0,
                    value=float(current_price),
                    step=0.01,
                    format="%.2f",
                    help="This is the price YOU actually bought/entered at - it may differ from the suggested price above",
                    key=f"entry_input_{prediction_id}"
                )
                
                st.info(f"📝 **Will save:** Entry Price = ${actual_entry_price:,.2f}")
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    submit_track = st.form_submit_button("✅ Save Trade Entry", type="primary", use_container_width=True)
                with col_btn2:
                    st.caption("Entry saved immediately ✨")
                
                if submit_track:
                    if actual_entry_price > 0:
                        st.info(f"🔍 DEBUG: Saving entry price: ${actual_entry_price:,.2f} for prediction ID: {prediction_id}")
                        
                        success = mark_prediction_for_trading(prediction_id, actual_entry_price)
                        
                        if success:
                            st.success(f"""
                            ✅ **Trade Saved Successfully!**
                            
                            **Pair:** {symbol}  
                            **Your Actual Entry:** ${actual_entry_price:,.2f}  
                            **AI Predicted Exit:** ${predictions[0]:,.2f}  
                            
                            🧠 **AI will learn from this trade when you close it!**
                            """)
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save trade!")
                    else:
                        st.error("⚠️ Please enter a valid entry price greater than 0")
        else:
            actual_entry = result[1] if result and result[1] else current_price
            st.success(f"✅ **Trade Tracked** - Your Entry: ${actual_entry:,.2f}")
            st.caption(f"AI suggested entry was: ${current_price:,.2f}")
        
        st.markdown("---")
        
        if show_learning_dashboard:
            st.markdown("---")
            st.markdown("## 📊 Your Tracked Trades (AI Learning)")
            
            all_predictions = get_all_recent_predictions(limit=50)
            
            if len(all_predictions) > 0:
                open_count = len([p for _, p in all_predictions.iterrows() if p['status'] == 'will_trade'])
                closed_count = len([p for _, p in all_predictions.iterrows() if p['status'] == 'completed'])
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("🟢 Open Trades", open_count)
                col_m2.metric("✅ Closed Trades", closed_count)
                
                st.markdown("---")
                
                st.markdown("### 📋 Trades Table")
                
                table_data = []
                for _, row in all_predictions.iterrows():
                    entry_price = row['actual_entry_price'] if pd.notna(row['actual_entry_price']) else row['current_price']
                    entry_time = pd.to_datetime(row['entry_timestamp']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['entry_timestamp']) else pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                    
                    has_actual_entry = pd.notna(row['actual_entry_price'])
                    
                    if row['status'] == 'will_trade':
                        status_emoji = "🟢 OPEN"
                        exit_val = "—"
                        pl_val = "—"
                        pl_pct_val = "—"
                        ai_error_val = "—"
                    else:
                        status_emoji = "✅ CLOSED"
                        
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
                            ai_error = ((row['predicted_price'] - result[0]) / result[0]) * 100
                            ai_error_val = f"{ai_error:+.2f}%"
                        else:
                            exit_val = "—"
                            pl_val = "—"
                            pl_pct_val = "—"
                            ai_error_val = "—"
                    
                    entry_indicator = ""
                    if has_actual_entry and abs(entry_price - row['current_price']) > 0.01:
                        entry_indicator = " 📝"
                    
                    table_data.append({
                        'ID': int(row['id']),
                        'Status': status_emoji,
                        'Date': entry_time,
                        'Pair': row['pair'],
                        'AI Entry': f"{row['current_price']:,.2f}",
                        'Your Entry': f"{entry_price:,.2f}{entry_indicator}",
                        'AI Exit': f"{row['predicted_price']:,.2f}",
                        'Actual Exit': exit_val,
                        'P/L': pl_val,
                        'P/L %': pl_pct_val,
                        'AI Error': ai_error_val,
                        'Signal': f"{row['signal_strength']}/10"
                    })
                
                df_display = pd.DataFrame(table_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                open_trades_df = all_predictions[all_predictions['status'] == 'will_trade']
                
                if len(open_trades_df) > 0:
                    st.markdown("### 📥 Close a Trade")
                    
                    trade_options = {}
                    for _, row in open_trades_df.iterrows():
                        entry = row['actual_entry_price'] if pd.notna(row['actual_entry_price']) else row['current_price']
                        trade_options[int(row['id'])] = f"ID {int(row['id'])} - {row['pair']} (Entry: ${entry:,.2f})"
                    
                    selected_id = st.selectbox("Select trade to close:", list(trade_options.keys()), format_func=lambda x: trade_options[x], key="close_trade_pair_page")
                    
                    selected_row = open_trades_df[open_trades_df['id'] == selected_id].iloc[0]
                    actual_entry = selected_row['actual_entry_price'] if pd.notna(selected_row['actual_entry_price']) else selected_row['current_price']
                    
# ==================== SURGICAL FIX 2: ADD POSITION TYPE SELECTOR ====================
                    with st.form("close_trade_form_pair_page"):
                        st.info(f"**{selected_row['pair']}** - Entry: ${actual_entry:,.2f}")
                        
                        # Add position type selector
                        position_type = st.selectbox(
                            "📊 Position Type:",
                            options=['LONG', 'SHORT'],
                            index=0,
                            help="LONG = profit when price goes UP | SHORT = profit when price goes DOWN"
                        )
                        
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
                            # Calculate preview based on position type
                            if position_type == 'SHORT':
                                est_pl = actual_entry - exit_price
                            else:  # LONG
                                est_pl = exit_price - actual_entry
                            est_pl_pct = (est_pl / actual_entry * 100) if actual_entry > 0 else 0
                            st.metric("Est. P/L", f"${est_pl:,.2f}", f"{est_pl_pct:+.2f}%")
                        
                        notes = st.text_area("Notes (Optional)")
                        
                        submit = st.form_submit_button("✅ Close Trade & Trigger AI Learning", type="primary", use_container_width=True)
                        
                        if submit and exit_price > 0:
                            success, retrain_message = save_trade_result(selected_id, actual_entry, exit_price, notes, position_type)
                            if success:
                                st.success(f"✅ Trade closed! P/L: ${est_pl:,.2f} ({est_pl_pct:+.2f}%)")
                                
                                if retrain_message:
                                    st.info(retrain_message)
                                
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Error closing trade")
# ==================== END SURGICAL FIX 2 ====================
                else:
                    st.success("✅ All trades are closed!")
            else:
                st.info("ℹ️ No tracked trades yet. Use 'Save Trade Entry' button above to track trades.")
            
            st.markdown("---")
        
        if rsi_insights:
            st.success(f"**📊 RSI Historical Analysis:**\n\n{rsi_insights}")
        
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
    
    st.markdown("### 💰 Trading Recommendations")
    st.markdown("*Powered by AI/ML + Trading Central Format*")
    
    stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 50
    mfi = df['mfi'].iloc[-1] if 'mfi' in df.columns else 50
    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
    
    sr_levels = calculate_support_resistance_levels(df, current_price)
    
    key_point = sr_levels[3]
    
    if signal_strength >= 3:
        warning_count, warning_details = calculate_warning_signs(df, signal_strength)
        
        st.markdown("---")
        st.markdown("## 📊 Trading Central Signal Format")
        
        st.markdown("### 🎯 Key Point")
        st.info(f"**${key_point:,.2f}**")
        
        st.markdown("### 📈 Our Preference")
        if warning_count == 0:
            target_price = predictions[0] if predictions else current_price * 1.025
            st.success(f"**Rise to ${target_price:,.2f}**")
            st.caption("All technical indicators confirm bullish momentum. Strong buy opportunity.")
        elif warning_count == 1:
            target_price = predictions[0] if predictions else current_price * 1.015
            st.warning(f"**Rise to ${target_price:,.2f}** (with caution)")
            st.caption(f"1 warning detected: {warning_details['price_details'] if warning_details['price_warning'] else warning_details['volume_details'] if warning_details['volume_warning'] else warning_details['momentum_details']}")
        elif warning_count == 2:
            target_price = predictions[0] if predictions else current_price * 1.01
            st.warning(f"**Cautious rise to ${target_price:,.2f}**")
            st.caption(f"2 warnings present. Reduced conviction.")
        else:
            st.error("**Bearish reversal expected**")
            st.caption(f"All 3 warning systems triggered. Consider exit.")
        
        st.markdown("### 🔄 Alternative Scenario")
        downside_target_1 = key_point * 0.98
        downside_target_2 = key_point * 0.96
        st.warning(f"The downside breakout of **${key_point:,.2f}** would call for **${downside_target_1:,.2f}** and **${downside_target_2:,.2f}**")
        
        st.markdown("### 💬 Comment")
        rsi_current = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd_current = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        macd_signal_current = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else current_price
        
        comment_parts = []
        
        if rsi_current < 30:
            comment_parts.append(f"The RSI is below its neutrality area at 50 (currently {rsi_current:.1f}), indicating oversold conditions.")
        elif rsi_current > 70:
            comment_parts.append(f"The RSI is above its neutrality area at 50 (currently {rsi_current:.1f}), indicating overbought conditions.")
        else:
            comment_parts.append(f"The RSI is near neutrality at {rsi_current:.1f}.")
        
        if macd_current > macd_signal_current:
            comment_parts.append("The MACD is above its signal line and positive, suggesting bullish momentum.")
        else:
            comment_parts.append("The MACD is below its signal line, indicating bearish pressure.")
        
        if current_price > sma_20 and current_price > sma_50:
            comment_parts.append(f"The price is trading above both its 20-period MA (${sma_20:.2f}) and 50-period MA (${sma_50:.2f}).")
        elif current_price > sma_20:
            comment_parts.append(f"The price is trading above its 20-period MA (${sma_20:.2f}) but below its 50-period MA (${sma_50:.2f}).")
        else:
            comment_parts.append(f"The price is trading below both its 20-period MA (${sma_20:.2f}) and 50-period MA (${sma_50:.2f}).")
        
        full_comment = " ".join(comment_parts)
        st.info(full_comment)
        
        st.markdown("### 📊 Supports and Resistances")
        for i, level in enumerate(sr_levels):
            if i < 3:
                st.caption(f"**R{3-i}:** ${level:,.2f}")
            elif i == 3:
                st.caption(f"**Pivot:** ${level:,.2f} (Current: ${current_price:,.2f})")
            else:
                st.caption(f"**S{i-3}:** ${level:,.2f}")
        
        st.markdown("---")
        
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
        
        st.markdown("---")
        
        if warning_count == 0:
            st.success("### 🟢 STRONG BUY - ALL SYSTEMS CONFIRM")
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
        
        elif warning_count == 1:
            st.warning("### 🟡 STRONG BULLISH - CAUTION (1 Warning)")
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
            st.caption("⚠️ Reduced position size recommended (50-75%)")
        
        elif warning_count == 2:
            st.warning("### 🟠 STRONG BULLISH BUT WARNINGS PRESENT (2/3)")
            entry = current_price
            tp1 = entry * 1.01
            sl = entry * 0.99
            
            trade_data = {
                'Level': ['Entry (25% position)', 'TP1', 'Stop Loss'],
                'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${sl:,.2f}"],
                'Change': ['0%', '+1%', '-1%'],
                'Risk/Reward': ['-', '1:1', '-']
            }
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            st.info("**If NOT in position:** Wait or very small position (25%)")
            st.info("**If IN position:** Take 50% profit, tighten stops")
        
        else:
            st.error("### 🔴 RESISTANCE CONFIRMED - EXIT SIGNAL (3/3 Warnings)")
            current = current_price
            exit_target = current * 0.995
            
            trade_data = {
                'Action': ['Current Price', 'Suggested Exit', 'Max Loss Accepted'],
                'Price': [f"${current:,.2f}", f"${exit_target:,.2f}", f"${current * 1.005:,.2f}"],
                'Note': ['Exit NOW', 'Target exit', 'Emergency stop']
            }
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            st.info("**If NOT in position:** ❌ DO NOT ENTER")
            st.info("**If IN position:** 🚨 EXIT 75-100% NOW")
    
    elif signal_strength >= 1:
        st.markdown("---")
        st.markdown("## 📊 Trading Central Signal Format")
        
        st.markdown("### 🎯 Key Point")
        st.info(f"**${key_point:,.2f}**")
        
        st.markdown("### 📈 Our Preference")
        target_price = predictions[0] if predictions else current_price * 1.015
        st.warning(f"**Cautious rise to ${target_price:,.2f}**")
        st.caption(f"Signal: {signal_strength}/10 - Weak bullish momentum detected.")
        
        st.markdown("### 🔄 Alternative Scenario")
        downside_target_1 = key_point * 0.985
        downside_target_2 = key_point * 0.975
        st.warning(f"The downside breakout of **${key_point:,.2f}** would call for **${downside_target_1:,.2f}** and **${downside_target_2:,.2f}**")
        
        st.markdown("### 💬 Comment")
        st.info(f"Weak buy signal detected. Signal strength: {signal_strength}/10. Consider smaller position (50%) or wait for confirmation.")
        
        st.markdown("### 📊 Supports and Resistances")
        for i, level in enumerate(sr_levels):
            if i < 3:
                st.caption(f"**R{3-i}:** ${level:,.2f}")
            elif i == 3:
                st.caption(f"**Pivot:** ${level:,.2f} (Current: ${current_price:,.2f})")
            else:
                st.caption(f"**S{i-3}:** ${level:,.2f}")
        
        st.markdown("---")
        st.warning("### 🟡 WEAK BUY SIGNAL")
        st.info(f"Signal: {signal_strength}/10 - Consider smaller position (50%) or wait for confirmation")
        
        entry = current_price
        tp1 = entry * 1.01
        tp2 = entry * 1.02
        sl = entry * 0.99
        
        trade_data = {
            'Level': ['Entry (50% position)', 'TP1', 'TP2', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '+1%', '+2%', '-1%'],
            'Risk/Reward': ['-', '1:1', '1:2', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        st.caption("⚠️ Wait for signal ≥ 3 for full position")
    
    elif signal_strength <= -3:
        st.markdown("---")
        st.markdown("## 📊 Trading Central Signal Format")
        
        st.markdown("### 🎯 Key Point")
        st.info(f"**${key_point:,.2f}**")
        
        st.markdown("### 📉 Our Preference")
        target_price = predictions[0] if predictions else current_price * 0.975
        st.error(f"**Fall to ${target_price:,.2f}**")
        st.caption("Strong bearish signals across all indicators.")
        
        st.markdown("### 🔄 Alternative Scenario")
        upside_target_1 = key_point * 1.015
        upside_target_2 = key_point * 1.025
        st.info(f"The upside breakout of **${key_point:,.2f}** would call for **${upside_target_1:,.2f}** and **${upside_target_2:,.2f}**")
        
        st.markdown("### 💬 Comment")
        rsi_current = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        st.info(f"Strong bearish momentum confirmed. RSI at {rsi_current:.1f}, MACD bearish. Short position recommended.")
        
        st.markdown("### 📊 Supports and Resistances")
        for i, level in enumerate(sr_levels):
            if i < 3:
                st.caption(f"**R{3-i}:** ${level:,.2f}")
            elif i == 3:
                st.caption(f"**Pivot:** ${level:,.2f} (Current: ${current_price:,.2f})")
            else:
                st.caption(f"**S{i-3}:** ${level:,.2f}")
        
        st.markdown("---")
        st.error("### 🔴 STRONG SELL SIGNAL")
        entry = current_price
        tp1 = entry * 0.985
        tp2 = entry * 0.975
        tp3 = entry * 0.965
        sl = entry * 1.02
        
        trade_data = {
            'Level': ['Entry (Short)', 'TP1', 'TP2', 'TP3', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${tp3:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '-1.5%', '-2.5%', '-3.5%', '+2%'],
            'Risk/Reward': ['-', '1:0.75', '1:1.25', '1:1.75', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        st.caption("📉 Short position recommended")
    
    elif signal_strength <= -1:
        st.markdown("---")
        st.markdown("## 📊 Trading Central Signal Format")
        
        st.markdown("### 🎯 Key Point")
        st.info(f"**${key_point:,.2f}**")
        
        st.markdown("### 📉 Our Preference")
        target_price = predictions[0] if predictions else current_price * 0.99
        st.warning(f"**Cautious fall to ${target_price:,.2f}**")
        st.caption(f"Signal: {signal_strength}/10 - Weak bearish pressure.")
        
        st.markdown("### 🔄 Alternative Scenario")
        upside_target_1 = key_point * 1.01
        upside_target_2 = key_point * 1.02
        st.info(f"The upside breakout of **${key_point:,.2f}** would call for **${upside_target_1:,.2f}** and **${upside_target_2:,.2f}**")
        
        st.markdown("### 💬 Comment")
        st.info(f"Weak bearish signal. Signal strength: {signal_strength}/10. Consider smaller short position or wait.")
        
        st.markdown("### 📊 Supports and Resistances")
        for i, level in enumerate(sr_levels):
            if i < 3:
                st.caption(f"**R{3-i}:** ${level:,.2f}")
            elif i == 3:
                st.caption(f"**Pivot:** ${level:,.2f} (Current: ${current_price:,.2f})")
            else:
                st.caption(f"**S{i-3}:** ${level:,.2f}")
        
        st.markdown("---")
        st.warning("### 🟡 WEAK SELL SIGNAL")
        st.info(f"Signal: {signal_strength}/10 - Consider smaller position (50%) or wait for confirmation")
        
        entry = current_price
        tp1 = entry * 0.99
        tp2 = entry * 0.98
        sl = entry * 1.01
        
        trade_data = {
            'Level': ['Entry (50% short)', 'TP1', 'TP2', 'Stop Loss'],
            'Price': [f"${entry:,.2f}", f"${tp1:,.2f}", f"${tp2:,.2f}", f"${sl:,.2f}"],
            'Change': ['0%', '-1%', '-2%', '+1%'],
            'Risk/Reward': ['-', '1:1', '1:2', '-']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        st.caption("⚠️ Wait for signal ≤ -3 for full position")
    
    else:
        st.markdown("---")
        st.markdown("## 📊 Trading Central Signal Format")
        
        st.markdown("### 🎯 Key Point")
        st.info(f"**${key_point:,.2f}**")
        
        st.markdown("### ⚪ Market Status")
        st.info("**Neutral - No clear directional bias**")
        st.caption("Await breakout confirmation above or below key level.")
        
        st.markdown("### 🔄 Scenarios")
        upside_target = key_point * 1.015
        downside_target = key_point * 0.985
        st.info(f"**Bullish Breakout (>3 signal):** Target ${upside_target:,.2f}")
        st.warning(f"**Bearish Breakout (<-3 signal):** Target ${downside_target:,.2f}")
        
        st.markdown("### 💬 Comment")
        st.info("Indicators are mixed. Price consolidating near key level. No trading action recommended until clear signal emerges.")
        
        st.markdown("### 📊 Supports and Resistances")
        for i, level in enumerate(sr_levels):
            if i < 3:
                st.caption(f"**R{3-i}:** ${level:,.2f}")
            elif i == 3:
                st.caption(f"**Pivot:** ${level:,.2f} (Current: ${current_price:,.2f})")
            else:
                st.caption(f"**S{i-3}:** ${level:,.2f}")
        
        st.markdown("---")
        st.info("### ⚪ NEUTRAL - NO CLEAR DIRECTION")
        st.warning("🚫 Do NOT trade - Wait for signal ≥ 3 or ≤ -3")
        
        current = current_price
        bull_entry = current * 1.005
        bear_entry = current * 0.995
        
        trade_data = {
            'Scenario': ['Wait for Bullish (≥3)', 'Wait for Bearish (≤-3)', 'Current Price'],
            'Price Level': [f"${bull_entry:,.2f}", f"${bear_entry:,.2f}", f"${current:,.2f}"],
            'Action': ['Set alert above', 'Set alert below', 'Do nothing']
        }
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
    
    st.warning("⚠️ **Risk Warning:** Use stop-losses. Never risk more than 1-2% per trade. Not financial advice.")
    
    st.markdown("---")
    
    st.markdown("### 📊 Technical Chart")
    
    base_indicators = []
    if use_rsi:
        base_indicators.append(('RSI', 'rsi'))
    if use_macd:
        base_indicators.append(('MACD', 'macd'))
    
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
    
    all_indicators = base_indicators + phase1_indicators
    total_rows = 1 + len(all_indicators)
    
    if total_rows == 1:
        row_heights = [1.0]
    elif total_rows == 2:
        row_heights = [0.7, 0.3]
    elif total_rows == 3:
        row_heights = [0.6, 0.2, 0.2]
    else:
        indicator_height = 0.5 / len(all_indicators)
        row_heights = [0.5] + [indicator_height] * len(all_indicators)
    
    subplot_titles = ['Price'] + [ind[0] for ind in all_indicators]
    
    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles
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
    
    if use_sma:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    if use_bb:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    current_row = 2
    
    if use_rsi and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='blue')), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    if use_macd and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')), row=current_row, col=1)
        current_row += 1
    
    if use_mfi and 'mfi' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mfi'], name='MFI', line=dict(color='purple')), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    if use_stoch and 'stoch_k' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_k'], name='%K', line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_d'], name='%D', line=dict(color='red')), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    if use_adx and 'adx' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], name='ADX', line=dict(color='black', width=2)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['plus_di'], name='+DI', line=dict(color='green')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['minus_di'], name='-DI', line=dict(color='red')), row=current_row, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=current_row, col=1, annotation_text="Trend Threshold")
        current_row += 1
    
    if use_cci and 'cci' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cci'], name='CCI', line=dict(color='orange')), row=current_row, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
        current_row += 1
    
    if use_obv and 'obv' in df.columns and current_row <= total_rows:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['obv'], name='OBV', line=dict(color='teal'), fill='tozeroy'), row=current_row, col=1)
        current_row += 1
    
    chart_height = 400 + (len(all_indicators) * 150)
    
    fig.update_layout(height=chart_height, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    if any([use_obv, use_mfi, use_adx, use_stoch, use_cci]):
        st.markdown("---")
        st.markdown("### 🆕 Advanced Technical Indicators")
        
        indicator_cols = st.columns(3)
        col_idx = 0
        
        if use_obv and 'obv' in df.columns:
            with indicator_cols[col_idx % 3]:
                obv_current = df['obv'].iloc[-1]
                obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
                obv_change = obv_current - obv_prev
                
                if obv_current < 0:
                    pressure_type = "Selling"
                else:
                    pressure_type = "Buying"
                
                if obv_change > 0:
                    if obv_current < 0:
                        momentum = "Decreasing"
                        momentum_emoji = "📊"
                        trend_color = "normal"
                    else:
                        momentum = "Increasing"
                        momentum_emoji = "📈"
                        trend_color = "normal"
                elif obv_change < 0:
                    if obv_current < 0:
                        momentum = "Increasing"
                        momentum_emoji = "📉"
                        trend_color = "inverse"
                    else:
                        momentum = "Decreasing"
                        momentum_emoji = "📊"
                        trend_color = "inverse"
                else:
                    momentum = "Flat"
                    momentum_emoji = "➡️"
                    trend_color = "off"
                
                obv_status = f"{momentum_emoji} {pressure_type} - {momentum}"
                
                st.metric("OBV (Volume Flow)", 
                         f"{obv_current:,.0f}",
                         obv_status,
                         delta_color=trend_color)
                st.caption("Tracks cumulative buying/selling pressure")
            col_idx += 1
        
        if use_mfi and 'mfi' in df.columns:
            with indicator_cols[col_idx % 3]:
                mfi_current = df['mfi'].iloc[-1]
                mfi_status = "🔴 Overbought" if mfi_current > 80 else "🟢 Oversold" if mfi_current < 20 else "⚪ Neutral"
                
                st.metric("MFI (Money Flow)", 
                         f"{mfi_current:.1f}",
                         mfi_status)
                st.caption("Volume-weighted RSI")
            col_idx += 1
        
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
        
        if use_cci and 'cci' in df.columns:
            with indicator_cols[col_idx % 3]:
                cci_current = df['cci'].iloc[-1]
                cci_status = "🔴 Overbought" if cci_current > 100 else "🟢 Oversold" if cci_current < -100 else "⚪ Neutral"
                
                st.metric("CCI (Cyclical)", 
                         f"{cci_current:.1f}",
                         cci_status)
                st.caption("Commodity Channel Index")
            col_idx += 1

else:
    st.error("❌ Unable to fetch data. Please check symbol and try again.")

st.markdown("---")
st.markdown(f"""
<div style='text-align: center;'>
    <p><b>🚀 AI TRADING PLATFORM - ENHANCED WITH TRADING CENTRAL FORMAT</b></p>
    <p><b>🧠 AI/ML Hybrid:</b> Machine Learning + Trading Central Presentation</p>
    <p><b>🎓 AI Learning:</b> System learns from every trade automatically!</p>
    <p><b>📡 Data Source:</b> Multi-API with fallbacks (Binance, OKX, CryptoCompare, etc.)</p>
    <p><b>🔄 Last Update:</b> {current_time}</p>
    <p style='color: #888;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
