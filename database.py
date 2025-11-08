"""
Database Module - SQLite operations for trade tracking and AI learning
"""
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

# Database path
HOME = Path.home()
DB_PATH = HOME / 'trading_ai_learning.db'
print(f"💾 Database location: {DB_PATH}")


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
            indicator_snapshot TEXT,
            position_type TEXT,
            target_price REAL,
            stop_loss REAL,
            committee_position TEXT,
            committee_confidence REAL,
            committee_reasoning TEXT
        )
    ''')
    
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'actual_entry_price' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_entry_price REAL")
    if 'entry_timestamp' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN entry_timestamp TEXT")
    if 'indicator_snapshot' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN indicator_snapshot TEXT")
    
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
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(pair, timestamp DESC)")
    
    conn.commit()
    conn.close()


def save_prediction(asset_type, pair, timeframe, current_price, predicted_price, 
                   prediction_horizon, confidence, signal_strength, features, 
                   status='analysis_only', actual_entry_price=None, entry_timestamp=None,
                   indicator_snapshot=None, position_type=None, target_price=None, 
                   stop_loss=None, committee_position=None, committee_confidence=None, 
                   committee_reasoning=None):
    """Save prediction to database"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, asset_type, pair, timeframe, current_price, predicted_price,
            prediction_horizon, confidence, signal_strength, features, status,
            actual_entry_price, entry_timestamp, indicator_snapshot, position_type,
            target_price, stop_loss, committee_position, committee_confidence, committee_reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, asset_type, pair, timeframe, current_price, predicted_price,
          prediction_horizon, confidence, signal_strength, str(features), status,
          actual_entry_price, entry_timestamp, str(indicator_snapshot), position_type,
          target_price, stop_loss, committee_position, committee_confidence, committee_reasoning))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id


def mark_prediction_for_trading(prediction_id, actual_entry_price, entry_timestamp, 
                                position_type, target_price, stop_loss):
    """Mark prediction as will_trade"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE predictions 
        SET actual_entry_price = ?, entry_timestamp = ?, status = 'will_trade',
            position_type = ?, target_price = ?, stop_loss = ?
        WHERE id = ?
    ''', (actual_entry_price, entry_timestamp, position_type, target_price, stop_loss, prediction_id))
    
    conn.commit()
    conn.close()


def get_all_recent_predictions(limit=50):
    """Get recent predictions for dashboard"""
    conn = sqlite3.connect(str(DB_PATH))
    
    query = '''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT ?
    '''
    
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    
    return df


def save_trade_result(prediction_id, entry_price, exit_price, profit_loss, 
                     profit_loss_pct, prediction_error, notes=''):
    """Save trade result"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    trade_date = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO trade_results (
            prediction_id, entry_price, exit_price, trade_date, 
            profit_loss, profit_loss_pct, prediction_error, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (prediction_id, entry_price, exit_price, trade_date, 
          profit_loss, profit_loss_pct, prediction_error, notes))
    
    cursor.execute('''
        UPDATE predictions SET status = 'completed' WHERE id = ?
    ''', (prediction_id,))
    
    conn.commit()
    conn.close()


def get_indicator_weights():
    """Get current indicator weights"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT indicator_name, weight_multiplier FROM indicator_accuracy")
        weights = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return weights
    except:
        return {
            'OBV': 1.0, 'ADX': 1.0, 'Stochastic': 1.0, 'MFI': 1.0,
            'CCI': 1.0, 'Hammer': 1.0, 'Doji': 1.0, 'Shooting_Star': 1.0
        }
