"""
Database Module - SQLite operations for trade tracking and AI learning
ENHANCED WITH COMMITTEE LEARNING SYSTEM
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
    """Initialize SQLite database for trade tracking with AI learning + Committee System"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # ==================== YOUR EXISTING TABLES ====================
    
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
    
    # Check and add missing columns
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'actual_entry_price' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_entry_price REAL")
    if 'entry_timestamp' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN entry_timestamp TEXT")
    if 'indicator_snapshot' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN indicator_snapshot TEXT")
    if 'position_type' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN position_type TEXT")
    if 'target_price' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN target_price REAL")
    if 'stop_loss' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN stop_loss REAL")
    if 'committee_position' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN committee_position TEXT")
    if 'committee_confidence' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN committee_confidence REAL")
    if 'committee_reasoning' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN committee_reasoning TEXT")
    
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
            predicted_entry_price REAL,
            predicted_exit_price REAL,
            entry_slippage REAL,
            exit_slippage REAL,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    
    # Add new columns if they don't exist
    cursor.execute("PRAGMA table_info(trade_results)")
    trade_columns = [column[1] for column in cursor.fetchall()]
    
    if 'predicted_entry_price' not in trade_columns:
        cursor.execute("ALTER TABLE trade_results ADD COLUMN predicted_entry_price REAL")
    if 'predicted_exit_price' not in trade_columns:
        cursor.execute("ALTER TABLE trade_results ADD COLUMN predicted_exit_price REAL")
    if 'entry_slippage' not in trade_columns:
        cursor.execute("ALTER TABLE trade_results ADD COLUMN entry_slippage REAL")
    if 'exit_slippage' not in trade_columns:
        cursor.execute("ALTER TABLE trade_results ADD COLUMN exit_slippage REAL")
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicator_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name TEXT NOT NULL UNIQUE,
            correct_count INTEGER DEFAULT 0,
            wrong_count INTEGER DEFAULT 0,
            missed_count INTEGER DEFAULT 0,
            accuracy_rate REAL DEFAULT 0,
            weight_multiplier REAL DEFAULT 1.0,
            last_updated TEXT NOT NULL
        )
    ''')
    
    indicators = ['OBV', 'ADX', 'Stochastic', 'MFI', 'CCI', 'Hammer', 'Doji', 'Shooting_Star']
    for ind in indicators:
        cursor.execute('''
            INSERT OR IGNORE INTO indicator_accuracy 
            (indicator_name, correct_count, wrong_count, missed_count, accuracy_rate, weight_multiplier, last_updated)
            VALUES (?, 0, 0, 0, 0.5, 1.0, ?)
        ''', (ind, datetime.now().isoformat()))
    
    # ==================== 🆕 COMMITTEE LEARNING TABLES ====================
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultant_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultant_name TEXT UNIQUE NOT NULL,
            specialty TEXT,
            total_votes INTEGER DEFAULT 0,
            correct_votes INTEGER DEFAULT 0,
            wrong_votes INTEGER DEFAULT 0,
            accuracy_rate REAL DEFAULT 50.0,
            high_confidence_votes INTEGER DEFAULT 0,
            high_confidence_correct INTEGER DEFAULT 0,
            high_confidence_wrong INTEGER DEFAULT 0,
            high_confidence_accuracy REAL DEFAULT 50.0,
            medium_confidence_votes INTEGER DEFAULT 0,
            medium_confidence_correct INTEGER DEFAULT 0,
            low_confidence_votes INTEGER DEFAULT 0,
            low_confidence_correct INTEGER DEFAULT 0,
            current_weight REAL DEFAULT 1.0,
            current_streak INTEGER DEFAULT 0,
            best_streak INTEGER DEFAULT 0,
            worst_streak INTEGER DEFAULT 0,
            performance_history TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultant_name TEXT NOT NULL,
            signal_name TEXT NOT NULL,
            signal_description TEXT,
            total_occurrences INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            wrong_predictions INTEGER DEFAULT 0,
            accuracy_rate REAL DEFAULT 50.0,
            signal_weight REAL DEFAULT 1.0,
            base_score REAL DEFAULT 1.0,
            first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
            last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            market_conditions TEXT,
            UNIQUE(consultant_name, signal_name)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS committee_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            market_type TEXT,
            price_at_decision REAL NOT NULL,
            indicators_snapshot TEXT,
            
            c1_vote TEXT,
            c1_confidence TEXT,
            c1_score REAL,
            c1_weight REAL,
            c1_reasoning TEXT,
            
            c2_vote TEXT,
            c2_confidence TEXT,
            c2_score REAL,
            c2_weight REAL,
            c2_reasoning TEXT,
            
            c3_vote TEXT,
            c3_confidence TEXT,
            c3_score REAL,
            c3_weight REAL,
            c3_reasoning TEXT,
            
            c4_vote TEXT,
            c4_confidence TEXT,
            c4_score REAL,
            c4_weight REAL,
            c4_reasoning TEXT,
            
            final_decision TEXT,
            consensus_level REAL,
            total_weighted_votes TEXT,
            recommended_entry REAL,
            recommended_stop_loss REAL,
            recommended_take_profit REAL,
            
            actual_outcome TEXT,
            outcome_determined_at TEXT,
            price_at_outcome REAL,
            profit_loss_pct REAL,
            was_correct INTEGER,
            hours_to_outcome REAL,
            
            FOREIGN KEY (trade_id) REFERENCES predictions(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS committee_meeting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            meeting_summary TEXT,
            had_conflict INTEGER DEFAULT 0,
            conflict_description TEXT,
            agreement_matrix TEXT,
            volatility REAL,
            trend_strength REAL,
            volume_level TEXT,
            FOREIGN KEY (decision_id) REFERENCES committee_decisions(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_1_consultant TEXT,
            signal_1_name TEXT,
            signal_2_consultant TEXT,
            signal_2_name TEXT,
            both_present_count INTEGER DEFAULT 0,
            both_present_wins INTEGER DEFAULT 0,
            both_present_accuracy REAL DEFAULT 50.0,
            synergy_score REAL DEFAULT 0.0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Initialize consultants
    consultants = [
        ('C1', 'Technical Analysis'),
        ('C2', 'Market Sentiment'),
        ('C3', 'Risk Management'),
        ('C4', 'Trend Analysis')
    ]
    
    for name, specialty in consultants:
        cursor.execute("""
            INSERT OR IGNORE INTO consultant_performance 
            (consultant_name, specialty, current_weight, accuracy_rate, performance_history)
            VALUES (?, ?, 1.0, 50.0, '[]')
        """, (name, specialty))
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(pair, timestamp DESC)")
    
    conn.commit()
    conn.close()
    
    print("✅ Database initialized with all tables including learning system!")


# ==================== YOUR EXISTING FUNCTIONS (UNCHANGED) ====================

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


def get_all_recent_predictions(limit=200):
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


def evaluate_indicator_prediction(indicator_name, indicator_value, position_type, trade_won):
    """Evaluate if an indicator correctly predicted the trade outcome"""
    if indicator_value is None or pd.isna(indicator_value):
        return None
    
    if isinstance(indicator_value, dict):
        signal = indicator_value.get('signal', 'neutral')
        if signal == 'neutral':
            return None
        return trade_won
    
    try:
        indicator_value = float(indicator_value)
    except:
        return None
    
    if indicator_name == 'OBV':
        if indicator_value == 0:
            return None
        predicted_bullish = indicator_value > 0
        actual_bullish = (position_type == 'LONG' and trade_won) or (position_type == 'SHORT' and not trade_won)
        return predicted_bullish == actual_bullish
    
    elif indicator_name == 'MFI':
        if 20 <= indicator_value <= 80:
            return None
        if indicator_value > 80:
            predicted_bearish = True
            actual_bearish = (position_type == 'SHORT' and trade_won) or (position_type == 'LONG' and not trade_won)
            return predicted_bearish == actual_bearish
        else:
            predicted_bullish = True
            actual_bullish = (position_type == 'LONG' and trade_won) or (position_type == 'SHORT' and not trade_won)
            return predicted_bullish == actual_bullish
    
    elif indicator_name == 'ADX':
        if indicator_value <= 25:
            return None
        return trade_won
    
    elif indicator_name == 'Stochastic':
        if 20 <= indicator_value <= 80:
            return None
        if indicator_value > 80:
            predicted_bearish = True
            actual_bearish = (position_type == 'SHORT' and trade_won) or (position_type == 'LONG' and not trade_won)
            return predicted_bearish == actual_bearish
        else:
            predicted_bullish = True
            actual_bullish = (position_type == 'LONG' and trade_won) or (position_type == 'SHORT' and not trade_won)
            return predicted_bullish == actual_bullish
    
    elif indicator_name == 'CCI':
        if -100 <= indicator_value <= 100:
            return None
        if indicator_value > 100:
            predicted_bearish = True
            actual_bearish = (position_type == 'SHORT' and trade_won) or (position_type == 'LONG' and not trade_won)
            return predicted_bearish == actual_bearish
        else:
            predicted_bullish = True
            actual_bullish = (position_type == 'LONG' and trade_won) or (position_type == 'SHORT' and not trade_won)
            return predicted_bullish == actual_bullish
    
    return None


def parse_indicator_snapshot(snapshot_str):
    """Parse indicator snapshot string into dictionary"""
    try:
        import json
        import ast
        
        try:
            return json.loads(snapshot_str)
        except:
            pass
        
        try:
            return ast.literal_eval(snapshot_str)
        except:
            pass
        
        indicators = {}
        if isinstance(snapshot_str, str):
            clean_str = snapshot_str.strip('{}[]')
            pairs = clean_str.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, val = pair.split(':', 1)
                    key = key.strip().strip("'\"")
                    val = val.strip().strip("'\"")
                    try:
                        indicators[key] = float(val)
                    except:
                        indicators[key] = val
        
        return indicators
    except:
        return {}


def evaluate_and_learn_from_trade(prediction_id, trade_won, cursor):
    """Evaluate which indicators were correct and update learning"""
    cursor.execute("""
        SELECT indicator_snapshot, position_type, committee_reasoning
        FROM predictions WHERE id = ?
    """, (prediction_id,))
    
    row = cursor.fetchone()
    if not row:
        return
    
    indicator_snapshot, position_type, committee_reasoning = row
    
    indicators = parse_indicator_snapshot(indicator_snapshot)
    
    if not indicators:
        return
    
    tracked_indicators = ['OBV', 'MFI', 'ADX', 'Stochastic', 'CCI']
    
    for indicator_name in tracked_indicators:
        indicator_value = indicators.get(indicator_name.lower(), indicators.get(indicator_name))
        
        was_correct = evaluate_indicator_prediction(indicator_name, indicator_value, position_type, trade_won)
        
        if was_correct is None:
            continue
        
        if was_correct:
            cursor.execute("""
                UPDATE indicator_accuracy 
                SET correct_count = correct_count + 1,
                    last_updated = ?
                WHERE indicator_name = ?
            """, (datetime.now().isoformat(), indicator_name))
        else:
            cursor.execute("""
                UPDATE indicator_accuracy 
                SET wrong_count = wrong_count + 1,
                    last_updated = ?
                WHERE indicator_name = ?
            """, (datetime.now().isoformat(), indicator_name))
    
    cursor.execute("""
        UPDATE indicator_accuracy
        SET accuracy_rate = CAST(correct_count AS REAL) / NULLIF(correct_count + wrong_count, 0),
            weight_multiplier = CASE
                WHEN CAST(correct_count AS REAL) / NULLIF(correct_count + wrong_count, 0) >= 0.6 THEN 1.0
                WHEN CAST(correct_count AS REAL) / NULLIF(correct_count + wrong_count, 0) >= 0.5 THEN 0.9
                WHEN CAST(correct_count AS REAL) / NULLIF(correct_count + wrong_count, 0) >= 0.45 THEN 0.7
                ELSE 0.5
            END
        WHERE correct_count + wrong_count > 0
    """)


def save_trade_result(prediction_id, entry_price, exit_price, profit_loss, 
                     profit_loss_pct, prediction_error, notes=''):
    """Save trade result AND update AI learning"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    trade_date = datetime.now().isoformat()
    
    cursor.execute("""
        SELECT current_price, target_price, stop_loss, position_type
        FROM predictions WHERE id = ?
    """, (prediction_id,))
    pred_data = cursor.fetchone()
    
    predicted_entry = None
    predicted_exit = None
    entry_slippage = None
    exit_slippage = None
    
    if pred_data:
        predicted_entry = pred_data[0]
        target = pred_data[1]
        stop = pred_data[2]
        position_type = pred_data[3]
        
        if profit_loss > 0:
            predicted_exit = target
        else:
            predicted_exit = stop
        
        if predicted_entry:
            entry_slippage = entry_price - predicted_entry
        
        if predicted_exit:
            if position_type == 'LONG':
                exit_slippage = predicted_exit - exit_price
            else:
                exit_slippage = exit_price - predicted_exit
    
    cursor.execute('''
        INSERT INTO trade_results (
            prediction_id, entry_price, exit_price, trade_date, 
            profit_loss, profit_loss_pct, prediction_error, notes,
            predicted_entry_price, predicted_exit_price,
            entry_slippage, exit_slippage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (prediction_id, entry_price, exit_price, trade_date, 
          profit_loss, profit_loss_pct, prediction_error, notes,
          predicted_entry, predicted_exit, entry_slippage, exit_slippage))
    
    cursor.execute('''
        UPDATE predictions SET status = 'completed' WHERE id = ?
    ''', (prediction_id,))
    
    trade_won = profit_loss > 0
    evaluate_and_learn_from_trade(prediction_id, trade_won, cursor)
    
    cursor.execute("SELECT COUNT(*) FROM trade_results")
    total_trades = cursor.fetchone()[0]
    
    if total_trades % 25 == 0:
        print(f"\n🔄 MILESTONE: {total_trades} trades completed!")
        print(f"📊 Running full AI learning revalidation...")
        revalidate_all_indicators(cursor)
        print(f"✅ Revalidation complete!")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Trade saved and AI learning updated for prediction #{prediction_id}")
    
    return True


def revalidate_all_indicators(cursor):
    """Comprehensive revalidation of all indicators"""
    cursor.execute("""
        SELECT 
            tr.prediction_id,
            tr.profit_loss,
            p.indicator_snapshot,
            p.position_type
        FROM trade_results tr
        JOIN predictions p ON tr.prediction_id = p.id
        WHERE p.status = 'completed'
        ORDER BY tr.trade_date ASC
    """)
    
    all_trades = cursor.fetchall()
    
    indicator_stats = {
        'OBV': {'correct': 0, 'wrong': 0, 'total': 0},
        'MFI': {'correct': 0, 'wrong': 0, 'total': 0},
        'ADX': {'correct': 0, 'wrong': 0, 'total': 0},
        'Stochastic': {'correct': 0, 'wrong': 0, 'total': 0},
        'CCI': {'correct': 0, 'wrong': 0, 'total': 0}
    }
    
    for trade in all_trades:
        prediction_id, profit_loss, indicator_snapshot, position_type = trade
        trade_won = profit_loss > 0
        
        try:
            indicators = parse_indicator_snapshot(indicator_snapshot)
        except:
            indicators = {}
        
        for indicator_name in indicator_stats.keys():
            indicator_value = indicators.get(indicator_name.lower(), indicators.get(indicator_name))
            
            was_correct = evaluate_indicator_prediction(indicator_name, indicator_value, position_type, trade_won)
            
            if was_correct is not None:
                indicator_stats[indicator_name]['total'] += 1
                if was_correct:
                    indicator_stats[indicator_name]['correct'] += 1
                else:
                    indicator_stats[indicator_name]['wrong'] += 1
    
    for indicator_name, stats in indicator_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            
            if stats['total'] >= 50:
                if accuracy >= 0.65:
                    weight = 1.0
                elif accuracy >= 0.60:
                    weight = 0.95
                elif accuracy >= 0.55:
                    weight = 0.9
                elif accuracy >= 0.50:
                    weight = 0.85
                elif accuracy >= 0.45:
                    weight = 0.7
                else:
                    weight = 0.5
            elif stats['total'] >= 25:
                if accuracy >= 0.60:
                    weight = 1.0
                elif accuracy >= 0.55:
                    weight = 0.9
                elif accuracy >= 0.50:
                    weight = 0.85
                elif accuracy >= 0.45:
                    weight = 0.7
                else:
                    weight = 0.6
            else:
                if accuracy >= 0.60:
                    weight = 0.95
                elif accuracy >= 0.50:
                    weight = 0.85
                else:
                    weight = 0.75
            
            cursor.execute("""
                UPDATE indicator_accuracy
                SET correct_count = ?,
                    wrong_count = ?,
                    accuracy_rate = ?,
                    weight_multiplier = ?,
                    last_updated = ?
                WHERE indicator_name = ?
            """, (stats['correct'], stats['wrong'], accuracy, weight, 
                  datetime.now().isoformat(), indicator_name))


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


def relearn_from_past_trades():
    """Retroactively learn from all completed trades"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM trade_results tr
        JOIN predictions p ON tr.prediction_id = p.id
        WHERE p.status = 'completed'
    """)
    learned_count = cursor.fetchone()[0]
    
    print(f"\n🔄 Relearning from {learned_count} past trades...")
    
    revalidate_all_indicators(cursor)
    
    conn.commit()
    conn.close()
    
    print(f"✅ AI Learning: Analyzed {learned_count} past trades!")
    return learned_count
