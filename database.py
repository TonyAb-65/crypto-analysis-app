# database.py - COMPLETE WITH LEARNING TABLES
# Your existing database + Committee Learning System tables

import sqlite3
from pathlib import Path
from datetime import datetime
import json

# Database path - Using your existing database with 88 trades
DB_PATH = Path.home() / "trading_ai_learning.db"  # Your old database
# If your database has a different name, update it here

def init_database():
    """Initialize database with all tables including learning system"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # ==================== YOUR EXISTING TABLES ====================
    
    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            asset_type TEXT,
            pair TEXT,
            timeframe TEXT,
            current_price REAL,
            predicted_price REAL,
            prediction_horizon INTEGER,
            confidence REAL,
            signal_strength REAL,
            features TEXT,
            status TEXT DEFAULT 'analysis_only',
            actual_entry_price REAL,
            entry_timestamp DATETIME,
            indicator_snapshot TEXT,
            position_type TEXT,
            target_price REAL,
            stop_loss REAL,
            committee_position TEXT,
            committee_confidence REAL,
            committee_reasoning TEXT
        )
    """)
    
    # Trade results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            trade_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_loss_pct REAL,
            prediction_error REAL,
            notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)
    
    # Indicator accuracy table (your existing learning)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS indicator_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name TEXT UNIQUE,
            correct_count INTEGER DEFAULT 0,
            wrong_count INTEGER DEFAULT 0,
            accuracy_rate REAL DEFAULT 0.5,
            weight_multiplier REAL DEFAULT 1.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ==================== 🆕 COMMITTEE LEARNING TABLES ====================
    
    # Consultant Performance table
    cursor.execute("""
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Signal Performance table
    cursor.execute("""
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
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            market_conditions TEXT,
            UNIQUE(consultant_name, signal_name)
        )
    """)
    
    # Committee Decisions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS committee_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
            outcome_determined_at DATETIME,
            price_at_outcome REAL,
            profit_loss_pct REAL,
            was_correct INTEGER,
            hours_to_outcome REAL,
            
            FOREIGN KEY (trade_id) REFERENCES predictions(id)
        )
    """)
    
    # Committee Meeting Logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS committee_meeting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            meeting_summary TEXT,
            had_conflict INTEGER DEFAULT 0,
            conflict_description TEXT,
            agreement_matrix TEXT,
            volatility REAL,
            trend_strength REAL,
            volume_level TEXT,
            FOREIGN KEY (decision_id) REFERENCES committee_decisions(id)
        )
    """)
    
    # Signal Correlations table
    cursor.execute("""
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
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ==================== INITIALIZE CONSULTANTS ====================
    
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
    
    conn.commit()
    conn.close()
    
    print("✅ Database initialized with all tables including learning system!")


# ==================== YOUR EXISTING FUNCTIONS ====================

def save_prediction(asset_type, pair, timeframe, current_price, predicted_price, 
                   prediction_horizon, confidence, signal_strength, features,
                   status='analysis_only', actual_entry_price=None, entry_timestamp=None,
                   indicator_snapshot=None, position_type=None, target_price=None,
                   stop_loss=None, committee_position=None, committee_confidence=None,
                   committee_reasoning=None):
    """Save prediction to database"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (
            asset_type, pair, timeframe, current_price, predicted_price,
            prediction_horizon, confidence, signal_strength, features, status,
            actual_entry_price, entry_timestamp, indicator_snapshot,
            position_type, target_price, stop_loss,
            committee_position, committee_confidence, committee_reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        asset_type, pair, timeframe, current_price, predicted_price,
        prediction_horizon, confidence, signal_strength, features, status,
        actual_entry_price, entry_timestamp, indicator_snapshot,
        position_type, target_price, stop_loss,
        committee_position, committee_confidence, committee_reasoning
    ))
    
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return pred_id


def mark_prediction_for_trading(prediction_id, actual_entry_price, entry_timestamp,
                                position_type, target_price, stop_loss):
    """Mark a prediction as an active trade"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE predictions
        SET status = 'will_trade',
            actual_entry_price = ?,
            entry_timestamp = ?,
            position_type = ?,
            target_price = ?,
            stop_loss = ?
        WHERE id = ?
    """, (actual_entry_price, entry_timestamp, position_type, target_price, 
          stop_loss, prediction_id))
    
    conn.commit()
    conn.close()


def save_trade_result(prediction_id, entry_price, exit_price, profit_loss,
                     profit_loss_pct, prediction_error, notes=None):
    """Save completed trade result"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO trade_results (
            prediction_id, entry_price, exit_price, profit_loss,
            profit_loss_pct, prediction_error, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (prediction_id, entry_price, exit_price, profit_loss,
          profit_loss_pct, prediction_error, notes))
    
    # Mark prediction as completed
    cursor.execute("""
        UPDATE predictions
        SET status = 'completed'
        WHERE id = ?
    """, (prediction_id,))
    
    conn.commit()
    conn.close()
    
    return True


def get_all_recent_predictions(limit=50):
    """Get recent predictions"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id, timestamp, asset_type, pair, timeframe, current_price,
                predicted_price, prediction_horizon, confidence, signal_strength,
                status, actual_entry_price, position_type, target_price, stop_loss,
                committee_position, committee_confidence
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows, columns=[
                'id', 'timestamp', 'asset_type', 'pair', 'timeframe', 'current_price',
                'predicted_price', 'prediction_horizon', 'confidence', 'signal_strength',
                'status', 'actual_entry_price', 'position_type', 'target_price', 'stop_loss',
                'committee_position', 'committee_confidence'
            ])
            return df
        
        return None
    
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return None


def get_indicator_weights():
    """Get learned indicator weights"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT indicator_name, weight_multiplier, accuracy_rate
        FROM indicator_accuracy
        WHERE correct_count + wrong_count > 0
    """)
    
    weights = {}
    for row in cursor.fetchall():
        indicator_name, weight, accuracy = row
        weights[indicator_name] = {
            'weight': weight,
            'accuracy': accuracy
        }
    
    conn.close()
    return weights


def relearn_from_past_trades():
    """Analyze past trades to update indicator weights"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get all closed trades
    cursor.execute("""
        SELECT 
            tr.profit_loss,
            p.indicator_snapshot
        FROM trade_results tr
        JOIN predictions p ON tr.prediction_id = p.id
        WHERE p.indicator_snapshot IS NOT NULL
    """)
    
    trades = cursor.fetchall()
    
    if not trades:
        conn.close()
        return 0
    
    # Initialize indicator tracking
    indicator_stats = {}
    
    for profit_loss, snapshot_str in trades:
        if not snapshot_str:
            continue
        
        try:
            # Parse indicator snapshot
            snapshot = eval(snapshot_str) if snapshot_str else {}
            
            # Determine if trade was successful
            was_win = profit_loss > 0
            
            # Track each indicator
            for indicator_name, indicator_value in snapshot.items():
                if indicator_name not in indicator_stats:
                    indicator_stats[indicator_name] = {
                        'correct': 0,
                        'wrong': 0
                    }
                
                if was_win:
                    indicator_stats[indicator_name]['correct'] += 1
                else:
                    indicator_stats[indicator_name]['wrong'] += 1
        
        except:
            continue
    
    # Update database
    for indicator_name, stats in indicator_stats.items():
        correct = stats['correct']
        wrong = stats['wrong']
        total = correct + wrong
        
        if total > 0:
            accuracy = correct / total
            
            # Calculate weight (0.5x to 2.0x based on accuracy)
            if accuracy >= 0.7:
                weight = 1.5
            elif accuracy >= 0.6:
                weight = 1.2
            elif accuracy >= 0.5:
                weight = 1.0
            elif accuracy >= 0.4:
                weight = 0.8
            else:
                weight = 0.5
            
            cursor.execute("""
                INSERT OR REPLACE INTO indicator_accuracy 
                (indicator_name, correct_count, wrong_count, accuracy_rate, weight_multiplier, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (indicator_name, correct, wrong, accuracy, weight))
    
    conn.commit()
    conn.close()
    
    return len(trades)


# ==================== INITIALIZE ON IMPORT ====================
if __name__ == "__main__":
    init_database()
