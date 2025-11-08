"""
Utility Functions - Helper functions for the trading system
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd
from database import DB_PATH


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


def analyze_indicator_accuracy(prediction_id):
    """Analyze which indicators were accurate for this trade"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.indicator_snapshot, t.profit_loss
            FROM predictions p
            JOIN trade_results t ON p.id = t.prediction_id
            WHERE p.id = ?
        ''', (prediction_id,))
        
        result = cursor.fetchone()
        if not result or not result[0]:
            conn.close()
            return
        
        import json
        snapshot = json.loads(result[0])
        was_profitable = result[1] > 0
        
        for indicator_name, indicator_data in snapshot.items():
            signal = indicator_data.get('signal', 'neutral')
            
            if signal == 'bullish' and was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET correct_count = correct_count + 1, last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bearish' and not was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET correct_count = correct_count + 1, last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bullish' and not was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET wrong_count = wrong_count + 1, last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
                
            elif signal == 'bearish' and was_profitable:
                cursor.execute('''
                    UPDATE indicator_accuracy 
                    SET wrong_count = wrong_count + 1, last_updated = ?
                    WHERE indicator_name = ?
                ''', (datetime.now().isoformat(), indicator_name))
        
        cursor.execute("SELECT indicator_name, correct_count, wrong_count FROM indicator_accuracy")
        for row in cursor.fetchall():
            indicator_name, correct, wrong = row
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
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing indicator accuracy: {e}")


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
        conn = sqlite3.connect(str(DB_PATH))
        query = '''
            SELECT 
                p.timestamp, p.asset_type, p.pair, p.timeframe,
                p.predicted_price, t.entry_price, t.exit_price,
                t.profit_loss, t.profit_loss_pct, t.trade_date
            FROM predictions p
            JOIN trade_results t ON p.id = t.prediction_id
            ORDER BY t.trade_date DESC
            LIMIT 1000
        '''
        trades = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(trades) > 0:
            csv_dir = Path.home() / 'Downloads'
            csv_dir.mkdir(exist_ok=True)
            csv_path = csv_dir / f'trades_export_{datetime.now():%Y%m%d_%H%M%S}.csv'
            trades.to_csv(csv_path, index=False)
            return csv_path
        return None
    except Exception as e:
        return None
