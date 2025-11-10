"""
Active Trade Monitor - Real-time monitoring for open positions
Analyzes trend and provides HOLD/EXIT recommendations every 15 minutes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def get_active_trades_for_monitoring():
    """
    Get all active trades that need monitoring
    Returns list of trades with status = 'will_trade' and not completed
    """
    import sqlite3
    from pathlib import Path
    
    DB_PATH = Path.home() / "ai_trading.db"
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        
        query = """
        SELECT 
            p.id,
            p.pair as symbol,
            p.position_type,
            p.actual_entry_price as entry_price,
            p.target_price,
            p.stop_loss,
            p.current_price as entry_current_price,
            p.timestamp as entry_time,
            p.status
        FROM predictions p
        WHERE p.status = 'will_trade'
        AND p.actual_entry_price IS NOT NULL
        ORDER BY p.timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df if len(df) > 0 else None
        
    except Exception as e:
        print(f"Error fetching active trades: {e}")
        return None


def calculate_trade_progress(entry_price, current_price, target_price, position_type):
    """
    Calculate progress toward target as percentage
    """
    try:
        if position_type == 'LONG':
            total_move = target_price - entry_price
            current_move = current_price - entry_price
        else:  # SHORT
            total_move = entry_price - target_price
            current_move = entry_price - current_price
        
        if total_move == 0:
            return 0
        
        progress = (current_move / total_move) * 100
        return max(0, min(progress, 200))  # Cap at 200% (allow overshoot)
        
    except:
        return 0


def calculate_profit_pct(entry_price, current_price, position_type):
    """
    Calculate current profit/loss percentage
    """
    try:
        if position_type == 'LONG':
            return ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            return ((entry_price - current_price) / entry_price) * 100
    except:
        return 0


def quick_trend_check(df):
    """
    Quick 3-indicator trend check (from our agreed 80% exit strategy)
    Returns: (bullish_count, bearish_count)
    """
    if df is None or len(df) < 20:
        return 0, 0
    
    latest = df.iloc[-1]
    rsi = latest.get('rsi', 50)
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    close = latest.get('close', 0)
    sma_20 = latest.get('sma_20', close)
    
    bullish = 0
    bearish = 0
    
    # RSI
    if rsi < 40:
        bullish += 1
    elif rsi > 60:
        bearish += 1
    
    # MACD
    if macd > macd_signal:
        bullish += 1
    else:
        bearish += 1
    
    # Price vs SMA
    if close > sma_20:
        bullish += 1
    else:
        bearish += 1
    
    return bullish, bearish


def get_exit_recommendation(progress_pct, position_type, df):
    """
    Determine HOLD or EXIT based on our agreed 80% strategy
    
    Rules:
    - Progress >= 80% → EXIT (secure profit)
    - Progress 50-79% → Quick trend check
    - Progress < 50% → Check for reversal
    """
    
    # Rule 1: 80%+ of target → EXIT
    if progress_pct >= 80:
        return "EXIT", "🎯 80%+ target reached - Secure profit!", "success"
    
    # Rule 2: 50-79% → Trend check
    if 50 <= progress_pct < 80:
        bullish, bearish = quick_trend_check(df)
        
        if position_type == 'LONG':
            if bullish > bearish:
                return "HOLD", f"💪 {int(progress_pct)}% progress, trend strong ({bullish} vs {bearish})", "info"
            else:
                return "EXIT", f"⚠️ {int(progress_pct)}% progress, trend weakening ({bullish} vs {bearish})", "warning"
        else:  # SHORT
            if bearish > bullish:
                return "HOLD", f"💪 {int(progress_pct)}% progress, trend strong ({bearish} vs {bullish})", "info"
            else:
                return "EXIT", f"⚠️ {int(progress_pct)}% progress, trend weakening ({bearish} vs {bullish})", "warning"
    
    # Rule 3: <50% → Check for reversal
    if progress_pct < 50:
        bullish, bearish = quick_trend_check(df)
        
        if position_type == 'LONG':
            if bearish >= 2:  # Majority bearish = reversal
                return "EXIT", f"🔴 Trend reversed! Only {int(progress_pct)}% progress ({bullish} vs {bearish})", "error"
            else:
                return "HOLD", f"⏳ {int(progress_pct)}% progress, trend intact ({bullish} vs {bearish})", "info"
        else:  # SHORT
            if bullish >= 2:  # Majority bullish = reversal
                return "EXIT", f"🔴 Trend reversed! Only {int(progress_pct)}% progress ({bearish} vs {bullish})", "error"
            else:
                return "HOLD", f"⏳ {int(progress_pct)}% progress, trend intact ({bearish} vs {bullish})", "info"
    
    return "HOLD", "Monitoring...", "info"


def get_next_check_countdown():
    """
    Calculate seconds until next 15-minute check
    Returns remaining seconds
    """
    now = datetime.now()
    current_minute = now.minute
    
    # Next check is at 0, 15, 30, or 45 minutes
    next_check_minute = ((current_minute // 15) + 1) * 15
    if next_check_minute >= 60:
        next_check_minute = 0
        next_check_time = now.replace(hour=(now.hour + 1) % 24, minute=0, second=0, microsecond=0)
    else:
        next_check_time = now.replace(minute=next_check_minute, second=0, microsecond=0)
    
    remaining_seconds = (next_check_time - now).total_seconds()
    return int(remaining_seconds)


def format_countdown(seconds):
    """
    Format seconds into readable countdown
    """
    if seconds < 60:
        return f"{seconds}s"
    else:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
