#!/usr/bin/env python3
"""
Surgical Fix: Correct Trade ID 411 (and other SHORT trades saved as LONG)
This script fixes trades that were incorrectly saved as LONG when they should be SHORT.
"""

import sqlite3
from pathlib import Path

# Database path
HOME = Path.home()
DB_PATH = HOME / 'trading_ai_learning.db'

def fix_trade(trade_id):
    """Fix a specific trade by correcting its position type and recalculating P/L"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get trade info
    cursor.execute('''
        SELECT p.id, p.signal_strength, p.position_type, 
               t.entry_price, t.exit_price, t.profit_loss, t.profit_loss_pct
        FROM predictions p
        JOIN trade_results t ON p.id = t.prediction_id
        WHERE p.id = ?
    ''', (trade_id,))
    
    result = cursor.fetchone()
    
    if not result:
        print(f"❌ Trade ID {trade_id} not found or not completed")
        conn.close()
        return False
    
    trade_id, signal_strength, position_type, entry_price, exit_price, old_pl, old_pl_pct = result
    
    print(f"\n📊 Trade ID {trade_id}:")
    print(f"   Signal Strength: {signal_strength}")
    print(f"   Current Position Type: {position_type}")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Exit: ${exit_price:,.2f}")
    print(f"   OLD P/L: ${old_pl:,.2f} ({old_pl_pct:+.2f}%)")
    
    # Determine correct position type based on signal
    correct_position_type = 'SHORT' if signal_strength < 0 else 'LONG'
    
    if position_type == correct_position_type:
        print(f"✅ Position type is already correct: {position_type}")
        conn.close()
        return True
    
    print(f"\n⚠️  Needs correction: {position_type} → {correct_position_type}")
    
    # Recalculate P/L correctly
    if correct_position_type == 'SHORT':
        # SHORT: profit when price goes DOWN
        new_pl = entry_price - exit_price
    else:
        # LONG: profit when price goes UP
        new_pl = exit_price - entry_price
    
    new_pl_pct = (new_pl / entry_price) * 100
    
    print(f"   NEW P/L: ${new_pl:,.2f} ({new_pl_pct:+.2f}%)")
    
    # Update position type in predictions table
    cursor.execute('''
        UPDATE predictions 
        SET position_type = ?
        WHERE id = ?
    ''', (correct_position_type, trade_id))
    
    # Update P/L in trade_results table
    cursor.execute('''
        UPDATE trade_results
        SET profit_loss = ?,
            profit_loss_pct = ?
        WHERE prediction_id = ?
    ''', (new_pl, new_pl_pct, trade_id))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Trade ID {trade_id} corrected!")
    return True

def fix_all_trades():
    """Fix all trades where position_type doesn't match signal_strength"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Find all completed trades that need fixing
    cursor.execute('''
        SELECT p.id, p.signal_strength, p.position_type
        FROM predictions p
        JOIN trade_results t ON p.id = t.prediction_id
        WHERE (p.signal_strength < 0 AND (p.position_type IS NULL OR p.position_type = 'LONG'))
           OR (p.signal_strength >= 0 AND p.position_type = 'SHORT')
    ''')
    
    trades_to_fix = cursor.fetchall()
    conn.close()
    
    if not trades_to_fix:
        print("✅ All trades have correct position types!")
        return
    
    print(f"\n🔧 Found {len(trades_to_fix)} trade(s) that need correction:\n")
    
    for trade_id, signal, pos_type in trades_to_fix:
        print(f"Trade ID {trade_id}: signal={signal}, stored as {pos_type}")
    
    print("\n" + "="*60)
    response = input("Fix all these trades? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        for trade_id, signal, pos_type in trades_to_fix:
            fix_trade(trade_id)
        print(f"\n✅ Fixed {len(trades_to_fix)} trade(s)!")
    else:
        print("❌ Operation cancelled")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 SURGICAL FIX: Correct SHORT/LONG Position Types")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Fix specific trade (e.g., Trade ID 411)")
    print("2. Fix all incorrect trades automatically")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        trade_id = input("Enter Trade ID: ")
        try:
            fix_trade(int(trade_id))
        except ValueError:
            print("❌ Invalid trade ID")
    elif choice == "2":
        fix_all_trades()
    else:
        print("👋 Exiting")
