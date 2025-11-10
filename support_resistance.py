"""
Support & Resistance Module - PROFESSIONAL TRADER APPROACH
Uses REAL trading methodology - not academic algorithms
"""
import numpy as np
import pandas as pd


def find_support_resistance_zones(df, lookback=100):
    """
    Professional S/R detection based on REAL trading methodology
    
    How Real Traders Find S/R:
    1. Recent swing highs/lows (20+ candle window)
    2. Multiple touches at same price level
    3. Clear visual rejection zones
    4. Price must have REACTED at these levels
    """
    
    if df is None or len(df) < 50:
        return {'support': [], 'resistance': []}
    
    # Use reasonable lookback
    lookback = min(lookback, len(df))
    df_recent = df.tail(lookback).copy()
    current_price = df_recent['close'].iloc[-1]
    
    # ==================== STEP 1: Find Swing Points ====================
    # Use LARGER window (20 candles) - real traders don't look at every tiny wiggle
    window = 20
    
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df_recent) - window):
        # Swing High: Highest point in 20-candle window on both sides
        if df_recent['high'].iloc[i] == df_recent['high'].iloc[i-window:i+window+1].max():
            swing_highs.append(df_recent['high'].iloc[i])
        
        # Swing Low: Lowest point in 20-candle window on both sides
        if df_recent['low'].iloc[i] == df_recent['low'].iloc[i-window:i+window+1].min():
            swing_lows.append(df_recent['low'].iloc[i])
    
    # ==================== STEP 2: Cluster Nearby Levels ====================
    # Group levels within 1.5% - real support/resistance are zones, not exact prices
    def cluster_levels(levels, tolerance=0.015):
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # If within 1.5% of cluster, add to it
            if level <= np.mean(current_cluster) * (1 + tolerance):
                current_cluster.append(level)
            else:
                # Finalize cluster as average
                if len(current_cluster) >= 1:  # At least 1 touch
                    clusters.append({
                        'price': np.mean(current_cluster),
                        'touches': len(current_cluster)
                    })
                current_cluster = [level]
        
        # Don't forget last cluster
        if len(current_cluster) >= 1:
            clusters.append({
                'price': np.mean(current_cluster),
                'touches': len(current_cluster)
            })
        
        return clusters
    
    resistance_clusters = cluster_levels(swing_highs)
    support_clusters = cluster_levels(swing_lows)
    
    # ==================== STEP 3: Filter by Direction ====================
    # RESISTANCE must be ABOVE current price
    # SUPPORT must be BELOW current price
    
    valid_resistance = []
    for cluster in resistance_clusters:
        level = cluster['price']
        touches = cluster['touches']
        
        # Must be above current price
        if level > current_price * 1.002:  # At least 0.2% above
            # Must be within reasonable range (not too far away)
            if level < current_price * 1.20:  # Within 20%
                strength = 'STRONG' if touches >= 3 else 'MEDIUM' if touches >= 2 else 'WEAK'
                valid_resistance.append({
                    'price': level,
                    'touches': touches,
                    'strength': strength,
                    'status': 'INTACT'
                })
    
    valid_support = []
    for cluster in support_clusters:
        level = cluster['price']
        touches = cluster['touches']
        
        # Must be below current price
        if level < current_price * 0.998:  # At least 0.2% below
            # Must be within reasonable range
            if level > current_price * 0.80:  # Within 20%
                strength = 'STRONG' if touches >= 3 else 'MEDIUM' if touches >= 2 else 'WEAK'
                valid_support.append({
                    'price': level,
                    'touches': touches,
                    'strength': strength,
                    'status': 'INTACT'
                })
    
    # ==================== STEP 4: Sort by Proximity ====================
    # Nearest levels are most important
    valid_resistance.sort(key=lambda x: x['price'])  # Closest resistance first
    valid_support.sort(key=lambda x: x['price'], reverse=True)  # Closest support first
    
    # Return top 5 of each
    return {
        'resistance': valid_resistance[:5],
        'support': valid_support[:5]
    }


def check_at_key_level(current_price, sr_zones, threshold=0.01):
    """
    Check if price is AT a key S/R level (within 1%)
    """
    for r in sr_zones.get('resistance', []):
        if abs(current_price - r['price']) / current_price < threshold:
            return True, 'RESISTANCE', r
    
    for s in sr_zones.get('support', []):
        if abs(current_price - s['price']) / current_price < threshold:
            return True, 'SUPPORT', s
    
    return False, None, None


def get_price_targets_based_on_sr(current_price, sr_zones):
    """
    Get next S/R levels for targets
    """
    supports = sr_zones.get('support', [])
    resistances = sr_zones.get('resistance', [])
    
    # Next resistance (first one above price)
    next_resistance = resistances[0] if resistances else None
    
    # Next support (first one below price)
    next_support = supports[0] if supports else None
    
    return {
        'next_resistance': next_resistance,
        'next_support': next_support,
        'nearest_support': next_support,
        'nearest_resistance': next_resistance
    }


def calculate_risk_reward(entry_price, target_price, stop_loss):
    """Calculate risk-reward ratio"""
    if entry_price == stop_loss:
        return 0
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    
    return reward / risk if risk > 0 else 0


def check_support_resistance_barriers(df, predicted_price, current_price):
    """Check if predicted price needs to break through major S/R"""
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    
    barriers = []
    
    if current_price < predicted_price:
        if predicted_price > high_20:
            barriers.append(('resistance', high_20, abs(predicted_price - high_20)))
    else:
        if predicted_price < low_20:
            barriers.append(('support', low_20, abs(predicted_price - low_20)))
    
    return barriers


def analyze_timeframe_volatility(df, predicted_change_pct, timeframe_hours):
    """Check if predicted change is realistic"""
    recent_changes = df['close'].pct_change().tail(50)
    
    avg_hourly_change = abs(recent_changes).mean() * 100
    max_hourly_change = abs(recent_changes).max() * 100
    
    predicted_hourly_rate = abs(predicted_change_pct) / timeframe_hours
    
    is_realistic = predicted_hourly_rate <= (avg_hourly_change * 2)
    
    return {
        'avg_hourly_change': avg_hourly_change,
        'max_hourly_change': max_hourly_change,
        'predicted_hourly_rate': predicted_hourly_rate,
        'is_realistic': is_realistic
    }


def adjust_confidence_for_barriers(base_confidence, barriers, volatility_context):
    """Adjust confidence based on barriers"""
    adjusted_confidence = base_confidence
    
    for barrier_type, price_level, distance in barriers:
        if barrier_type == 'strong_resistance' or barrier_type == 'strong_support':
            adjusted_confidence *= 0.7
        else:
            adjusted_confidence *= 0.85
    
    if not volatility_context['is_realistic']:
        adjusted_confidence *= 0.6
    
    return max(30.0, min(95.0, adjusted_confidence))
