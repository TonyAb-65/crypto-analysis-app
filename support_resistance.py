"""
Support & Resistance Module - S/R zone detection and price target calculation
"""
import numpy as np
import pandas as pd


def find_support_resistance_zones(df, lookback=100):
    """
    Find support and resistance zones using pivot points and clustering
    Returns: {'support': [...], 'resistance': [...]}
    """
    if df is None or len(df) < 20:
        return {'support': [], 'resistance': []}
    
    df_recent = df.tail(lookback).copy()
    
    resistance_levels = []
    support_levels = []
    
    # Find pivot highs (potential resistance)
    for i in range(2, len(df_recent) - 2):
        if (df_recent['high'].iloc[i] > df_recent['high'].iloc[i-1] and
            df_recent['high'].iloc[i] > df_recent['high'].iloc[i-2] and
            df_recent['high'].iloc[i] > df_recent['high'].iloc[i+1] and
            df_recent['high'].iloc[i] > df_recent['high'].iloc[i+2]):
            
            resistance_levels.append(df_recent['high'].iloc[i])
    
    # Find pivot lows (potential support)
    for i in range(2, len(df_recent) - 2):
        if (df_recent['low'].iloc[i] < df_recent['low'].iloc[i-1] and
            df_recent['low'].iloc[i] < df_recent['low'].iloc[i-2] and
            df_recent['low'].iloc[i] < df_recent['low'].iloc[i+1] and
            df_recent['low'].iloc[i] < df_recent['low'].iloc[i+2]):
            
            support_levels.append(df_recent['low'].iloc[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, threshold=0.02):
        if len(levels) == 0:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                avg_level = np.mean(current_cluster)
                touches = len(current_cluster)
                strength = 'STRONG' if touches >= 3 else 'MEDIUM'
                clustered.append({'price': avg_level, 'touches': touches, 'strength': strength})
                current_cluster = [level]
        
        if current_cluster:
            avg_level = np.mean(current_cluster)
            touches = len(current_cluster)
            strength = 'STRONG' if touches >= 3 else 'MEDIUM'
            clustered.append({'price': avg_level, 'touches': touches, 'strength': strength})
        
        return clustered
    
    support_zones = cluster_levels(support_levels)
    resistance_zones = cluster_levels(resistance_levels)
    
    # Sort by strength
    support_zones = sorted(support_zones, key=lambda x: x['touches'], reverse=True)
    resistance_zones = sorted(resistance_zones, key=lambda x: x['touches'], reverse=True)
    
    return {
        'support': support_zones,
        'resistance': resistance_zones
    }


def check_at_key_level(current_price, sr_zones, threshold=0.005):
    """
    Check if current price is AT a key support or resistance level
    Returns: (at_level, level_type, level_info)
    """
    # Check resistance levels
    for r in sr_zones['resistance']:
        distance_pct = abs(current_price - r['price']) / current_price
        if distance_pct < threshold:
            return True, 'RESISTANCE', r
    
    # Check support levels
    for s in sr_zones['support']:
        distance_pct = abs(current_price - s['price']) / current_price
        if distance_pct < threshold:
            return True, 'SUPPORT', s
    
    return False, None, None


def get_price_targets_based_on_sr(current_price, sr_zones):
    """
    Get next support and resistance targets based on current price
    Returns: {'next_resistance': {...}, 'next_support': {...}}
    """
    next_resistance = None
    next_support = None
    
    # Find nearest resistance above current price
    for r in sr_zones['resistance']:
        if r['price'] > current_price:
            if next_resistance is None or r['price'] < next_resistance['price']:
                next_resistance = r
    
    # Find nearest support below current price
    for s in sr_zones['support']:
        if s['price'] < current_price:
            if next_support is None or s['price'] > next_support['price']:
                next_support = s
    
    return {
        'next_resistance': next_resistance,
        'next_support': next_support
    }


def check_support_resistance_barriers(df, predicted_price, current_price):
    """Check if predicted price needs to break through major S/R levels"""
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    
    recent_highs = df['high'].tail(50).nlargest(5).mean()
    recent_lows = df['low'].tail(50).nsmallest(5).mean()
    
    barriers = []
    
    if current_price < predicted_price:
        if predicted_price > high_20:
            barriers.append(('resistance', high_20, abs(predicted_price - high_20)))
        if predicted_price > recent_highs:
            barriers.append(('strong_resistance', recent_highs, abs(predicted_price - recent_highs)))
    else:
        if predicted_price < low_20:
            barriers.append(('support', low_20, abs(predicted_price - low_20)))
        if predicted_price < recent_lows:
            barriers.append(('strong_support', recent_lows, abs(predicted_price - recent_lows)))
    
    return barriers


def analyze_timeframe_volatility(df, predicted_change_pct, timeframe_hours):
    """Check if predicted change is realistic for the timeframe"""
    recent_changes = df['close'].pct_change().tail(50)
    
    avg_hourly_change = abs(recent_changes).mean() * 100
    max_hourly_change = abs(recent_changes).max() * 100
    
    predicted_hourly_rate = abs(predicted_change_pct) / timeframe_hours
    
    is_realistic = predicted_hourly_rate <= (avg_hourly_change * 2)
    
    volatility_context = {
        'avg_hourly_change': avg_hourly_change,
        'max_hourly_change': max_hourly_change,
        'predicted_hourly_rate': predicted_hourly_rate,
        'is_realistic': is_realistic
    }
    
    return volatility_context


def adjust_confidence_for_barriers(base_confidence, barriers, volatility_context):
    """Adjust AI confidence based on barriers and volatility"""
    adjusted_confidence = base_confidence
    
    for barrier_type, price_level, distance in barriers:
        if barrier_type == 'strong_resistance' or barrier_type == 'strong_support':
            adjusted_confidence *= 0.7
        else:
            adjusted_confidence *= 0.85
    
    if not volatility_context['is_realistic']:
        adjusted_confidence *= 0.6
    
    adjusted_confidence = max(adjusted_confidence, 30.0)
    adjusted_confidence = min(adjusted_confidence, 95.0)
    
    return adjusted_confidence
