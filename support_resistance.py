"""
Support & Resistance Module - DYNAMIC S/R with role reversal
Based on original working code with full dynamic behavior
"""
import numpy as np
import pandas as pd


def find_support_resistance_zones(df, lookback=100):
    """
    Find S/R levels where price reversed multiple times
    DYNAMIC: Updates when levels are broken (support becomes resistance and vice versa)
    """
    
    if df is None or len(df) < 20:
        return {'support': [], 'resistance': []}
    
    if len(df) < lookback:
        lookback = len(df)
    
    current_price = df['close'].iloc[-1]
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    # Find local peaks (resistance candidates)
    resistance_zones = []
    for i in range(5, len(highs) - 5):
        if highs.iloc[i] == highs.iloc[i-5:i+5].max():
            resistance_zones.append(highs.iloc[i])
    
    # Find local bottoms (support candidates)
    support_zones = []
    for i in range(5, len(lows) - 5):
        if lows.iloc[i] == lows.iloc[i-5:i+5].min():
            support_zones.append(lows.iloc[i])
    
    # Group nearby levels (within 2% = same zone)
    def cluster_levels(levels, tolerance=0.02):
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level <= current_cluster[-1] * (1 + tolerance):
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    # Get strongest S/R levels (price tested multiple times)
    strong_resistance = cluster_levels(resistance_zones)
    strong_support = cluster_levels(support_zones)
    
    # Count how many times each level was tested
    def count_touches(price_level, df_subset, tolerance=0.02):
        touches = 0
        for i in range(len(df_subset)):
            high = df_subset['high'].iloc[i]
            low = df_subset['low'].iloc[i]
            if (abs(high - price_level) / price_level < tolerance or 
                abs(low - price_level) / price_level < tolerance):
                touches += 1
        return touches
    
    # Check if level was broken (role reversal)
    def is_level_broken(price_level, df, current_price):
        """
        Detect if price broke through a level
        - If price is now ABOVE old resistance → resistance became support (FLIPPED)
        - If price is now BELOW old support → support became resistance (FLIPPED)
        """
        # Check recent price action (last 20 candles)
        recent_lows = df['low'].tail(20)
        recent_highs = df['high'].tail(20)
        
        # Level was support, but price broke below it
        if current_price < price_level * 0.98:  # 2% below
            if any(recent_lows > price_level * 0.98):  # Was above recently
                return 'SUPPORT_BROKEN'
        
        # Level was resistance, but price broke above it
        if current_price > price_level * 1.02:  # 2% above
            if any(recent_highs < price_level * 1.02):  # Was below recently
                return 'RESISTANCE_BROKEN'
        
        return 'INTACT'
    
    # Build resistance list with role reversal logic
    resistance_strength = []
    for level in strong_resistance:
        touches = count_touches(level, df.tail(lookback))
        status = is_level_broken(level, df, current_price)
        
        # If resistance was broken, it becomes support (skip here, add to support later)
        if status == 'RESISTANCE_BROKEN':
            continue
        
        if touches >= 2:
            resistance_strength.append({
                'price': level,
                'touches': touches,
                'strength': 'STRONG' if touches >= 3 else 'MEDIUM',
                'status': status
            })
    
    # Build support list with role reversal logic
    support_strength = []
    for level in strong_support:
        touches = count_touches(level, df.tail(lookback))
        status = is_level_broken(level, df, current_price)
        
        # If support was broken, it becomes resistance (skip here, add to resistance later)
        if status == 'SUPPORT_BROKEN':
            continue
        
        if touches >= 2:
            support_strength.append({
                'price': level,
                'touches': touches,
                'strength': 'STRONG' if touches >= 3 else 'MEDIUM',
                'status': status
            })
    
    # Add broken levels to opposite list (role reversal)
    # Broken resistance becomes support
    for level in strong_resistance:
        status = is_level_broken(level, df, current_price)
        if status == 'RESISTANCE_BROKEN':
            touches = count_touches(level, df.tail(lookback))
            if touches >= 2:
                support_strength.append({
                    'price': level,
                    'touches': touches,
                    'strength': 'FLIPPED',  # Was resistance, now support
                    'status': 'FLIPPED'
                })
    
    # Broken support becomes resistance
    for level in strong_support:
        status = is_level_broken(level, df, current_price)
        if status == 'SUPPORT_BROKEN':
            touches = count_touches(level, df.tail(lookback))
            if touches >= 2:
                resistance_strength.append({
                    'price': level,
                    'touches': touches,
                    'strength': 'FLIPPED',  # Was support, now resistance
                    'status': 'FLIPPED'
                })
    
    # Filter to only show RELEVANT levels (within 10% of current price)
    relevant_resistance = [r for r in resistance_strength if r['price'] < current_price * 1.10]
    relevant_support = [s for s in support_strength if s['price'] > current_price * 0.90]
    
    # Sort by price (resistance descending, support descending)
    relevant_resistance.sort(key=lambda x: x['price'], reverse=True)
    relevant_support.sort(key=lambda x: x['price'], reverse=True)
    
    return {
        'resistance': relevant_resistance,
        'support': relevant_support
    }


def check_at_key_level(current_price, sr_zones, threshold=0.005):
    """
    Check if current price is AT a key support or resistance level
    Returns: (at_level, level_type, level_info)
    """
    # Check resistance levels
    for r in sr_zones.get('resistance', []):
        distance_pct = abs(current_price - r['price']) / current_price
        if distance_pct < threshold:
            return True, 'RESISTANCE', r
    
    # Check support levels
    for s in sr_zones.get('support', []):
        distance_pct = abs(current_price - s['price']) / current_price
        if distance_pct < threshold:
            return True, 'SUPPORT', s
    
    return False, None, None


def get_price_targets_based_on_sr(current_price, sr_zones):
    """
    Trader logic: If price breaks level, next target is the next level
    Returns nearest S/R levels and price targets
    """
    supports = sr_zones.get('support', [])
    resistances = sr_zones.get('resistance', [])
    
    # Find nearest support (below current price)
    nearest_support = None
    for s in supports:
        if s['price'] < current_price:
            nearest_support = s
            break
    
    # Find nearest resistance (above current price)
    nearest_resistance = None
    for r in resistances:
        if r['price'] > current_price:
            nearest_resistance = r
            break
    
    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance
    }


def calculate_risk_reward(entry_price, target_price, stop_loss):
    """Calculate risk-reward ratio"""
    if entry_price == stop_loss:
        return 0
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    
    return reward / risk if risk > 0 else 0
