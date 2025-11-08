"""
Consultants Module - Committee-based trading decisions
FIXED: Updated fetch_data calls to use timeframe_config
"""
import pandas as pd
import numpy as np
from data_api import fetch_data
from indicators import calculate_technical_indicators


def fetch_data_for_timeframe(symbol_param, asset_type_param, timeframe_hours):
    """
    Fetch data for specific timeframe (1h, 4h, 24h)
    FIXED: Now creates proper timeframe_config dict for fetch_data
    """
    # Create timeframe_config based on hours
    if timeframe_hours == 1:
        timeframe_config = {
            'binance': '1h',
            'okx': '1H',
            'limit': 100
        }
    elif timeframe_hours == 4:
        timeframe_config = {
            'binance': '4h',
            'okx': '4H',
            'limit': 100
        }
    elif timeframe_hours == 24:
        timeframe_config = {
            'binance': '1d',
            'okx': '1D',
            'limit': 100
        }
    else:
        timeframe_config = {
            'binance': '1h',
            'okx': '1H',
            'limit': 100
        }
    
    # Call fetch_data with proper config dict
    df, source = fetch_data(symbol_param, asset_type_param, timeframe_config)
    
    if df is not None and len(df) > 0:
        df = calculate_technical_indicators(df)
    
    return df, source


def analyze_multi_timeframe(df_1h, df_4h, df_1d):
    """Analyze multiple timeframes for trend alignment"""
    trends = {}
    
    # 1H trend
    if df_1h is not None and len(df_1h) >= 50:
        if 'sma_20' in df_1h.columns and 'sma_50' in df_1h.columns:
            sma_20_1h = df_1h['sma_20'].iloc[-1]
            sma_50_1h = df_1h['sma_50'].iloc[-1]
            trends['1h'] = 'bullish' if sma_20_1h > sma_50_1h else 'bearish'
        else:
            trends['1h'] = 'neutral'
    else:
        trends['1h'] = 'neutral'
    
    # 4H trend
    if df_4h is not None and len(df_4h) >= 50:
        if 'sma_20' in df_4h.columns and 'sma_50' in df_4h.columns:
            sma_20_4h = df_4h['sma_20'].iloc[-1]
            sma_50_4h = df_4h['sma_50'].iloc[-1]
            trends['4h'] = 'bullish' if sma_20_4h > sma_50_4h else 'bearish'
        else:
            trends['4h'] = 'neutral'
    else:
        trends['4h'] = 'neutral'
    
    # 1D trend
    if df_1d is not None and len(df_1d) >= 50:
        if 'sma_20' in df_1d.columns and 'sma_50' in df_1d.columns:
            sma_20_1d = df_1d['sma_20'].iloc[-1]
            sma_50_1d = df_1d['sma_50'].iloc[-1]
            trends['1d'] = 'bullish' if sma_20_1d > sma_50_1d else 'bearish'
        else:
            trends['1d'] = 'neutral'
    else:
        trends['1d'] = 'neutral'
    
    # Check alignment
    bullish_count = sum(1 for t in trends.values() if t == 'bullish')
    bearish_count = sum(1 for t in trends.values() if t == 'bearish')
    
    if bullish_count >= 2:
        return 'aligned_bullish', trends
    elif bearish_count >= 2:
        return 'aligned_bearish', trends
    else:
        return 'conflicted', trends


def detect_support_resistance_dynamic(df, current_price):
    """Dynamically detect support and resistance levels"""
    if df is None or len(df) < 20:
        return None, None
    
    # Get recent highs and lows
    recent_highs = df['high'].tail(50).nlargest(10).values
    recent_lows = df['low'].tail(50).nsmallest(10).values
    
    # Find closest resistance (above current price)
    resistances = [h for h in recent_highs if h > current_price]
    resistance = min(resistances) if resistances else current_price * 1.02
    
    # Find closest support (below current price)
    supports = [l for l in recent_lows if l < current_price]
    support = max(supports) if supports else current_price * 0.98
    
    return support, resistance


def calculate_position_size_risk(current_price, stop_loss, account_risk_pct=0.02):
    """Calculate position size based on risk management"""
    risk_per_trade = account_risk_pct  # 2% default
    distance_to_stop = abs(current_price - stop_loss)
    risk_reward_ratio = distance_to_stop / current_price
    
    return {
        'risk_per_trade': risk_per_trade,
        'distance_to_stop_pct': risk_reward_ratio * 100,
        'recommended_position_size': risk_per_trade / risk_reward_ratio if risk_reward_ratio > 0 else 0
    }


def run_consultant_meeting(symbol, asset_type, current_price, warning_details):
    """
    Run committee meeting with 4 consultants analyzing multiple timeframes
    Returns: {'position': 'LONG'/'SHORT'/'NEUTRAL', 'entry': price, 'target': price, 'stop_loss': price, 'reasoning': str}
    """
    
    # Fetch data for multiple timeframes
    df_1h, source_1h = fetch_data_for_timeframe(symbol, asset_type, 1)
    
    # Only fetch additional timeframes if 1h was successful
    df_4h, source_4h = fetch_data_for_timeframe(symbol, asset_type, 4) if df_1h is not None else (None, None)
    
    df_1d, source_1d = fetch_data_for_timeframe(symbol, asset_type, 24) if df_1h is not None else (None, None)
    
    # If we can't get any data, return NEUTRAL
    if df_1h is None or len(df_1h) < 50:
        return {
            'position': 'NEUTRAL',
            'entry': current_price,
            'target': current_price,
            'stop_loss': current_price,
            'reasoning': 'Insufficient data for analysis'
        }
    
    # Multi-timeframe analysis
    trend_alignment, trends = analyze_multi_timeframe(df_1h, df_4h, df_1d)
    
    # Dynamic support/resistance
    support, resistance = detect_support_resistance_dynamic(df_1h, current_price)
    
    # Technical indicators from 1H
    current_rsi = df_1h['rsi'].iloc[-1] if 'rsi' in df_1h.columns else 50
    
    # Count warnings
    warning_count = sum(1 for k, v in warning_details.items() if isinstance(v, dict) and v.get('active')) if isinstance(warning_details, dict) else 0
    
    # CONSULTANT VOTES
    votes = []
    
    # Consultant 1: Trend Follower
    if trend_alignment == 'aligned_bullish' and current_rsi < 70:
        votes.append('LONG')
    elif trend_alignment == 'aligned_bearish' and current_rsi > 30:
        votes.append('SHORT')
    else:
        votes.append('NEUTRAL')
    
    # Consultant 2: Mean Reversion
    if current_rsi < 30:
        votes.append('LONG')
    elif current_rsi > 70:
        votes.append('SHORT')
    else:
        votes.append('NEUTRAL')
    
    # Consultant 3: Support/Resistance
    distance_to_support = abs(current_price - support) / current_price
    distance_to_resistance = abs(resistance - current_price) / current_price
    
    if distance_to_support < 0.01:  # Near support
        votes.append('LONG')
    elif distance_to_resistance < 0.01:  # Near resistance
        votes.append('SHORT')
    else:
        votes.append('NEUTRAL')
    
    # Consultant 4: Risk Manager (considers warnings)
    if warning_count >= 3:
        votes.append('NEUTRAL')
    elif warning_count >= 2:
        # Cautious, only agrees if majority is strong
        if votes.count('LONG') >= 2:
            votes.append('LONG')
        elif votes.count('SHORT') >= 2:
            votes.append('SHORT')
        else:
            votes.append('NEUTRAL')
    else:
        # Low warnings, follows majority
        if votes.count('LONG') >= votes.count('SHORT'):
            votes.append('LONG')
        else:
            votes.append('SHORT')
    
    # COMMITTEE DECISION
    long_votes = votes.count('LONG')
    short_votes = votes.count('SHORT')
    neutral_votes = votes.count('NEUTRAL')
    
    # Need majority (3/4) for position
    if long_votes >= 3:
        position = 'LONG'
    elif short_votes >= 3:
        position = 'SHORT'
    else:
        position = 'NEUTRAL'
    
    # Calculate entry, target, stop loss
    if position == 'LONG':
        entry = current_price
        
        # Asset-aware target calculation
        if "Forex" in asset_type or "Precious Metals" in asset_type:
            target_pct = 0.01  # 1% for forex
        else:
            target_pct = 0.03  # 3% for crypto
        
        target = entry * (1 + target_pct)
        stop_loss = max(support, entry * 0.98)  # 2% or support, whichever is closer
        
        reasoning = f"LONG: Trend={trend_alignment}, RSI={current_rsi:.0f}, Votes={long_votes}/4, Warnings={warning_count}"
        
    elif position == 'SHORT':
        entry = current_price
        
        # Asset-aware target calculation
        if "Forex" in asset_type or "Precious Metals" in asset_type:
            target_pct = 0.01  # 1% for forex
        else:
            target_pct = 0.03  # 3% for crypto
        
        target = entry * (1 - target_pct)
        stop_loss = min(resistance, entry * 1.02)  # 2% or resistance, whichever is closer
        
        reasoning = f"SHORT: Trend={trend_alignment}, RSI={current_rsi:.0f}, Votes={short_votes}/4, Warnings={warning_count}"
        
    else:
        entry = current_price
        target = current_price
        stop_loss = current_price
        reasoning = f"NEUTRAL: Conflicting signals, RSI={current_rsi:.0f}, Votes=L:{long_votes}/S:{short_votes}/N:{neutral_votes}, Warnings={warning_count}"
    
    return {
        'position': position,
        'entry': entry,
        'target': target,
        'stop_loss': stop_loss,
        'reasoning': reasoning
    }
