"""
Consultants Module - Advanced Committee-Based Trading Decisions
Implements full brainstorming session logic with C1/C2 collaboration
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Import from other modules
from data_api import fetch_data
from indicators import calculate_technical_indicators


# ==================== SUPPORT/RESISTANCE ANALYSIS ====================

def find_support_resistance_zones(df, lookback=100):
    """
    Professional S/R zone detection with rejection counting
    Returns zones with strength ratings based on historical tests
    """
    if df is None or len(df) < lookback:
        return {'support': [], 'resistance': []}
    
    # Use last N candles
    df_analysis = df.tail(lookback).copy()
    
    # Find local peaks (resistance) and troughs (support)
    highs = df_analysis['high'].values
    lows = df_analysis['low'].values
    closes = df_analysis['close'].values
    
    support_zones = []
    resistance_zones = []
    
    # Group similar price levels (within 1% tolerance)
    def group_levels(prices, tolerance=0.01):
        if len(prices) == 0:
            return []
        
        prices_sorted = sorted(prices)
        groups = []
        current_group = [prices_sorted[0]]
        
        for price in prices_sorted[1:]:
            if abs(price - current_group[-1]) / current_group[-1] <= tolerance:
                current_group.append(price)
            else:
                groups.append(current_group)
                current_group = [price]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    # Find resistance (price failed to break above)
    resistance_prices = []
    for i in range(2, len(highs) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and \
           highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            resistance_prices.append(highs[i])
    
    # Find support (price failed to break below)
    support_prices = []
    for i in range(2, len(lows) - 2):
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and \
           lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            support_prices.append(lows[i])
    
    # Group and count touches
    resistance_groups = group_levels(resistance_prices)
    for group in resistance_groups:
        avg_price = np.mean(group)
        touches = len(group)
        strength = 'STRONG' if touches >= 3 else 'MEDIUM' if touches >= 2 else 'WEAK'
        resistance_zones.append({
            'price': avg_price,
            'touches': touches,
            'strength': strength
        })
    
    support_groups = group_levels(support_prices)
    for group in support_groups:
        avg_price = np.mean(group)
        touches = len(group)
        strength = 'STRONG' if touches >= 3 else 'MEDIUM' if touches >= 2 else 'WEAK'
        support_zones.append({
            'price': avg_price,
            'touches': touches,
            'strength': strength
        })
    
    # Sort by strength (touches)
    support_zones.sort(key=lambda x: x['touches'], reverse=True)
    resistance_zones.sort(key=lambda x: x['touches'], reverse=True)
    
    return {
        'support': support_zones[:5],  # Top 5
        'resistance': resistance_zones[:5]
    }


def get_price_targets_based_on_sr(current_price, sr_zones):
    """
    Get next support and resistance targets based on current price
    """
    support_levels = sr_zones.get('support', [])
    resistance_levels = sr_zones.get('resistance', [])
    
    # Find next resistance (above current price)
    next_resistance = None
    for level in resistance_levels:
        if level['price'] > current_price:
            if next_resistance is None or level['price'] < next_resistance['price']:
                next_resistance = level
    
    # Find next support (below current price)
    next_support = None
    for level in support_levels:
        if level['price'] < current_price:
            if next_support is None or level['price'] > next_support['price']:
                next_support = level
    
    return {
        'next_support': next_support,
        'next_resistance': next_resistance
    }


def check_at_key_level(current_price, sr_zones, tolerance=0.01):
    """
    Check if current price is AT a key support or resistance level
    Returns: (is_at_level, level_type, level_info)
    """
    support_levels = sr_zones.get('support', [])
    resistance_levels = sr_zones.get('resistance', [])
    
    # Check resistance
    for level in resistance_levels:
        distance_pct = abs(current_price - level['price']) / current_price
        if distance_pct <= tolerance:
            return True, 'RESISTANCE', level
    
    # Check support
    for level in support_levels:
        distance_pct = abs(current_price - level['price']) / current_price
        if distance_pct <= tolerance:
            return True, 'SUPPORT', level
    
    return False, None, None


# ==================== CONSULTANT C1: PATTERN & STRUCTURE ====================

def consultant_c1_pattern_structure(df, symbol):
    """
    C1: Pattern & Structure Analysis
    Focus: Identifies LOCATION (Support/Resistance/Mid-range)
    DOES NOT predict direction - only identifies WHERE price is
    """
    if df is None or len(df) < 50:
        return {
            "signal": "MID_RANGE", 
            "strength": 5, 
            "reasoning": "Insufficient data", 
            "targets": None,
            "at_key_level": False,
            "level_type": None
        }
    
    latest = df.iloc[-1]
    close = latest.get('close', 0)
    
    signal = "MID_RANGE"
    strength = 5
    reasoning = []
    at_key_level = False
    level_type = None
    
    # Professional S/R Analysis
    sr_zones = find_support_resistance_zones(df, lookback=100)
    targets = get_price_targets_based_on_sr(close, sr_zones)
    at_level, detected_level_type, level_info = check_at_key_level(close, sr_zones)
    
    # Check if AT a key level
    if at_level:
        at_key_level = True
        level_type = detected_level_type
        
        if detected_level_type == 'RESISTANCE':
            signal = "AT_RESISTANCE"
            if level_info['strength'] == 'STRONG':
                strength = 9
                reasoning.append(f"At STRONG resistance ${level_info['price']:,.2f} ({level_info['touches']} rejections)")
            else:
                strength = 7
                reasoning.append(f"At resistance ${level_info['price']:,.2f} ({level_info['touches']} tests)")
        
        elif detected_level_type == 'SUPPORT':
            signal = "AT_SUPPORT"
            if level_info['strength'] == 'STRONG':
                strength = 9
                reasoning.append(f"At STRONG support ${level_info['price']:,.2f} ({level_info['touches']} bounces)")
            else:
                strength = 7
                reasoning.append(f"At support ${level_info['price']:,.2f} ({level_info['touches']} tests)")
    
    # If not at key level, check proximity
    else:
        if targets['next_resistance']:
            next_r = targets['next_resistance']
            distance_pct = ((next_r['price'] - close) / close) * 100
            
            if distance_pct < 2:  # Very close to resistance
                signal = "NEAR_RESISTANCE"
                strength = 6
                reasoning.append(f"Near resistance ${next_r['price']:,.2f}")
        
        if targets['next_support']:
            next_s = targets['next_support']
            distance_pct = ((close - next_s['price']) / close) * 100
            
            if distance_pct < 2:  # Very close to support
                signal = "NEAR_SUPPORT"
                strength = 6
                reasoning.append(f"Near support ${next_s['price']:,.2f}")
    
    # Build detailed reasoning with targets
    reasoning_text = " ".join(reasoning) if reasoning else "Mid-range"
    
    # Add price targets to reasoning
    if targets['next_resistance']:
        reasoning_text += f" | Next R: ${targets['next_resistance']['price']:,.2f}"
    if targets['next_support']:
        reasoning_text += f" | Next S: ${targets['next_support']['price']:,.2f}"
    
    return {
        "signal": signal,  # AT_SUPPORT, AT_RESISTANCE, NEAR_SUPPORT, NEAR_RESISTANCE, or MID_RANGE
        "strength": strength,
        "reasoning": reasoning_text,
        "targets": targets,
        "sr_zones": sr_zones,
        "at_key_level": at_key_level,
        "level_type": level_type
    }


# ==================== CONSULTANT C2: MOMENTUM CONFIRMATION ====================

def consultant_c2_trend_momentum(df, symbol, c1_result=None):
    """
    C2: Momentum Confirmation Analyst
    Focus: Confirms if momentum is ACTUALLY reversing at support/resistance
    Uses: RSI reversal patterns, Volume spikes, Historical success rate
    """
    if df is None or len(df) < 50:
        return {
            "signal": "NO_CONFIRMATION", 
            "strength": 0, 
            "reasoning": "Insufficient data",
            "reversal_confirmed": False
        }
    
    latest = df.iloc[-1]
    close = latest.get('close', 0)
    rsi = latest.get('rsi', 50)
    volume = latest.get('volume', 0)
    adx = latest.get('adx', 20)
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    
    signal = "NO_CONFIRMATION"
    strength = 0
    reasoning = []
    reversal_confirmed = False
    confirmation_score = 0
    
    # === IF C1 IS AT SUPPORT OR RESISTANCE, CHECK FOR REVERSAL ===
    if c1_result and c1_result.get('at_key_level'):
        level_type = c1_result.get('level_type')
        
        if level_type == 'SUPPORT':
            # === CHECK FOR BULLISH REVERSAL AT SUPPORT ===
            
            # Check #1: RSI Reversal Pattern (Most Important)
            rsi_prev_5 = [df['rsi'].iloc[i] for i in range(-6, -1) if i >= -len(df)]
            if len(rsi_prev_5) >= 4:
                rsi_low = min(rsi_prev_5)
                if rsi < 40 and rsi > rsi_low + 3:
                    # RSI forming higher low (reversal pattern!)
                    confirmation_score += 4
                    reasoning.append(f"RSI reversal: {rsi_low:.0f}→{rsi:.0f}")
            
            # Check #2: Volume Spike
            avg_volume = df['volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 1.5:
                confirmation_score += 3
                reasoning.append(f"Volume spike: {volume_ratio:.1f}x")
            elif volume_ratio > 1.2:
                confirmation_score += 1
                reasoning.append(f"Volume elevated: {volume_ratio:.1f}x")
            
            # Check #3: MACD Turning Up
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
            if macd > macd_prev and macd > macd_signal:
                confirmation_score += 1
                reasoning.append("MACD turning bullish")
            
            # Check #4: ADX (Trend Strength)
            if adx > 25:
                confirmation_score += 1
                reasoning.append(f"Strong trend ADX:{adx:.0f}")
            
            # Decision for Support
            if confirmation_score >= 6:
                signal = "BULLISH_REVERSAL_CONFIRMED"
                strength = min(confirmation_score, 10)
                reversal_confirmed = True
            elif confirmation_score >= 3:
                signal = "POSSIBLE_BULLISH_REVERSAL"
                strength = confirmation_score
                reversal_confirmed = False
            else:
                signal = "NO_BULLISH_CONFIRMATION"
                strength = confirmation_score
                reversal_confirmed = False
        
        elif level_type == 'RESISTANCE':
            # === CHECK FOR BEARISH REVERSAL AT RESISTANCE ===
            
            # Check #1: RSI Reversal Pattern
            rsi_prev_5 = [df['rsi'].iloc[i] for i in range(-6, -1) if i >= -len(df)]
            if len(rsi_prev_5) >= 4:
                rsi_high = max(rsi_prev_5)
                if rsi > 60 and rsi < rsi_high - 3:
                    # RSI forming lower high (reversal pattern!)
                    confirmation_score += 4
                    reasoning.append(f"RSI reversal: {rsi_high:.0f}→{rsi:.0f}")
            
            # Check #2: Volume Spike
            avg_volume = df['volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 1.5:
                confirmation_score += 3
                reasoning.append(f"Volume spike: {volume_ratio:.1f}x")
            elif volume_ratio > 1.2:
                confirmation_score += 1
                reasoning.append(f"Volume elevated: {volume_ratio:.1f}x")
            
            # Check #3: MACD Turning Down
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
            if macd < macd_prev and macd < macd_signal:
                confirmation_score += 1
                reasoning.append("MACD turning bearish")
            
            # Check #4: ADX (Trend Strength)
            if adx > 25:
                confirmation_score += 1
                reasoning.append(f"Strong trend ADX:{adx:.0f}")
            
            # Decision for Resistance
            if confirmation_score >= 6:
                signal = "BEARISH_REVERSAL_CONFIRMED"
                strength = min(confirmation_score, 10)
                reversal_confirmed = True
            elif confirmation_score >= 3:
                signal = "POSSIBLE_BEARISH_REVERSAL"
                strength = confirmation_score
                reversal_confirmed = False
            else:
                signal = "NO_BEARISH_CONFIRMATION"
                strength = confirmation_score
                reversal_confirmed = False
    
    # === IF C1 IS MID-RANGE, CHECK MOMENTUM DIRECTION ===
    else:
        # Standard momentum analysis (when not at S/R)
        if adx > 40:
            if close > df['sma_20'].iloc[-1]:
                signal = "BULLISH"
                strength = 8
                reasoning.append(f"ADX {adx:.1f} strong uptrend")
            else:
                signal = "BEARISH"
                strength = 8
                reasoning.append(f"ADX {adx:.1f} strong downtrend")
        
        elif macd > macd_signal and macd > 0:
            signal = "BULLISH"
            strength = 6
            reasoning.append("MACD bullish")
        
        elif macd < macd_signal and macd < 0:
            signal = "BEARISH"
            strength = 6
            reasoning.append("MACD bearish")
        
        else:
            signal = "NEUTRAL"
            strength = 5
            reasoning.append("No clear momentum")
    
    return {
        "signal": signal,
        "strength": strength,
        "reasoning": " | ".join(reasoning) if reasoning else "neutral",
        "reversal_confirmed": reversal_confirmed,
        "confirmation_score": confirmation_score
    }


# ==================== CONSULTANT C3: RISK & WARNINGS ====================

def consultant_c3_risk_warnings(df, symbol, warnings):
    """
    C3: Risk & Warning Analysis
    Focus: Counts warnings and assesses risk level
    """
    if df is None or len(df) < 50:
        return {"signal": "ACCEPTABLE", "strength": 5, "reasoning": "Insufficient data"}
    
    # Count active warnings
    warning_count = 0
    warning_types = []
    
    if warnings:
        if warnings.get('news_warning'):
            warning_count += 1
            warning_types.append("news")
        if warnings.get('price_warning'):
            warning_count += 1
            warning_types.append("price")
        if warnings.get('volume_warning'):
            warning_count += 1
            warning_types.append("volume")
        if warnings.get('momentum_warning'):
            warning_count += 1
            warning_types.append("momentum")
    
    # Risk assessment
    if warning_count == 0:
        signal = "ACCEPTABLE"
        strength = 8
        reasoning = "0 warnings"
    elif warning_count == 1:
        signal = "CAUTION"
        strength = 6
        reasoning = f"1 warning ({warning_types[0]})"
    elif warning_count == 2:
        signal = "HIGH_RISK"
        strength = 5
        reasoning = f"2 warnings ({','.join(warning_types)})"
    else:
        signal = "EXTREME_RISK"
        strength = 1
        reasoning = f"{warning_count} warnings - DO_NOT_TRADE"
    
    return {
        "signal": signal,
        "strength": strength,
        "reasoning": reasoning
    }


# ==================== CONSULTANT C4: NEWS & SENTIMENT ====================

def consultant_c4_news_sentiment(symbol, news_data=None):
    """
    C4: News & Sentiment Analysis
    Focus: Critical and Major news only
    """
    if not news_data:
        return {
            "signal": "NO_NEWS",
            "strength": 5,
            "weight": 5,  # Low weight when no news
            "reasoning": "No significant news"
        }
    
    # Classify news importance
    critical_keywords = ['sec lawsuit', 'government ban', 'exchange hack', 'bankruptcy', 'fraud', 'investigation']
    major_keywords = ['partnership', 'listing', 'institutional', 'adoption', 'regulation approved']
    
    has_critical = any(keyword in str(news_data).lower() for keyword in critical_keywords)
    has_major = any(keyword in str(news_data).lower() for keyword in major_keywords)
    
    if has_critical:
        # Critical news - very high weight
        sentiment = news_data.get('sentiment', 'neutral')
        return {
            "signal": "BULLISH" if sentiment == 'positive' else "BEARISH",
            "strength": 9,
            "weight": 70,  # Critical news gets 70% weight
            "reasoning": "CRITICAL_NEWS_OVERRIDE"
        }
    elif has_major:
        # Major news - moderate weight
        sentiment = news_data.get('sentiment', 'neutral')
        return {
            "signal": "BULLISH" if sentiment == 'positive' else "BEARISH",
            "strength": 7,
            "weight": 40,  # Major news gets 40% weight
            "reasoning": "MAJOR_NEWS_IMPACT"
        }
    else:
        # Regular news - ignored
        return {
            "signal": "NO_NEWS",
            "strength": 5,
            "weight": 5,  # Regular news basically ignored
            "reasoning": "Regular news (ignored)"
        }


# ==================== MEETING RESOLUTION ====================

def consultant_meeting_resolution(c1, c2, c3, c4, current_price, asset_type=None, timeframe_hours=4):
    """
    NEW LOGIC: C1 identifies location, C2 confirms reversal
    Only trades when BOTH agree on confirmed reversals
    """
    
    # If C3 shows extreme risk (3+ warnings), DO NOT TRADE
    if c3['strength'] <= 2:
        return {
            "position": "NEUTRAL",
            "entry": current_price,
            "target": current_price,
            "stop_loss": current_price,
            "hold_hours": 0,
            "confidence": 0,
            "reasoning": f"DO NOT TRADE - {c3['reasoning']}",
            "risk_reward": 0
        }
    
    # ==================== NEW DECISION LOGIC ====================
    # C1 identifies WHERE (support/resistance/mid-range)
    # C2 confirms IF momentum is reversing
    
    c1_signal = c1['signal']
    c1_strength = c1['strength']
    c2_confirmed = c2.get('reversal_confirmed', False)
    c2_signal = c2['signal']
    c2_strength = c2['strength']
    
    position = "NEUTRAL"
    confidence = 0
    reasoning_parts = []
    
    # === CASE 1: AT SUPPORT ===
    if c1_signal in ['AT_SUPPORT', 'NEAR_SUPPORT']:
        if c2_confirmed and 'BULLISH' in c2_signal:
            # CONFIRMED BULLISH REVERSAL
            position = "LONG"
            confidence = min((c1_strength + c2_strength) / 20 * 100, 90)
            reasoning_parts.append(f"✅ Confirmed bullish reversal at support")
        else:
            # NO CONFIRMATION - WAIT
            position = "NEUTRAL"
            confidence = 0
            reasoning_parts.append(f"⏸️ At support but no reversal confirmation - WAIT")
    
    # === CASE 2: AT RESISTANCE ===
    elif c1_signal in ['AT_RESISTANCE', 'NEAR_RESISTANCE']:
        if c2_confirmed and 'BEARISH' in c2_signal:
            # CONFIRMED BEARISH REVERSAL
            position = "SHORT"
            confidence = min((c1_strength + c2_strength) / 20 * 100, 90)
            reasoning_parts.append(f"✅ Confirmed bearish reversal at resistance")
        else:
            # NO CONFIRMATION - WAIT
            position = "NEUTRAL"
            confidence = 0
            reasoning_parts.append(f"⏸️ At resistance but no reversal confirmation - WAIT")
    
    # === CASE 3: MID-RANGE ===
    else:
        # Follow C2 momentum when not at S/R
        if c2_signal == 'BULLISH' and c2_strength >= 6:
            position = "LONG"
            confidence = c2_strength * 10
            reasoning_parts.append(f"📈 Mid-range bullish momentum")
        elif c2_signal == 'BEARISH' and c2_strength >= 6:
            position = "SHORT"
            confidence = c2_strength * 10
            reasoning_parts.append(f"📉 Mid-range bearish momentum")
        else:
            position = "NEUTRAL"
            confidence = 0
            reasoning_parts.append(f"⚪ Mid-range, no clear momentum")
    
    # === APPLY RISK PENALTY (C3) ===
    risk_multiplier = c3['strength'] / 10.0
    if c3['signal'] == 'HIGH_RISK':
        confidence *= max(risk_multiplier, 0.7)  # Max 30% reduction
    else:
        confidence *= risk_multiplier
    
    # === APPLY NEWS WEIGHT (C4) ===
    if c4['weight'] >= 70:  # Critical news
        if c4['signal'] == 'BULLISH' and position == 'SHORT':
            confidence *= 0.5  # Cut in half if against critical news
        elif c4['signal'] == 'BEARISH' and position == 'LONG':
            confidence *= 0.5
    
    # === MINIMUM CONFIDENCE CHECK ===
    if confidence < 20:
        position = "NEUTRAL"
        confidence = 0
        reasoning_parts.append(f"⚠️ Confidence too low")
    
    # ==================== CALCULATE TARGETS ====================
    # Hold duration - 1 candle prediction
    if c4['weight'] >= 70:
        hold_hours = timeframe_hours * 2  # Critical news: 2 candles
    else:
        hold_hours = timeframe_hours  # Normal: 1 candle
    
    entry = current_price
    
    # Asset-aware max move
    if asset_type and ("Forex" in asset_type or "Precious Metals" in asset_type):
        if hold_hours <= 12:
            max_move_pct = 0.01  # 1% for forex
        else:
            max_move_pct = 0.015  # 1.5% for forex daily
    else:
        if hold_hours <= 8:
            max_move_pct = 0.03  # 3% for crypto
        else:
            max_move_pct = 0.05  # 5% for crypto daily
    
    if position == "LONG":
        # Get S/R target
        targets_sr = c1.get('targets', {})
        next_resistance = targets_sr.get('next_resistance')
        
        if next_resistance and next_resistance['price'] > entry * 1.01:
            target = min(next_resistance['price'] * 0.98, entry * (1 + max_move_pct))
        else:
            target = entry * (1 + max_move_pct)
        
        if target <= entry:
            target = entry * (1 + max_move_pct)
        
        # Stop loss with minimum distance
        min_stop_pct = 0.015 if "Forex" not in str(asset_type) else 0.008
        stop_loss = entry * (1 - max(0.02, min_stop_pct))
    
    elif position == "SHORT":
        # Get S/R target
        targets_sr = c1.get('targets', {})
        next_support = targets_sr.get('next_support')
        
        if next_support and next_support['price'] < entry * 0.99:
            target = max(next_support['price'] * 1.02, entry * (1 - max_move_pct))
        else:
            target = entry * (1 - max_move_pct)
        
        if target >= entry:
            target = entry * (1 - max_move_pct)
        
        # Stop loss with minimum distance
        min_stop_pct = 0.015 if "Forex" not in str(asset_type) else 0.008
        stop_loss = entry * (1 + max(0.02, min_stop_pct))
    
    else:
        target = current_price
        stop_loss = current_price
    
    # Calculate risk/reward
    if position != "NEUTRAL":
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        risk_reward = reward / risk if risk > 0 else 0
    else:
        risk_reward = 0
    
    # Build full reasoning
    full_reasoning = [
        f"C1: {c1['signal']} {c1['strength']}/10 ({c1['reasoning']})",
        f"C2: {c2['signal']} {c2['strength']}/10 ({c2['reasoning']})",
        f"C3: {c3['signal']} ({c3['reasoning']})",
        f"C4: {c4['signal']} (weight: {c4['weight']}%)"
    ]
    
    full_reasoning.extend(reasoning_parts)
    
    return {
        "position": position,
        "entry": entry,
        "target": target,
        "stop_loss": stop_loss,
        "hold_hours": hold_hours,
        "confidence": int(confidence),
        "reasoning": " | ".join(full_reasoning),
        "risk_reward": round(risk_reward, 2)
    }


# ==================== MAIN ENTRY POINT ====================

def run_consultant_meeting(symbol, asset_type, current_price, warning_details):
    """
    Main function called by app.py
    Runs the full consultant meeting process
    """
    
    # Fetch 1H data for analysis
    timeframe_config = {'binance': '1h', 'okx': '1H', 'limit': 100}
    df, source = fetch_data(symbol, asset_type, timeframe_config)
    
    if df is None or len(df) < 50:
        return {
            'position': 'NEUTRAL',
            'entry': current_price,
            'target': current_price,
            'stop_loss': current_price,
            'reasoning': 'Insufficient data for analysis',
            'confidence': 0
        }
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Run all 4 consultants
    c1_result = consultant_c1_pattern_structure(df, symbol)
    c2_result = consultant_c2_trend_momentum(df, symbol, c1_result)  # Pass C1 result!
    c3_result = consultant_c3_risk_warnings(df, symbol, warning_details)
    c4_result = consultant_c4_news_sentiment(symbol, news_data=None)
    
    # Resolve their votes
    meeting_result = consultant_meeting_resolution(
        c1_result, c2_result, c3_result, c4_result,
        current_price, asset_type, timeframe_hours=1
    )
    
    return meeting_result
