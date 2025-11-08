"""
Consultants Module - 4 AI consultants (C1, C2, C3, C4) + multi-timeframe analysis
"""
from support_resistance import find_support_resistance_zones, check_at_key_level, get_price_targets_based_on_sr
from data_api import fetch_data
from indicators import calculate_technical_indicators


def consultant_c1_pattern_structure(df, symbol):
    """
    C1: Pattern & Structure Analysis
    Focus: Identifies LOCATION (Support/Resistance/Mid-range)
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
                reasoning.append(f"At STRONG resistance ${level_info['price']:,.0f} ({level_info['touches']} rejections)")
            else:
                strength = 7
                reasoning.append(f"At resistance ${level_info['price']:,.0f} ({level_info['touches']} tests)")
        
        elif detected_level_type == 'SUPPORT':
            signal = "AT_SUPPORT"
            if level_info['strength'] == 'STRONG':
                strength = 9
                reasoning.append(f"At STRONG support ${level_info['price']:,.0f} ({level_info['touches']} bounces)")
            else:
                strength = 7
                reasoning.append(f"At support ${level_info['price']:,.0f} ({level_info['touches']} tests)")
    
    # If not at key level, check proximity
    else:
        if targets['next_resistance']:
            next_r = targets['next_resistance']
            distance_pct = ((next_r['price'] - close) / close) * 100
            
            if distance_pct < 2:
                signal = "NEAR_RESISTANCE"
                strength = 6
                reasoning.append(f"Near resistance ${next_r['price']:,.0f}")
        
        if targets['next_support']:
            next_s = targets['next_support']
            distance_pct = ((close - next_s['price']) / close) * 100
            
            if distance_pct < 2:
                signal = "NEAR_SUPPORT"
                strength = 6
                reasoning.append(f"Near support ${next_s['price']:,.0f}")
    
    # Build detailed reasoning with targets
    reasoning_text = " ".join(reasoning) if reasoning else "neutral"
    
    # Add price targets to reasoning
    if targets['next_resistance']:
        reasoning_text += f" | Next R: ${targets['next_resistance']['price']:,.0f}"
    if targets['next_support']:
        reasoning_text += f" | Next S: ${targets['next_support']['price']:,.0f}"
    
    return {
        "signal": signal,
        "strength": strength,
        "reasoning": reasoning_text,
        "targets": targets,
        "sr_zones": sr_zones,
        "at_key_level": at_key_level,
        "level_type": level_type
    }


def consultant_c2_trend_momentum(df, symbol, c1_result=None):
    """
    C2: Momentum Confirmation Analyst
    Focus: Confirms if momentum is ACTUALLY reversing at support/resistance
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
    
    # IF C1 IS AT SUPPORT OR RESISTANCE, CHECK FOR REVERSAL
    if c1_result and c1_result.get('at_key_level'):
        level_type = c1_result.get('level_type')
        
        if level_type == 'SUPPORT':
            # CHECK FOR BULLISH REVERSAL AT SUPPORT
            rsi_prev_5 = [df['rsi'].iloc[i] for i in range(-6, -1) if i >= -len(df)]
            if len(rsi_prev_5) >= 4:
                rsi_low = min(rsi_prev_5)
                if rsi < 40 and rsi > rsi_low + 3:
                    confirmation_score += 4
                    reasoning.append(f"RSI reversal: {rsi_low:.0f}→{rsi:.0f}")
            
            avg_volume = df['volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 1.5:
                confirmation_score += 3
                reasoning.append(f"Volume spike: {volume_ratio:.1f}x")
            elif volume_ratio > 1.2:
                confirmation_score += 1
                reasoning.append(f"Volume elevated: {volume_ratio:.1f}x")
            
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
            if macd > macd_prev and macd > macd_signal:
                confirmation_score += 1
                reasoning.append("MACD turning bullish")
            
            if adx > 25:
                confirmation_score += 1
                reasoning.append(f"Strong trend ADX:{adx:.0f}")
            
            if confirmation_score >= 6:
                signal = "BULLISH_REVERSAL_CONFIRMED"
                strength = min(confirmation_score, 10)
                reversal_confirmed = True
            elif confirmation_score >= 3:
                signal = "POSSIBLE_BULLISH_REVERSAL"
                strength = confirmation_score
            else:
                signal = "NO_BULLISH_CONFIRMATION"
                strength = confirmation_score
        
        elif level_type == 'RESISTANCE':
            # CHECK FOR BEARISH REVERSAL AT RESISTANCE
            rsi_prev_5 = [df['rsi'].iloc[i] for i in range(-6, -1) if i >= -len(df)]
            if len(rsi_prev_5) >= 4:
                rsi_high = max(rsi_prev_5)
                if rsi > 60 and rsi < rsi_high - 3:
                    confirmation_score += 4
                    reasoning.append(f"RSI reversal: {rsi_high:.0f}→{rsi:.0f}")
            
            avg_volume = df['volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 1.5:
                confirmation_score += 3
                reasoning.append(f"Volume spike: {volume_ratio:.1f}x")
            elif volume_ratio > 1.2:
                confirmation_score += 1
                reasoning.append(f"Volume elevated: {volume_ratio:.1f}x")
            
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
            if macd < macd_prev and macd < macd_signal:
                confirmation_score += 1
                reasoning.append("MACD turning bearish")
            
            if adx > 25:
                confirmation_score += 1
                reasoning.append(f"Strong trend ADX:{adx:.0f}")
            
            if confirmation_score >= 6:
                signal = "BEARISH_REVERSAL_CONFIRMED"
                strength = min(confirmation_score, 10)
                reversal_confirmed = True
            elif confirmation_score >= 3:
                signal = "POSSIBLE_BEARISH_REVERSAL"
                strength = confirmation_score
            else:
                signal = "NO_BEARISH_CONFIRMATION"
                strength = confirmation_score
    
    # IF C1 IS MID-RANGE, CHECK MOMENTUM DIRECTION
    else:
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


def consultant_c3_risk_warnings(df, symbol, warnings):
    """C3: Risk & Reversal Analysis"""
    if df is None or len(df) < 50:
        return {"signal": "ACCEPTABLE", "strength": 5, "reasoning": "Insufficient data"}
    
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


def consultant_c4_news_sentiment(symbol, news_data=None):
    """C4: News & Sentiment Analysis"""
    if not news_data:
        return {
            "signal": "NO_NEWS",
            "strength": 5,
            "weight": 5,
            "reasoning": "No significant news"
        }
    
    return {
        "signal": "NO_NEWS",
        "strength": 5,
        "weight": 5,
        "reasoning": "Regular news (ignored)"
    }


def fetch_data_for_timeframe(symbol_param, asset_type_param, timeframe_hours):
    """Fetch data for specific timeframe (1h, 4h, 24h)"""
    if timeframe_hours == 1:
        timeframe = "1h"
    elif timeframe_hours == 4:
        timeframe = "4h"
    elif timeframe_hours == 24:
        timeframe = "1d"
    else:
        timeframe = "1h"
    
    df, source = fetch_data(symbol_param, asset_type_param, timeframe, limit=100)
    
    if df is not None and len(df) > 0:
        df = calculate_technical_indicators(df)
    
    return df, source


def run_consultant_meeting(symbol, asset_type, current_price, warning_details):
    """Run multi-timeframe consultant meeting and return recommendation"""
    import streamlit as st
    
    st.markdown("🔄 Analyzing multiple timeframes for confirmation...")
    
    # Fetch 1h, 4h, 1d data
    df_1h, source_1h = fetch_data_for_timeframe(symbol, asset_type, 1)
    st.success(f"✅ Loaded {len(df_1h) if df_1h is not None else 0} data points from {source_1h}")
    
    df_4h, source_4h = fetch_data_for_timeframe(symbol, asset_type, 4)
    st.success(f"✅ Loaded {len(df_4h) if df_4h is not None else 0} data points from {source_4h}")
    
    df_1d, source_1d = fetch_data_for_timeframe(symbol, asset_type, 24)
    st.success(f"✅ Loaded {len(df_1d) if df_1d is not None else 0} data points from {source_1d}")
    
    # Run consultants on each timeframe
    results = {}
    for tf_name, df in [("1h", df_1h), ("4h", df_4h), ("1d", df_1d)]:
        if df is not None and len(df) >= 50:
            c1 = consultant_c1_pattern_structure(df, symbol)
            c2 = consultant_c2_trend_momentum(df, symbol, c1)
            c3 = consultant_c3_risk_warnings(df, symbol, warning_details)
            c4 = consultant_c4_news_sentiment(symbol)
            
            # Determine position for this timeframe
            if c1['signal'] in ['AT_SUPPORT', 'NEAR_SUPPORT'] and c2['reversal_confirmed']:
                position = "LONG"
            elif c1['signal'] in ['AT_RESISTANCE', 'NEAR_RESISTANCE'] and c2['reversal_confirmed']:
                position = "SHORT"
            else:
                position = "NEUTRAL"
            
            results[tf_name] = {
                "position": position,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c4": c4
            }
    
    # Multi-timeframe consensus
    positions = [results[tf]["position"] for tf in results]
    
    if positions.count("LONG") >= 2:
        final_position = "LONG"
    elif positions.count("SHORT") >= 2:
        final_position = "SHORT"
    else:
        final_position = "NEUTRAL"
    
    # Use 1h timeframe for detailed reasoning
    primary_result = results.get("1h", results.get("4h", results.get("1d")))
    
    if final_position != "NEUTRAL" and primary_result:
        c1 = primary_result['c1']
        target_pct = 0.03 if asset_type == "crypto" else 0.01
        stop_pct = 0.02
        
        if final_position == "LONG":
            target = current_price * (1 + target_pct)
            stop_loss = current_price * (1 - stop_pct)
        else:
            target = current_price * (1 - target_pct)
            stop_loss = current_price * (1 + stop_pct)
        
        return {
            "position": final_position,
            "confidence": 70,
            "entry": current_price,
            "target": target,
            "stop_loss": stop_loss,
            "reasoning": f"Multi-timeframe consensus: {positions}",
            "timeframes": results
        }
    
    return {
        "position": "NEUTRAL",
        "confidence": 0,
        "reasoning": "No clear setup",
        "timeframes": results
    }
