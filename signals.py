"""
Signals Module - Signal strength calculation and warning detection
"""
import json
from database import get_indicator_weights


def calculate_signal_strength(df, warning_details=None):
    """
    Calculate overall signal strength from technical indicators
    Returns: signal_strength (-10 to +10, where negative = bearish, positive = bullish)
    """
    if df is None or len(df) < 20:
        return 0
    
    latest = df.iloc[-1]
    signal_score = 0
    weights = get_indicator_weights()
    
    # RSI (Oversold/Overbought)
    rsi = latest.get('rsi', 50)
    if rsi < 30:
        signal_score += 2 * weights.get('RSI', 1.0)
    elif rsi < 40:
        signal_score += 1 * weights.get('RSI', 1.0)
    elif rsi > 70:
        signal_score -= 2 * weights.get('RSI', 1.0)
    elif rsi > 60:
        signal_score -= 1 * weights.get('RSI', 1.0)
    
    # MACD
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    if macd > macd_signal and macd > 0:
        signal_score += 1
    elif macd < macd_signal and macd < 0:
        signal_score -= 1
    
    # Moving Averages
    close = latest.get('close', 0)
    sma_20 = latest.get('sma_20', close)
    sma_50 = latest.get('sma_50', close)
    
    if close > sma_20 > sma_50:
        signal_score += 2
    elif close < sma_20 < sma_50:
        signal_score -= 2
    elif close > sma_20:
        signal_score += 1
    elif close < sma_20:
        signal_score -= 1
    
    # Bollinger Bands
    bb_upper = latest.get('bb_upper', close * 1.02)
    bb_lower = latest.get('bb_lower', close * 0.98)
    
    if close < bb_lower:
        signal_score += 1
    elif close > bb_upper:
        signal_score -= 1
    
    # Apply warning adjustments if provided
    if warning_details:
        if warning_details.get('price_warning'):
            signal_score *= 0.7
        if warning_details.get('volume_warning'):
            signal_score *= 0.8
        if warning_details.get('momentum_warning'):
            signal_score *= 0.8
        if warning_details.get('news_warning'):
            signal_score *= 0.9
    
    # Cap signal strength between -10 and +10
    signal_score = max(-10, min(10, int(signal_score)))
    
    return signal_score


def calculate_warning_signs(df, signal_strength, news_warning_data=None):
    """
    Calculate warning signs (4 types: price, volume, momentum, news)
    Returns: (warning_count, warning_details)
    """
    warnings = {
        'price_warning': False,
        'price_message': '',
        'volume_warning': False,
        'volume_message': '',
        'momentum_warning': False,
        'momentum_message': '',
        'news_warning': False,
        'news_message': ''
    }
    
    if df is None or len(df) < 20:
        return 0, warnings
    
    latest = df.iloc[-1]
    close = latest.get('close', 0)
    
    # WARNING 1: Price at extreme levels
    high_52 = df['high'].tail(1000).max() if len(df) >= 1000 else df['high'].max()
    low_52 = df['low'].tail(1000).min() if len(df) >= 1000 else df['low'].min()
    
    price_range = high_52 - low_52
    if price_range > 0:
        price_position = (close - low_52) / price_range
        
        if price_position > 0.95 and signal_strength > 0:
            warnings['price_warning'] = True
            warnings['price_message'] = "Price at 52-week high"
        elif price_position < 0.05 and signal_strength < 0:
            warnings['price_warning'] = True
            warnings['price_message'] = "Price at 52-week low"
    
    # WARNING 2: Volume anomaly
    if 'volume' in df.columns and len(df) >= 20:
        avg_volume = df['volume'].tail(20).mean()
        current_volume = latest.get('volume', avg_volume)
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio < 0.3:
                warnings['volume_warning'] = True
                warnings['volume_message'] = f"Very low volume ({volume_ratio:.1f}x avg)"
    
    # WARNING 3: Momentum divergence
    if 'rsi' in df.columns and len(df) >= 10:
        rsi = latest.get('rsi', 50)
        rsi_prev = df['rsi'].iloc[-10]
        price_change = (close - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        # Bearish divergence: price up, RSI down
        if price_change > 0.02 and rsi < rsi_prev - 5:
            warnings['momentum_warning'] = True
            warnings['momentum_message'] = "Bearish divergence (price up, RSI down)"
        
        # Bullish divergence: price down, RSI up
        elif price_change < -0.02 and rsi > rsi_prev + 5:
            warnings['momentum_warning'] = True
            warnings['momentum_message'] = "Bullish divergence (price down, RSI up)"
    
    # WARNING 4: News sentiment
    if news_warning_data and news_warning_data.get('has_warning'):
        warnings['news_warning'] = True
        warnings['news_message'] = news_warning_data.get('warning_message', '')
    
    # Count warnings
    warning_count = sum([
        warnings['price_warning'],
        warnings['volume_warning'],
        warnings['momentum_warning'],
        warnings['news_warning']
    ])
    
    return warning_count, warnings


def create_indicator_snapshot(df):
    """Create snapshot of indicator signals for AI learning"""
    try:
        snapshot = {}
        
        if 'obv' in df.columns:
            obv_current = df['obv'].iloc[-1]
            obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
            obv_change = obv_current - obv_prev
            
            if obv_change > 0 and obv_current > 0:
                signal = 'bullish'
            elif obv_change < 0 or obv_current < 0:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['OBV'] = {'value': float(obv_current), 'signal': signal}
        
        if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
            adx = df['adx'].iloc[-1]
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            
            if adx > 25 and plus_di > minus_di:
                signal = 'bullish'
            elif adx > 25 and minus_di > plus_di:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['ADX'] = {'value': float(adx), 'signal': signal}
        
        if 'stoch_k' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            
            if stoch_k < 20:
                signal = 'bullish'
            elif stoch_k > 80:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['Stochastic'] = {'value': float(stoch_k), 'signal': signal}
        
        if 'mfi' in df.columns:
            mfi = df['mfi'].iloc[-1]
            
            if mfi < 20:
                signal = 'bullish'
            elif mfi > 80:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['MFI'] = {'value': float(mfi), 'signal': signal}
        
        if 'cci' in df.columns:
            cci = df['cci'].iloc[-1]
            
            if cci < -100:
                signal = 'bullish'
            elif cci > 100:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            snapshot['CCI'] = {'value': float(cci), 'signal': signal}
        
        return snapshot
        
    except Exception as e:
        print(f"Error creating indicator snapshot: {e}")
        return {}
