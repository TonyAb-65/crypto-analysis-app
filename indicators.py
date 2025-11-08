"""
Indicators Module - Technical indicator calculations (RSI, MACD, Bollinger, etc.)
"""
import pandas as pd
import numpy as np


def calculate_technical_indicators(df):
    """Calculate all technical indicators for the dataframe"""
    if df is None or len(df) < 20:
        return df
    
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = true_range
    atr_14 = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (abs(minus_dm).rolling(14).mean() / atr_14)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = abs(minus_di)
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # MFI (Money Flow Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    
    mfi_ratio = positive_flow / negative_flow
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    # CCI (Commodity Channel Index)
    tp = typical_price
    sma_tp = tp.rolling(20).mean()
    mad = lambda x: np.mean(np.abs(x - np.mean(x)))
    mad_tp = tp.rolling(20).apply(mad, raw=True)
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def analyze_rsi_bounce_patterns(df):
    """Analyze RSI bounce patterns from historical data"""
    if df is None or len(df) < 50 or 'rsi' not in df.columns:
        return "Insufficient data for RSI analysis"
    
    try:
        rsi_series = df['rsi'].tail(50)
        price_series = df['close'].tail(50)
        
        # Count oversold bounces (RSI < 30)
        oversold_points = rsi_series[rsi_series < 30]
        oversold_count = len(oversold_points)
        
        # Count overbought rejections (RSI > 70)
        overbought_points = rsi_series[rsi_series > 70]
        overbought_count = len(overbought_points)
        
        # Analyze recent RSI trend
        recent_rsi = rsi_series.tail(5).mean()
        
        if recent_rsi < 30:
            rsi_status = "OVERSOLD - Potential bounce zone"
        elif recent_rsi > 70:
            rsi_status = "OVERBOUGHT - Potential rejection zone"
        elif recent_rsi < 45:
            rsi_status = "Below neutral - Bearish momentum"
        elif recent_rsi > 55:
            rsi_status = "Above neutral - Bullish momentum"
        else:
            rsi_status = "Neutral range"
        
        # Calculate RSI momentum
        rsi_change = rsi_series.iloc[-1] - rsi_series.iloc[-5]
        momentum = "rising" if rsi_change > 5 else "falling" if rsi_change < -5 else "stable"
        
        insights = f"""
        Current RSI: {rsi_series.iloc[-1]:.1f} ({rsi_status})
        RSI Momentum: {momentum} ({rsi_change:+.1f} over 5 periods)
        Oversold touches (RSI<30): {oversold_count} in last 50 periods
        Overbought touches (RSI>70): {overbought_count} in last 50 periods
        """
        
        return insights.strip()
        
    except Exception as e:
        return f"Error analyzing RSI: {str(e)}"
