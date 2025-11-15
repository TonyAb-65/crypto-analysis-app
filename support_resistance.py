"""
Support & Resistance Module - TWELVE DATA API
Uses professional S/R levels directly from Twelve Data pivot points
NO internal calculation - professional grade data only
"""
import requests
import numpy as np


def find_support_resistance_zones(df, symbol=None, interval='1h'):
    """
    Fetch Support/Resistance from Twelve Data API
    Uses professional pivot points calculation
    
    Args:
        df: Price dataframe (for fallback only)
        symbol: Trading symbol (e.g., 'BTC/USD', 'EUR/USD', 'XRP/USD')
        interval: Timeframe ('1h', '4h', '1d', etc.)
    
    Returns:
        Dictionary with resistance and support arrays
    """
    
    if symbol is None:
        print("⚠️ No symbol provided, using fallback S/R")
        return fallback_sr_from_chart(df)
    
    print(f"\n{'='*60}")
    print(f"📊 FETCHING S/R FROM TWELVE DATA API")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    
    # Try to fetch from Twelve Data
    sr_data = fetch_pivot_points_from_twelvedata(symbol, interval)
    
    if sr_data:
        resistance_levels = sr_data['resistance']
        support_levels = sr_data['support']
        
        print(f"✅ SUCCESS - Got {len(resistance_levels)} resistance, {len(support_levels)} support levels")
        print(f"{'='*60}\n")
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    # Fallback if API fails
    print(f"⚠️ Twelve Data API unavailable, using fallback")
    print(f"{'='*60}\n")
    return fallback_sr_from_chart(df)


def fetch_pivot_points_from_twelvedata(symbol, interval='1h'):
    """
    Fetch pivot points from Twelve Data API
    Returns R1/R2/R3 (resistance) and S1/S2/S3 (support)
    
    Twelve Data provides professional pivot point calculations
    """
    try:
        # Twelve Data pivot points endpoint
        # Note: This endpoint may require specific plan level
        url = "https://api.twelvedata.com/pivot_points"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': 1
        }
        
        print(f"🔍 Calling Twelve Data API...")
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"   Response Data: {data}")
            
            # Check for error in response
            if 'code' in data and data['code'] != 200:
                print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                return None
            
            if 'values' in data and len(data['values']) > 0:
                pivots = data['values'][0]
                
                resistance_levels = []
                support_levels = []
                
                # Extract resistance levels (R1, R2, R3)
                if 'classic_resistance_1' in pivots:
                    resistance_levels.append({
                        'price': float(pivots['classic_resistance_1']),
                        'touches': 1,
                        'strength': 'MEDIUM',
                        'status': 'INTACT',
                        'source': 'Twelve Data R1'
                    })
                
                if 'classic_resistance_2' in pivots:
                    resistance_levels.append({
                        'price': float(pivots['classic_resistance_2']),
                        'touches': 2,
                        'strength': 'STRONG',
                        'status': 'INTACT',
                        'source': 'Twelve Data R2'
                    })
                
                if 'classic_resistance_3' in pivots:
                    resistance_levels.append({
                        'price': float(pivots['classic_resistance_3']),
                        'touches': 3,
                        'strength': 'STRONG',
                        'status': 'INTACT',
                        'source': 'Twelve Data R3'
                    })
                
                # Extract support levels (S1, S2, S3)
                if 'classic_support_1' in pivots:
                    support_levels.append({
                        'price': float(pivots['classic_support_1']),
                        'touches': 1,
                        'strength': 'MEDIUM',
                        'status': 'INTACT',
                        'source': 'Twelve Data S1'
                    })
                
                if 'classic_support_2' in pivots:
                    support_levels.append({
                        'price': float(pivots['classic_support_2']),
                        'touches': 2,
                        'strength': 'STRONG',
                        'status': 'INTACT',
                        'source': 'Twelve Data S2'
                    })
                
                if 'classic_support_3' in pivots:
                    support_levels.append({
                        'price': float(pivots['classic_support_3']),
                        'touches': 3,
                        'strength': 'STRONG',
                        'status': 'INTACT',
                        'source': 'Twelve Data S3'
                    })
                
                # Filter by current price if available
                if len(resistance_levels) > 0 or len(support_levels) > 0:
                    # Get current price from pivot point
                    current_price = float(pivots.get('classic_pivot_point', 0))
                    
                    if current_price > 0:
                        # Keep only resistance ABOVE current price
                        resistance_levels = [r for r in resistance_levels if r['price'] > current_price]
                        # Keep only support BELOW current price
                        support_levels = [s for s in support_levels if s['price'] < current_price]
                    
                    print(f"   ✅ Extracted {len(resistance_levels)} resistance, {len(support_levels)} support")
                    
                    return {
                        'resistance': resistance_levels,
                        'support': support_levels
                    }
        
        print(f"❌ No valid data in response")
        return None
        
    except Exception as e:
        print(f"❌ Error fetching from Twelve Data: {e}")
        return None


def fallback_sr_from_chart(df):
    """
    SIMPLE FALLBACK: Use recent highs/lows if API fails
    This is NOT professional calculation - just emergency backup
    """
    
    if df is None or len(df) < 20:
        return {'support': [], 'resistance': []}
    
    current_price = df['close'].iloc[-1]
    
    # Simple: Recent 50-period high/low
    recent_high = df['high'].tail(50).max()
    recent_low = df['low'].tail(50).min()
    
    resistance_levels = []
    support_levels = []
    
    # Only add if in correct direction
    if recent_high > current_price * 1.01:  # At least 1% above
        resistance_levels.append({
            'price': recent_high,
            'touches': 1,
            'strength': 'WEAK',
            'status': 'INTACT',
            'source': 'Recent High (Fallback)'
        })
    
    if recent_low < current_price * 0.99:  # At least 1% below
        support_levels.append({
            'price': recent_low,
            'touches': 1,
            'strength': 'WEAK',
            'status': 'INTACT',
            'source': 'Recent Low (Fallback)'
        })
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
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
    
    # Next resistance (closest above current price)
    next_resistance = None
    if resistances:
        valid_resistance = [r for r in resistances if r['price'] > current_price]
        if valid_resistance:
            next_resistance = min(valid_resistance, key=lambda x: x['price'])
    
    # Next support (closest below current price)
    next_support = None
    if supports:
        valid_support = [s for s in supports if s['price'] < current_price]
        if valid_support:
            next_support = max(valid_support, key=lambda x: x['price'])
    
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
    """Check if predicted price faces S/R barriers"""
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    
    barriers = []
    
    if current_price < predicted_price and predicted_price > high_20:
        barriers.append(('resistance', high_20, abs(predicted_price - high_20)))
    elif current_price > predicted_price and predicted_price < low_20:
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
    """
    Adjust confidence based on barriers and volatility
    
    FIXED: Reduced penalties to allow higher confidence scores
    """
    adjusted_confidence = base_confidence
    
    # Apply gentler penalties for barriers
    for barrier_type, price_level, distance in barriers:
        if 'strong' in barrier_type:
            adjusted_confidence *= 0.90  # 10% penalty (was 0.7 = 30% penalty)
        else:
            adjusted_confidence *= 0.95  # 5% penalty (was 0.85 = 15% penalty)
    
    # Gentler penalty for unrealistic volatility
    if not volatility_context['is_realistic']:
        adjusted_confidence *= 0.85  # 15% penalty (was 0.6 = 40% penalty)
    
    # Keep same min/max bounds
    return max(30.0, min(95.0, adjusted_confidence))
