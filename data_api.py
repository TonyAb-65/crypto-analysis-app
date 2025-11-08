"""
Data API Module - Fetch market data from multiple sources with fallback
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime


def get_retry_session(retries=3, backoff_factor=0.3):
    """Create requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_forex_data(pair, interval="1h", limit=100):
    """Fetch forex data from Alpha Vantage or Twelve Data"""
    try:
        # Try Twelve Data free API (no key required for basic data)
        base_currency = pair.split('/')[0] if '/' in pair else pair[:3]
        quote_currency = pair.split('/')[1] if '/' in pair else pair[3:]
        
        # Format for API
        symbol = f"{base_currency}{quote_currency}"
        
        print(f"Fetching forex data for {symbol} ({base_currency}/{quote_currency})")
        
        # Using Twelve Data
        url = "https://api.twelvedata.com/time_series"
        
        # Map interval
        interval_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1day"
        }
        td_interval = interval_map.get(interval, "1h")
        
        params = {
            "symbol": f"{base_currency}/{quote_currency}",
            "interval": td_interval,
            "outputsize": str(min(limit, 100)),  # Twelve Data free tier limit
            "format": "JSON"
        }
        
        print(f"Twelve Data API request: {url} with params {params}")
        
        response = get_retry_session().get(url, params=params, timeout=15)
        
        print(f"Twelve Data response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for API errors
            if 'status' in data and data['status'] == 'error':
                print(f"Twelve Data API error: {data.get('message', 'Unknown error')}")
                return None
            
            if 'values' in data and len(data['values']) > 0:
                df = pd.DataFrame(data['values'])
                
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = 0  # Forex doesn't have traditional volume
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"Successfully fetched {len(df)} forex data points")
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
            else:
                print(f"No values in Twelve Data response: {data}")
        
        return None
    except Exception as e:
        print(f"Forex API error: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_metals_data(symbol, interval="1h", limit=100):
    """Fetch precious metals data (Gold, Silver)"""
    try:
        # Map symbols
        if symbol == "XAU/USD":
            base = "XAU"
        elif symbol == "XAG/USD":
            base = "XAG"
        else:
            base = symbol.split('/')[0]
        
        # Try Twelve Data
        url = "https://api.twelvedata.com/time_series"
        
        interval_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1day"
        }
        td_interval = interval_map.get(interval, "1h")
        
        params = {
            "symbol": f"{base}/USD",
            "interval": td_interval,
            "outputsize": str(limit),
            "format": "JSON"
        }
        
        response = get_retry_session().get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'values' in data and len(data['values']) > 0:
                df = pd.DataFrame(data['values'])
                
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = 0
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
        
        return None
    except Exception as e:
        print(f"Metals API error: {e}")
        return None


def fetch_data_okx(symbol, interval="1H", limit=100):
    """Fetch data from OKX API"""
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        params = {
            "instId": f"{symbol}-USDT",
            "bar": interval,
            "limit": str(limit)
        }
        
        response = get_retry_session().get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'], columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'volume_currency', 'volume_quote', 'confirm'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return None
    except Exception as e:
        print(f"OKX API error: {e}")
        return None


def fetch_data_binance(symbol, interval="1h", limit=100):
    """Fetch data from Binance API"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": limit
        }
        
        response = get_retry_session().get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return None
    except Exception as e:
        print(f"Binance API error: {e}")
        return None


def fetch_data_cryptocompare(symbol, limit=100):
    """Fetch data from CryptoCompare API"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            "fsym": symbol,
            "tsym": "USDT",
            "limit": limit
        }
        
        response = get_retry_session().get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'Data' in data and 'Data' in data['Data']:
                df = pd.DataFrame(data['Data']['Data'])
                
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volumefrom': 'volume'
                })
                
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return None
    except Exception as e:
        print(f"CryptoCompare API error: {e}")
        return None


def fetch_data_coingecko(symbol, days=4):
    """Fetch data from CoinGecko API (hourly for last N days)"""
    try:
        # Map common symbols to CoinGecko IDs
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network',
            'DOT': 'polkadot',
            'AVAX': 'avalanche-2'
        }
        
        coin_id = symbol_map.get(symbol.upper(), symbol.lower())
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "hourly"
        }
        
        response = get_retry_session().get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data:
                prices = data['prices']
                volumes = data.get('total_volumes', [])
                
                df = pd.DataFrame({
                    'timestamp': [pd.to_datetime(p[0], unit='ms') for p in prices],
                    'close': [p[1] for p in prices],
                    'volume': [v[1] if len(volumes) > 0 else 0 for v in volumes] if len(volumes) > 0 else [0] * len(prices)
                })
                
                # CoinGecko doesn't provide OHLC, so we approximate
                df['open'] = df['close']
                df['high'] = df['close'] * 1.001
                df['low'] = df['close'] * 0.999
                
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(100)
        
        return None
    except Exception as e:
        print(f"CoinGecko API error: {e}")
        return None


def fetch_data(symbol, asset_type="💰 Cryptocurrency", timeframe="1h", limit=100):
    """
    Main data fetching function with multiple fallbacks
    Returns: (df, data_source)
    """
    # Map timeframe
    interval_map = {
        "5minutes": "5m",
        "15minutes": "15m",
        "30minutes": "30m",
        "1hour": "1h",
        "4hours": "4h",
        "1day": "1d"
    }
    
    # Normalize timeframe
    timeframe_lower = timeframe.lower().replace(" ", "")
    interval = interval_map.get(timeframe_lower, "1h")
    
    # Detect asset type by symbol format or asset_type parameter
    is_forex = "/" in symbol and len(symbol.split("/")[0]) == 3
    is_metal = symbol in ["XAU/USD", "XAG/USD"] or "Precious Metals" in str(asset_type)
    is_forex_type = "Forex" in str(asset_type)
    
    # FOREX handling
    if is_forex or is_forex_type:
        print(f"Attempting to fetch Forex data for {symbol}")
        df = fetch_forex_data(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df, "Twelve Data (Forex)"
        print(f"Forex fetch failed for {symbol}")
        return None, "Forex data unavailable"
    
    # METALS handling
    if is_metal:
        print(f"Attempting to fetch Metals data for {symbol}")
        df = fetch_metals_data(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df, "Twelve Data (Metals)"
        print(f"Metals fetch failed for {symbol}")
        return None, "Metals data unavailable"
    
    # CRYPTO handling
    print(f"Attempting to fetch Crypto data for {symbol}")
    intervals = {
        "okx": {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1D"},
        "binance": {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
    }
    
    okx_interval = intervals["okx"].get(interval, "1H")
    binance_interval = intervals["binance"].get(interval, "1h")
    
    # Try OKX first (most reliable)
    df = fetch_data_okx(symbol, okx_interval, limit)
    if df is not None and len(df) > 0:
        return df, "OKX"
    
    # Try Binance
    df = fetch_data_binance(symbol, binance_interval, limit)
    if df is not None and len(df) > 0:
        return df, "Binance"
    
    # Try CryptoCompare
    df = fetch_data_cryptocompare(symbol, limit)
    if df is not None and len(df) > 0:
        return df, "CryptoCompare"
    
    # Try CoinGecko (last resort)
    df = fetch_data_coingecko(symbol, days=4)
    if df is not None and len(df) > 0:
        return df, "CoinGecko"
    
    # All failed
    print(f"All data sources failed for {symbol}")
    return None, "None"


def get_batch_data_binance(symbols_list, interval="1h", limit=100):
    """Batch request capability - fetch multiple symbols at once"""
    results = {}
    for symbol in symbols_list:
        df = fetch_data_binance(symbol, interval, limit)
        if df is not None:
            results[symbol] = df
    return results
