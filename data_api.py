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


def fetch_data(symbol, asset_type="crypto", timeframe="1h", limit=100):
    """
    Main data fetching function with multiple fallbacks
    Returns: (df, data_source)
    """
    # Map timeframe
    interval_map = {
        "1h": {"okx": "1H", "binance": "1h"},
        "4h": {"okx": "4H", "binance": "4h"},
        "1d": {"okx": "1D", "binance": "1d"}
    }
    
    intervals = interval_map.get(timeframe, {"okx": "1H", "binance": "1h"})
    
    # Try OKX first (most reliable)
    df = fetch_data_okx(symbol, intervals["okx"], limit)
    if df is not None and len(df) > 0:
        return df, "OKX"
    
    # Try Binance
    df = fetch_data_binance(symbol, intervals["binance"], limit)
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
    return None, "None"


def get_batch_data_binance(symbols_list, interval="1h", limit=100):
    """Batch request capability - fetch multiple symbols at once"""
    results = {}
    for symbol in symbols_list:
        df = fetch_data_binance(symbol, interval, limit)
        if df is not None:
            results[symbol] = df
    return results
