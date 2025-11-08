"""
Data API Module - Fetch market data from multiple sources with fallback
FIXED VERSION - Matches original working logic exactly
"""
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
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


def get_forex_metals_data(symbol_param, interval="1h", limit=100):
    """Fetch forex and precious metals data using Twelve Data API"""
    interval_map = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day"
    }
    
    mapped_interval = interval_map.get(interval, "1h")
    
    # FIX #1: Check for API key in secrets
    try:
        api_key = st.secrets.get("TWELVE_DATA_API_KEY", None)
    except:
        api_key = None
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol_param,
        "interval": mapped_interval,
        "outputsize": min(limit, 100),
        "format": "JSON"
    }
    
    # FIX #1: Add API key if available
    if api_key:
        params["apikey"] = api_key
    
    try:
        response = get_retry_session().get(url, params=params, timeout=10)
        # FIX #4: Use raise_for_status for proper error handling
        response.raise_for_status()
        data = response.json()
        
        # FIX #3: User-friendly error messages
        if 'status' in data and data['status'] == 'error':
            st.warning(f"⚠️ Twelve Data error: {data.get('message', 'Unknown error')}")
            raise Exception("API returned error status")
        
        if 'values' not in data or not data['values']:
            st.warning(f"⚠️ No data returned for {symbol_param}")
            raise Exception("No data in response")
        
        values = data['values']
        df = pd.DataFrame(values)
        
        df = df.rename(columns={
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        # FIX #5: Proper volume handling for forex
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        else:
            df['volume'] = 0
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # FIX #3: Success message
        api_status = "Twelve Data (API Key)" if api_key else "Twelve Data (Free)"
        st.success(f"✅ Loaded {len(df)} data points from {api_status}")
        return df, api_status
        
    except Exception as e:
        st.warning(f"⚠️ Twelve Data API failed: {str(e)}")
        
    # FIX #2: Sample data fallback
    try:
        st.info("📊 Using sample data for demonstration...")
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
        
        # Different base prices for different instruments
        base_price = 1.0900 if 'EUR' in symbol_param else 110.50 if 'JPY' in symbol_param else 1800 if 'XAU' in symbol_param else 1.2500
        
        prices = []
        current_price_calc = base_price
        for i in range(limit):
            change = np.random.normal(0, base_price * 0.002)
            current_price_calc += change
            prices.append(current_price_calc)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': [p + np.random.normal(0, p * 0.001) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in range(limit)]
        })
        
        st.warning("⚠️ Using sample data. Real data unavailable.")
        return df, "Sample Data"
        
    except Exception as e:
        st.error(f"❌ Error generating sample data: {str(e)}")
        return None, None


def get_okx_data(symbol, interval="1H", limit=100):
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
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], "OKX"
        
        return None, None
    except Exception as e:
        print(f"OKX API error: {e}")
        return None, None


def get_cryptocompare_data(symbol, limit=100):
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
                
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], "CryptoCompare"
        
        return None, None
    except Exception as e:
        print(f"CryptoCompare API error: {e}")
        return None, None


def get_coingecko_data(symbol, limit=100):
    """Fetch data from CoinGecko API"""
    try:
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
            "days": 4,
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
                
                df['open'] = df['close']
                df['high'] = df['close'] * 1.001
                df['low'] = df['close'] * 0.999
                
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit), "CoinGecko"
        
        return None, None
    except Exception as e:
        print(f"CoinGecko API error: {e}")
        return None, None


def fetch_data(symbol_param, asset_type_param, timeframe_config):
    """
    Main function to fetch data - matches original logic exactly
    FIX #6: Now uses timeframe_config properly
    """
    # FIX #7: Add UI feedback
    if asset_type_param == "💰 Cryptocurrency" or asset_type_param == "🔍 Custom Search":
        st.info("🔄 Fetching data from OKX...")
        
        df, source = get_okx_data(symbol_param, timeframe_config['okx'], timeframe_config['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.info("🔄 Trying backup API (CryptoCompare)...")
        df, source = get_cryptocompare_data(symbol_param, timeframe_config['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.info("🔄 Trying backup API (CoinGecko)...")
        df, source = get_coingecko_data(symbol_param, timeframe_config['limit'])
        if df is not None and len(df) > 0:
            return df, source
        
        st.error(f"❌ Could not fetch data for {symbol_param}")
        return None, None
    
    elif asset_type_param == "💱 Forex" or asset_type_param == "🏆 Precious Metals":
        st.info("🔄 Fetching forex/metals data...")
        
        # FIX #6: Use binance interval from config (it's the standard format)
        interval = timeframe_config['binance']
        df, source = get_forex_metals_data(symbol_param, interval, timeframe_config['limit'])
        
        if df is not None and len(df) > 0:
            return df, source
        
        st.error(f"❌ Could not fetch forex/metals data for {symbol_param}")
        return None, None
    
    return None, None


def get_batch_data_binance(symbols_list, interval="1h", limit=100):
    """Batch request capability - fetch multiple symbols at once"""
    results = {}
    try:
        for symbol in symbols_list:
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
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
                    
                    results[symbol] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
    except Exception as e:
        print(f"Batch request error: {e}")
    
    return results
