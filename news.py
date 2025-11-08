"""
News & Sentiment Module - Fear & Greed Index and News Sentiment Analysis
"""
import requests
from datetime import datetime


def get_fear_greed_index():
    """Get Fear & Greed Index from Alternative.me API"""
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                value = int(latest['value'])
                classification = latest['value_classification']
                return value, classification
        
        return None, None
    except Exception as e:
        print(f"Error fetching Fear & Greed: {e}")
        return None, None


def get_crypto_news_sentiment(symbol):
    """Get crypto news sentiment (simplified version)"""
    try:
        # This is a placeholder - you can integrate with CryptoCompare, CoinGecko, or NewsAPI
        # For now, returning neutral sentiment
        sentiment_score = 50  # Neutral
        headlines = [
            f"Market analysis for {symbol}",
            "Crypto market shows mixed signals",
            "Traders watching key support levels"
        ]
        return sentiment_score, headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return None, []


def analyze_news_sentiment_warning(fear_greed_value, news_sentiment, signal_strength):
    """
    Analyze if news sentiment conflicts with trading signal
    Returns: (has_warning, warning_message, sentiment_status)
    """
    if fear_greed_value is None:
        return False, "", "unknown"
    
    # Extreme Fear (< 25) or Extreme Greed (> 75)
    if fear_greed_value < 25:
        sentiment_status = "extreme_fear"
        if signal_strength > 0:  # Bullish signal during extreme fear
            return True, "Extreme Fear in market (contrarian opportunity?)", sentiment_status
        else:
            return False, "", sentiment_status
    
    elif fear_greed_value > 75:
        sentiment_status = "extreme_greed"
        if signal_strength < 0:  # Bearish signal during extreme greed
            return True, "Extreme Greed in market (potential reversal?)", sentiment_status
        else:
            return False, "", sentiment_status
    
    # Fear (25-45)
    elif fear_greed_value < 45:
        sentiment_status = "fear"
        return False, "", sentiment_status
    
    # Greed (55-75)
    elif fear_greed_value > 55:
        sentiment_status = "greed"
        return False, "", sentiment_status
    
    # Neutral (45-55)
    else:
        sentiment_status = "neutral"
        return False, "", sentiment_status
