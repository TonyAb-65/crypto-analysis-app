"""
Main Application - Streamlit UI
Imports functions from modularized files
ENHANCED WITH COMMITTEE LEARNING SYSTEM (Step 4)
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import from modules
from database import (
    init_database, save_prediction, mark_prediction_for_trading,
    get_all_recent_predictions, save_trade_result, get_indicator_weights, 
    relearn_from_past_trades, DB_PATH
)
from utils import (
    backup_database, export_trades_to_csv
)
from news import (
    get_fear_greed_index, get_crypto_news_sentiment, analyze_news_sentiment_warning
)
from signals import (
    calculate_signal_strength, calculate_warning_signs, create_indicator_snapshot
)
from data_api import fetch_data, get_batch_data_binance, get_retry_session
from support_resistance import find_support_resistance_zones
from indicators import calculate_technical_indicators
from ml_model import train_improved_model
from consultants import run_consultant_meeting

# ==================== 🆕 COMMITTEE SYSTEM INTEGRATION ====================
try:
    from committee_meeting import CommitteeMeeting
    from committee_learning import CommitteeLearningSystem
    COMMITTEE_AVAILABLE = True
except ImportError:
    COMMITTEE_AVAILABLE = False
    print("⚠️ Committee system not available - using fallback consultant meeting")

# Initialize database
init_database()

# ==================== STREAMLIT PAGE CONFIGURATION ====================

st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

st.title("🤖 AI Trading Analysis Platform - ENHANCED WITH LEARNING COMMITTEE")
st.markdown("*Crypto, Forex, Metals + AI ML Predictions + 4-Consultant Committee + Automatic Learning*")

if 'binance_blocked' not in st.session_state:
    st.session_state.binance_blocked = False

if st.session_state.binance_blocked:
    st.info("ℹ️ **Note:** Binance API is blocked in your region. Using OKX and backup APIs instead.")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")

# ==================== 🆕 COMMITTEE INITIALIZATION ====================
if COMMITTEE_AVAILABLE:
    if 'committee' not in st.session_state:
        with st.spinner("🏛️ Initializing Committee System..."):
            st.session_state.committee = CommitteeMeeting(enable_learning=True)
        st.success("✅ Committee System Ready!")
    
    committee = st.session_state.committee
else:
    committee = None

with st.expander("💾 Database Information", expanded=False):
    st.info(f"""
    **Database Location:** `{DB_PATH}`
    
    **File Exists:** {'✅ Yes' if DB_PATH.exists() else '❌ No'}
    
    **Note:** All your trade history and predictions are saved to this database file.
    """)
    
    st.markdown("#### 🔧 Database Maintenance")
    col_db1, col_db2 = st.columns(2)
    
    with col_db1:
        if st.button("🗑️ Delete Bad Trade (ID 59)", help="Remove SOL trade with incorrect -$3,470 loss"):
            try:
                conn_fix = sqlite3.connect(str(DB_PATH))
                cursor_fix = conn_fix.cursor()
                
                # Check before
                cursor_fix.execute("SELECT SUM(profit_loss) FROM trade_results")
                total_before = cursor_fix.fetchone()[0]
                
                # Delete bad trade
                cursor_fix.execute("DELETE FROM trade_results WHERE id = 59")
                conn_fix.commit()
                
                # Check after
                cursor_fix.execute("SELECT SUM(profit_loss) FROM trade_results")
                total_after = cursor_fix.fetchone()[0]
                
                conn_fix.close()
                
                st.success(f"✅ Deleted! P/L: ${total_before:.2f} → ${total_after:.2f}")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col_db2:
        st.caption("Fixes incorrect SOL SHORT exit price ($6 → correct value)")

st.markdown("---")

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("⚙️ Configuration")

debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed API information")

# ==================== 🆕 COMMITTEE STATUS (SIDEBAR) ====================
if COMMITTEE_AVAILABLE and committee:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏛️ Committee Status")
    
    # Show committee consultant performance
    with st.sidebar.expander("📊 Consultant Rankings", expanded=False):
        rankings = committee.get_consultant_rankings()
        for rank_info in rankings:
            streak_icon = "🔥" if rank_info['streak'] > 0 else "❄️" if rank_info['streak'] < 0 else "➖"
            st.caption(f"#{rank_info['rank']} {rank_info['name']}: {rank_info['accuracy']:.1f}% "
                      f"({rank_info['weight']:.2f}x) {streak_icon}")
    
    # Reload button
    if st.sidebar.button("🔄 Reload Committee", help="Refresh consultant weights from database"):
        committee.reload_consultant_performance()
        st.sidebar.success("✅ Reloaded!")

# ==================== TOP MOVERS ====================
show_market_movers = st.sidebar.checkbox("📈 Show Top Movers", value=True,
                                        help="Display today's top gainers and losers")

@st.cache_data(ttl=300, show_spinner=False)
def get_market_movers():
    """Get top movers from popular cryptocurrencies"""
    popular_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']
    movers = []
    
    binance_failed = False
    for symbol_temp in popular_symbols:
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": f"{symbol_temp}USDT"}
            response = get_retry_session().get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                price_change_pct = float(data['priceChangePercent'])
                current_price_temp = float(data['lastPrice'])
                volume_temp = float(data['volume'])
                
                movers.append({
                    'Symbol': symbol_temp,
                    'Price': current_price_temp,
                    'Change %': price_change_pct,
                    'Volume': volume_temp
                })
            else:
                binance_failed = True
                break
        except:
            binance_failed = True
            break
    
    if binance_failed or len(movers) == 0:
        movers = []
        for symbol_temp in popular_symbols:
            try:
                url = "https://www.okx.com/api/v5/market/ticker"
                params = {"instId": f"{symbol_temp}-USDT"}
                response = get_retry_session().get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == '0' and len(data.get('data', [])) > 0:
                        ticker = data['data'][0]
                        current_price_temp = float(ticker['last'])
                        open_24h = float(ticker['open24h'])
                        price_change_pct = ((current_price_temp - open_24h) / open_24h) * 100
                        volume_temp = float(ticker['vol24h'])
                        
                        movers.append({
                            'Symbol': symbol_temp,
                            'Price': current_price_temp,
                            'Change %': price_change_pct,
                            'Volume': volume_temp
                        })
            except:
                continue
    
    if movers:
        df_movers = pd.DataFrame(movers)
        df_movers = df_movers.sort_values('Change %', ascending=False)
        return df_movers
    return None

if show_market_movers:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📈 Market Movers")
        
        with st.spinner("Loading market movers..."):
            movers_df = get_market_movers()
        
        if movers_df is not None and len(movers_df) > 0:
            st.markdown("#### 📈 Top Gainers")
            top_gainers = movers_df.head(3)
            for _, row in top_gainers.iterrows():
                delta = f"+{row['Change %']:.2f}%"
                st.metric(row['Symbol'], f"${row['Price']:,.2f}", delta)
            
            st.markdown("#### 📉 Top Losers")
            top_losers = movers_df.tail(3).sort_values('Change %')
            for _, row in top_losers.iterrows():
                delta = f"{row['Change %']:.2f}%"
                st.metric(row['Symbol'], f"${row['Price']:,.2f}", delta)
        else:
            st.warning("⚠️ Unable to load market movers")
            if debug_mode:
                st.error("Debug: movers_df returned None or empty")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Database Status")
try:
    db_exists = DB_PATH.exists()
    if db_exists:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trade_results")
        trade_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(timestamp) FROM predictions")
        last_pred = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(trade_date) FROM trade_results")
        last_trade = cursor.fetchone()[0]
        
        conn.close()
        
        st.sidebar.success(f"✅ Connected")
        st.sidebar.caption(f"📍 `{DB_PATH.name}`")
        st.sidebar.caption(f"📊 Predictions: {pred_count}")
        st.sidebar.caption(f"💰 Trades: {trade_count}")
        if last_pred:
            st.sidebar.caption(f"🕐 Last prediction: {last_pred[:16]}")
        if last_trade:
            st.sidebar.caption(f"🕐 Last trade: {last_trade[:16]}")
        
        # AI Learning button
        if trade_count > 0:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🧠 AI Learning")
            if st.sidebar.button("🔄 Relearn from Past Trades", help=f"Analyze {trade_count} completed trades to update indicator weights"):
                with st.spinner("Learning from past trades..."):
                    learned = relearn_from_past_trades()
                    st.sidebar.success(f"✅ Learned from {learned} trades!")
                    st.sidebar.info("🔄 Refresh page to see updated weights")
                    
                    # 🆕 Also reload committee if available
                    if COMMITTEE_AVAILABLE and committee:
                        committee.reload_consultant_performance()
                        st.sidebar.success("✅ Committee updated!")
        
        with st.sidebar.expander("🔍 Full Path", expanded=False):
            st.code(str(DB_PATH))
            if st.button("📋 Copy Path"):
                st.info("Path shown above - copy manually")
    else:
        st.sidebar.warning(f"⚠️ Database not found")
        st.sidebar.caption(f"Creating at: `{DB_PATH}`")
        init_database()
except Exception as e:
    st.sidebar.error(f"❌ Error")
    with st.sidebar.expander("Details", expanded=False):
        st.code(str(e))

st.sidebar.markdown("---")

# ==================== ASSET SELECTION ====================
asset_type = st.sidebar.selectbox(
    "📊 Select Asset Type",
    ["💰 Cryptocurrency", "🏆 Precious Metals", "💱 Forex", "🔍 Custom Search"],
    index=0
)

CRYPTO_SYMBOLS = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Binance Coin (BNB)": "BNB",
    "XRP": "XRP",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL",
    "Dogecoin (DOGE)": "DOGE",
    "Polygon (MATIC)": "MATIC",
    "Polkadot (DOT)": "DOT",
    "Avalanche (AVAX)": "AVAX"
}

PRECIOUS_METALS = {
    "Gold (XAU/USD)": "XAU/USD",
    "Silver (XAG/USD)": "XAG/USD"
}

FOREX_PAIRS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "USD/CHF": "USD/CHF",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "NZD/USD": "NZD/USD",
    "EUR/GBP": "EUR/GBP",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "AUD/JPY": "AUD/JPY",
    "EUR/CHF": "EUR/CHF",
    "AUD/CAD": "AUD/CAD",
    "AUD/NZD": "AUD/NZD",
    "CAD/JPY": "CAD/JPY"
}

if asset_type == "💰 Cryptocurrency":
    pair_display = st.sidebar.selectbox("Select Cryptocurrency", list(CRYPTO_SYMBOLS.keys()), index=0)
    symbol = CRYPTO_SYMBOLS[pair_display]
elif asset_type == "🏆 Precious Metals":
    pair_display = st.sidebar.selectbox("Select Metal", list(PRECIOUS_METALS.keys()), index=0)
    symbol = PRECIOUS_METALS[pair_display]
elif asset_type == "💱 Forex":
    pair_display = st.sidebar.selectbox("Select Forex Pair", list(FOREX_PAIRS.keys()), index=0)
    symbol = FOREX_PAIRS[pair_display]
else:
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "BTC").upper()
    pair_display = f"Custom: {custom_symbol}"
    symbol = custom_symbol

TIMEFRAMES = {
    "5 Minutes": {"limit": 100, "binance": "5m", "okx": "5m"},
    "15 Minutes": {"limit": 100, "binance": "15m", "okx": "15m"},
    "30 Minutes": {"limit": 100, "binance": "30m", "okx": "30m"},
    "1 Hour": {"limit": 100, "binance": "1h", "okx": "1H"},
    "4 Hours": {"limit": 100, "binance": "4h", "okx": "4H"},
    "1 Day": {"limit": 100, "binance": "1d", "okx": "1D"}
}

timeframe_name = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=3)
timeframe_config = TIMEFRAMES[timeframe_name]

auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False, 
                                   help="Automatically refresh data every 60 seconds")

if auto_refresh:
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    elapsed = time.time() - st.session_state.last_refresh_time
    
    if elapsed >= 60:
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    else:
        remaining = int(60 - elapsed)
        st.sidebar.info(f"⏱️ Next refresh in {remaining}s")
        time.sleep(1)
        st.rerun()

st.sidebar.markdown("### 🤖 AI Configuration")
prediction_periods = st.sidebar.slider("Prediction Periods", 1, 10, 5)
lookback_hours = st.sidebar.slider("Context Window (hours)", 4, 12, 6, 
                                   help="How many hours to look back for pattern analysis")

st.sidebar.markdown("### 📊 Technical Indicators")
use_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (14)", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

st.sidebar.markdown("#### 🆕 Advanced Indicators")
use_obv = st.sidebar.checkbox("OBV (Volume)", value=True, help="On-Balance Volume - tracks volume flow")
use_mfi = st.sidebar.checkbox("MFI (14)", value=True, help="Money Flow Index - volume-weighted RSI")
use_adx = st.sidebar.checkbox("ADX (14)", value=True, help="Average Directional Index - trend strength")
use_stoch = st.sidebar.checkbox("Stochastic", value=True, help="Stochastic Oscillator - momentum indicator")
use_cci = st.sidebar.checkbox("CCI (20)", value=True, help="Commodity Channel Index - cyclical trends")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 AI Learning System")
show_learning_dashboard = st.sidebar.checkbox("📊 Show Trades Table on Page", value=True,
                                              help="✅ Shows your tracked trades and predictions")

# ==================== AI LEARNING DASHBOARD (Sidebar) ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI Learning Progress")

try:
    conn_learn = sqlite3.connect(str(DB_PATH))
    cursor_learn = conn_learn.cursor()
    
    cursor_learn.execute("SELECT COUNT(*) FROM trade_results")
    total_closed = cursor_learn.fetchone()[0]
    
    if total_closed > 0:
        cursor_learn.execute("""
            SELECT 
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                AVG(profit_loss_pct) as avg_return,
                SUM(profit_loss) as total_pl
            FROM trade_results
        """)
        wins, avg_return, total_pl = cursor_learn.fetchone()
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        
        st.sidebar.metric("📊 Closed Trades", total_closed)
        st.sidebar.metric("🎯 Win Rate", f"{win_rate:.1f}%", 
                         delta="Good" if win_rate >= 55 else "Review" if win_rate >= 45 else "Poor")
        st.sidebar.metric("💰 Total P/L", f"${total_pl:.2f}",
                         delta=f"{avg_return:.2f}% avg")
        
        st.sidebar.markdown("**🏆 Top Indicators:**")
        cursor_learn.execute("""
            SELECT indicator_name, accuracy_rate, weight_multiplier
            FROM indicator_accuracy
            WHERE correct_count + wrong_count > 0
            ORDER BY accuracy_rate DESC
            LIMIT 3
        """)
        
        top_indicators = cursor_learn.fetchall()
        if top_indicators:
            for ind_name, accuracy, weight in top_indicators:
                emoji = "🟢" if accuracy > 0.6 else "🟡" if accuracy > 0.5 else "🔴"
                st.sidebar.caption(f"{emoji} {ind_name}: {accuracy*100:.0f}% ({weight:.1f}x)")
        else:
            st.sidebar.caption("⚪ No indicator data yet")
    
    else:
        st.sidebar.info("ℹ️ No closed trades yet")
        st.sidebar.caption("Close trades to see AI learning!")
    
    conn_learn.close()

except Exception as e:
    st.sidebar.error("❌ Error loading learning stats")
    if debug_mode:
        st.sidebar.code(str(e))

st.sidebar.markdown("---")

# ==================== DATABASE MANAGEMENT ====================
st.sidebar.markdown("### 💾 Database Management")

col_backup, col_export = st.sidebar.columns(2)

with col_backup:
    if st.button("🔄 Backup"):
        with st.spinner("Creating backup..."):
            success, message = backup_database()
            if success:
                st.success("✅ Backed up!")
            else:
                st.error(f"❌ {message}")

with col_export:
    if st.button("📤 Export CSV"):
        with st.spinner("Exporting..."):
            success, filepath = export_trades_to_csv()
            if success:
                st.success(f"✅ Exported!")
                st.caption(f"📁 {filepath.name}")
            else:
                st.error("❌ Export failed")

# ==================== MAIN DATA FETCHING ====================

with st.spinner(f"🔄 Fetching {pair_display} data..."):
    df, data_source = fetch_data(symbol, asset_type, timeframe_config)

if df is not None and len(df) > 0:
    df = calculate_technical_indicators(df)
    
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
    st.markdown(f"### 📊 {pair_display} - Real-Time Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
    with col2:
        st.metric("24h High", f"${df['high'].tail(24).max():,.2f}" if len(df) >= 24 else "N/A")
    with col3:
        st.metric("24h Low", f"${df['low'].tail(24).min():,.2f}" if len(df) >= 24 else "N/A")
    with col4:
        st.metric("Data Source", data_source)
    
    st.markdown("---")
    
    # News Sentiment
    st.markdown("### 📰 Market Intelligence Check")
    
    with st.spinner("🔄 Fetching market sentiment..."):
        fear_greed_value, fear_greed_class = get_fear_greed_index()
        news_sentiment, news_headlines = get_crypto_news_sentiment(symbol)
    
    col_sentiment1, col_sentiment2 = st.columns(2)
    
    with col_sentiment1:
        if fear_greed_value:
            emoji = "😱" if fear_greed_value < 25 else "😰" if fear_greed_value < 45 else "😐" if fear_greed_value < 55 else "😃" if fear_greed_value < 75 else "🤑"
            st.metric("Fear & Greed Index", f"{emoji} {fear_greed_value}/100", fear_greed_class)
        else:
            st.warning("⚠️ Fear & Greed data unavailable")
    
    with col_sentiment2:
        if news_sentiment:
            emoji = "🔴" if news_sentiment < 40 else "🟡" if news_sentiment < 60 else "🟢"
            st.metric("News Sentiment", f"{emoji} {news_sentiment:.0f}/100",
                     "Bearish" if news_sentiment < 40 else "Neutral" if news_sentiment < 60 else "Bullish")
        else:
            st.info("ℹ️ News sentiment unavailable")
    
    if news_headlines and len(news_headlines) > 0:
        with st.expander("📰 Recent Headlines", expanded=False):
            for i, headline in enumerate(news_headlines, 1):
                st.caption(f"{i}. {headline}")
    
    st.markdown("---")
    
    # ML Predictions
    st.markdown("### 🤖 AI Predictions")
    
    with st.spinner("🧠 Training AI models..."):
        predictions, features, confidence, rsi_insights = train_improved_model(
            df, 
            lookback=lookback_hours,
            prediction_periods=prediction_periods
        )
    
    if predictions and len(predictions) > 0:
        pred_change = ((predictions[-1] - current_price) / current_price) * 100
        
        indicator_snapshot = create_indicator_snapshot(df)
        
        # Calculate signals
        raw_signal_strength = calculate_signal_strength(df, warning_details=None)
        
        news_warning_data = None
        if fear_greed_value is not None:
            has_news_warning, news_msg, sentiment_status = analyze_news_sentiment_warning(
                fear_greed_value, news_sentiment, raw_signal_strength
            )
            news_warning_data = {
                'has_warning': has_news_warning,
                'warning_message': news_msg,
                'sentiment_status': sentiment_status
            }
        
        warning_count, warning_details = calculate_warning_signs(
            df, raw_signal_strength, news_warning_data
        )
        
        final_signal_strength = calculate_signal_strength(df, warning_details)
        
        adjusted_confidence = confidence
        if warning_count >= 1:
            adjusted_confidence = confidence * (1 - (warning_count * 0.15))
            adjusted_confidence = max(adjusted_confidence, 30.0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Prediction", f"${predictions[-1]:,.2f}", f"{pred_change:+.2f}%")
        
        with col2:
            confidence_color = "🟢" if adjusted_confidence > 80 else "🟡" if adjusted_confidence > 60 else "🔴"
            st.metric("Confidence", f"{confidence_color} {adjusted_confidence:.1f}%", 
                     "High" if adjusted_confidence > 80 else "Medium" if adjusted_confidence > 60 else "Low")
        
        with col3:
            signal_emoji = "🟢" if final_signal_strength > 0 else "🔴" if final_signal_strength < 0 else "⚪"
            st.metric("Signal", f"{signal_emoji} {abs(final_signal_strength)}/10",
                     "Bullish" if final_signal_strength > 0 else "Bearish" if final_signal_strength < 0 else "Neutral")
        
        # ==================== WARNING SIGNS DISPLAY ====================
        if warning_count > 0:
            st.markdown("---")
            st.markdown("### ⚠️ Warning Signs Detected")
            
            # Display warnings in 4 columns as discussed
            warning_cols = st.columns(4)
            col_idx = 0
            
            # Safety check: ensure warning_details is a dictionary
            if isinstance(warning_details, dict):
                for warning_key, warning_info in warning_details.items():
                    # Check if warning_info is a dict and has required keys
                    if isinstance(warning_info, dict) and warning_info.get('active'):
                        with warning_cols[col_idx % 4]:
                            severity_emoji = "🔴" if warning_info.get('severity') == 'high' else "🟡"
                            category = warning_info.get('category', 'Warning')
                            message = warning_info.get('message', 'Check indicators')
                            st.warning(f"{severity_emoji} **{category}**")
                            st.caption(message)
                        col_idx += 1
            
            st.caption(f"⚠️ Total warnings: {warning_count} - Confidence reduced by {warning_count * 15}%")
        
        st.markdown("---")
        
        # ==================== 🆕 COMMITTEE MEETING (ENHANCED) ====================
        st.markdown("### 🏛️ Committee Trading Recommendation")
        st.markdown("*4-Consultant AI Committee with Learning System*")
        
        # Prepare data for committee
        # Get indicators dict
        indicators_dict = df.iloc[-1].to_dict() if len(df) > 0 else {}
        
        # Prepare news data for C2
        committee_news_data = None
        if fear_greed_value or news_sentiment:
            committee_news_data = {
                'fear_greed_index': fear_greed_value,
                'sentiment_score': (news_sentiment / 100 - 0.5) * 2 if news_sentiment else 0,  # Convert to -1 to +1
                'social_sentiment': 'bullish' if news_sentiment and news_sentiment > 60 else 'bearish' if news_sentiment and news_sentiment < 40 else 'neutral'
            }
        
        # Prepare risk metrics for C3
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
        risk_metrics = None
        if atr:
            volatility_pct = (atr / current_price) * 100
            risk_metrics = {
                'volatility_pct': volatility_pct,
                'risk_reward_ratio': 2.5  # Can be calculated dynamically
            }
        
        if COMMITTEE_AVAILABLE and committee:
            # Use new committee system
            with st.spinner("🏛️ Committee meeting in progress..."):
                meeting_result = committee.hold_meeting(
                    data=df,
                    indicators=indicators_dict,
                    signals={},  # Can add signals here if needed
                    news_data=committee_news_data,
                    risk_metrics=risk_metrics,
                    patterns=None,  # Can add patterns here
                    symbol=symbol,
                    current_price=current_price,
                    market_type='crypto' if asset_type == "💰 Cryptocurrency" else 'forex' if asset_type == "💱 Forex" else 'metals'
                )
            
            # Display committee result
            if meeting_result['final_decision'] == 'HOLD':
                st.warning(f"⚪ **COMMITTEE DECISION: HOLD**")
                st.info("💡 Committee advises waiting for better setup")
            elif meeting_result['final_decision'] == 'BUY':
                st.success(f"🟢 **COMMITTEE DECISION: {meeting_result['decision_strength']} BUY**")
                st.caption(f"Consensus: {meeting_result['consensus_percentage']}")
            else:
                st.error(f"🔴 **COMMITTEE DECISION: {meeting_result['decision_strength']} SELL/SHORT**")
                st.caption(f"Consensus: {meeting_result['consensus_percentage']}")
            
            # Show conflicts if any
            if meeting_result['has_conflict']:
                st.warning("⚠️ **Consultants Disagree:**")
                for conflict in meeting_result['conflicts']:
                    st.caption(f"• {conflict}")
            
            # Show detailed committee summary in expander
            with st.expander("📊 View Full Committee Discussion", expanded=False):
                st.text(meeting_result['summary'])
            
            # Store decision_id for later
            if 'decision_id' in meeting_result:
                st.session_state['last_decision_id'] = meeting_result['decision_id']
            
            # Convert committee result to format compatible with existing code
            meeting_result_legacy = {
                'position': meeting_result['final_decision'] if meeting_result['final_decision'] != 'HOLD' else 'NEUTRAL',
                'reasoning': meeting_result['summary_short'],
                'entry': current_price,
                'target': current_price * 1.03 if meeting_result['final_decision'] == 'BUY' else current_price * 0.97 if meeting_result['final_decision'] == 'SELL' else current_price,
                'stop_loss': current_price * 0.99 if meeting_result['final_decision'] == 'BUY' else current_price * 1.01 if meeting_result['final_decision'] == 'SELL' else current_price
            }
            
        else:
            # Fallback to old consultant meeting
            st.info("ℹ️ Using legacy consultant system (Committee not available)")
            meeting_result_legacy = run_consultant_meeting(symbol, asset_type, current_price, warning_details)
        
        # Use legacy variable name for rest of code
        meeting_result = meeting_result_legacy
        
        # Display recommendation (if not NEUTRAL)
        if meeting_result['position'] != 'NEUTRAL':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Price", f"${meeting_result['entry']:.2f}")
            with col2:
                change_pct = ((meeting_result['target']/meeting_result['entry']-1)*100)
                st.metric("Target Price", f"${meeting_result['target']:.2f}", f"{change_pct:+.1f}%")
            with col3:
                stop_change = ((meeting_result['stop_loss']/meeting_result['entry']-1)*100)
                st.metric("Stop Loss", f"${meeting_result['stop_loss']:.2f}", f"{stop_change:.1f}%")
            
            # ==================== SAVE PREDICTION TO DATABASE ====================
            st.markdown("---")
            st.markdown("### 💾 Track This Trade")
            
            col_save1, col_save2, col_save3 = st.columns([2, 1, 1])
            
            with col_save1:
                st.info("💡 Save this prediction to track performance and train AI")
            
            with col_save2:
                if st.button("💾 Save Prediction", use_container_width=True):
                    with st.spinner("Saving..."):
                        pred_id = save_prediction(
                            asset_type=asset_type,
                            pair=symbol,
                            timeframe=timeframe_name,
                            current_price=current_price,
                            predicted_price=predictions[-1],
                            prediction_horizon=prediction_periods,
                            confidence=adjusted_confidence,
                            signal_strength=final_signal_strength,
                            features=str(features),
                            status='analysis_only',
                            actual_entry_price=meeting_result['entry'] if meeting_result['position'] != 'NEUTRAL' else None,
                            entry_timestamp=None,
                            indicator_snapshot=str(indicator_snapshot),
                            position_type=meeting_result['position'],
                            target_price=meeting_result['target'] if meeting_result['position'] != 'NEUTRAL' else None,
                            stop_loss=meeting_result['stop_loss'] if meeting_result['position'] != 'NEUTRAL' else None,
                            committee_position=meeting_result['position'],
                            committee_confidence=adjusted_confidence,
                            committee_reasoning=meeting_result['reasoning']
                        )
                        if pred_id:
                            st.success(f"✅ Saved! ID: {pred_id}")
                        else:
                            st.error("❌ Failed to save")
            
            with col_save3:
                if st.button("📈 Mark for Trading", use_container_width=True):
                    # FIRST: Save current analysis as new prediction with TODAY's date
                    with st.spinner("Creating trade entry..."):
                        pred_id = save_prediction(
                            asset_type=asset_type,
                            pair=symbol,
                            timeframe=timeframe_name,
                            current_price=current_price,
                            predicted_price=predictions[-1],
                            prediction_horizon=prediction_periods,
                            confidence=adjusted_confidence,
                            signal_strength=final_signal_strength,
                            features=str(features),
                            status='analysis_only',
                            actual_entry_price=meeting_result['entry'] if meeting_result['position'] != 'NEUTRAL' else None,
                            entry_timestamp=None,
                            indicator_snapshot=str(indicator_snapshot),
                            position_type=meeting_result['position'],
                            target_price=meeting_result['target'] if meeting_result['position'] != 'NEUTRAL' else None,
                            stop_loss=meeting_result['stop_loss'] if meeting_result['position'] != 'NEUTRAL' else None,
                            committee_position=meeting_result['position'],
                            committee_confidence=adjusted_confidence,
                            committee_reasoning=meeting_result['reasoning']
                        )
                        
                        if pred_id:
                            # Now get the fresh prediction data
                            conn = sqlite3.connect(str(DB_PATH))
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT id, current_price, position_type, target_price, stop_loss 
                                FROM predictions 
                                WHERE id = ?
                            """, (pred_id,))
                            result = cursor.fetchone()
                            conn.close()
                            
                            if result:
                                pred_id, curr_price, pos_type, target, stop = result
                                # Show entry price form
                                st.session_state[f'marking_trade_{pred_id}'] = True
                        else:
                            st.error("❌ Failed to create prediction")
            
            # Entry price form (if marking for trading)
            if 'marking_trade' in str(st.session_state):
                for key in list(st.session_state.keys()):
                    if key.startswith('marking_trade_') and st.session_state[key]:
                        pred_id = int(key.replace('marking_trade_', ''))
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT id, current_price, position_type, target_price, stop_loss FROM predictions 
                            WHERE id = ?
                        """, (pred_id,))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            pred_id_form, curr_price, pos_type, target, stop = result
                            
                            with st.form(key=f"entry_form_{pred_id}"):
                                st.markdown("### 📝 Enter Trade Details")
                                
                                actual_entry = st.number_input(
                                    "Actual Entry Price ($)",
                                    min_value=0.0,
                                    value=float(curr_price),
                                    step=0.01,
                                    help="Enter the actual price you entered the trade at",
                                    key=f"entry_input_{pred_id}"
                                )
                                
                                col_e1, col_e2 = st.columns(2)
                                with col_e1:
                                    if st.form_submit_button("✅ Confirm Entry", use_container_width=True):
                                        mark_prediction_for_trading(
                                            prediction_id=pred_id,
                                            actual_entry_price=actual_entry,
                                            entry_timestamp=datetime.now().isoformat(),
                                            position_type=pos_type,
                                            target_price=target,
                                            stop_loss=stop
                                        )
                                        
                                        # 🆕 Link to committee decision if available
                                        if COMMITTEE_AVAILABLE and 'last_decision_id' in st.session_state:
                                            try:
                                                committee.learning_system.link_decision_to_trade(
                                                    st.session_state['last_decision_id'],
                                                    pred_id
                                                )
                                            except:
                                                pass  # Fail silently if linking fails
                                        
                                        st.success(f"✅ Trade marked! Entry: ${actual_entry:.2f}")
                                        st.session_state[f'marking_trade_{pred_id}'] = False
                                        time.sleep(1)
                                        st.rerun()
                                
                                with col_e2:
                                    if st.form_submit_button("❌ Cancel", use_container_width=True):
                                        st.session_state[f'marking_trade_{pred_id}'] = False
                                        st.rerun()
        
        # ==================== SUPPORT/RESISTANCE ZONES ====================
        st.markdown("---")
        st.markdown("### 🎯 Support & Resistance Zones")
        
        # Convert timeframe to Twelve Data format
        timeframe_map = {
            "5 Minutes": "5min",
            "15 Minutes": "15min",
            "30 Minutes": "30min",
            "1 Hour": "1h",
            "4 Hours": "4h",
            "1 Day": "1day"
        }
        twelvedata_interval = timeframe_map.get(timeframe_name, "1h")
        
        # Format symbol for Twelve Data
        # Crypto: BTC/USD, ETH/USD (add /USD if not present)
        # Forex: EUR/USD (already correct format)
        if asset_type == "💰 Cryptocurrency":
            twelvedata_symbol = f"{symbol}/USD" if '/' not in symbol else symbol
        else:
            twelvedata_symbol = symbol
        
        # Fetch S/R from Twelve Data API
        sr_zones = find_support_resistance_zones(df, symbol=twelvedata_symbol, interval=twelvedata_interval)
        support_zones = sr_zones.get('support', [])
        resistance_zones = sr_zones.get('resistance', [])
        
        if support_zones or resistance_zones:
            col_sr1, col_sr2 = st.columns(2)
            
            with col_sr1:
                st.markdown("**🟢 Support Levels:**")
                if support_zones and len(support_zones) > 0:
                    for i, zone in enumerate(support_zones[:3], 1):
                        try:
                            # Handle dictionary format from find_support_resistance_zones
                            if isinstance(zone, dict):
                                level = zone['price']
                                touches = zone.get('touches', 1)
                                strength = zone.get('strength', 'MEDIUM')
                                status = zone.get('status', 'INTACT')
                            elif isinstance(zone, (tuple, list)) and len(zone) >= 2:
                                level, touches = zone[0], zone[1]
                                strength = 'STRONG' if touches >= 3 else 'MEDIUM'
                                status = 'INTACT'
                            else:
                                level = float(zone)
                                touches = 1
                                strength = 'MEDIUM'
                                status = 'INTACT'
                            
                            distance_pct = ((current_price - level) / current_price) * 100
                            
                            # Show FLIPPED status for role reversals
                            if strength == 'FLIPPED':
                                status_emoji = "🔄"
                                status_text = "FLIPPED (was resistance)"
                            else:
                                status_emoji = "🟢"
                                status_text = f"{strength}"
                            
                            st.caption(f"{i}. {status_emoji} ${level:.2f} ({distance_pct:+.1f}%) - {status_text} ({touches} touches)")
                        except Exception as e:
                            if debug_mode:
                                st.error(f"Error displaying support zone: {e}")
                else:
                    st.caption("No clear support zones")
            
            with col_sr2:
                st.markdown("**🔴 Resistance Levels:**")
                if resistance_zones and len(resistance_zones) > 0:
                    for i, zone in enumerate(resistance_zones[:3], 1):
                        try:
                            # Handle dictionary format from find_support_resistance_zones
                            if isinstance(zone, dict):
                                level = zone['price']
                                touches = zone.get('touches', 1)
                                strength = zone.get('strength', 'MEDIUM')
                                status = zone.get('status', 'INTACT')
                            elif isinstance(zone, (tuple, list)) and len(zone) >= 2:
                                level, touches = zone[0], zone[1]
                                strength = 'STRONG' if touches >= 3 else 'MEDIUM'
                                status = 'INTACT'
                            else:
                                level = float(zone)
                                touches = 1
                                strength = 'MEDIUM'
                                status = 'INTACT'
                            
                            distance_pct = ((level - current_price) / current_price) * 100
                            
                            # Show FLIPPED status for role reversals
                            if strength == 'FLIPPED':
                                status_emoji = "🔄"
                                status_text = "FLIPPED (was support)"
                            else:
                                status_emoji = "🔴"
                                status_text = f"{strength}"
                            
                            st.caption(f"{i}. {status_emoji} ${level:.2f} ({distance_pct:+.1f}%) - {status_text} ({touches} touches)")
                        except Exception as e:
                            if debug_mode:
                                st.error(f"Error displaying resistance zone: {e}")
                else:
                    st.caption("No clear resistance zones")
        else:
            st.info("ℹ️ No clear support/resistance zones detected")
        
        # ==================== CHART WITH S/R ZONES ====================
        st.markdown("---")
        st.markdown("### 📈 Technical Chart with S/R Zones")
        
        # Debug S/R zones
        if debug_mode:
            st.write(f"Support zones: {support_zones}")
            st.write(f"Resistance zones: {resistance_zones}")
        
        # Calculate number of rows needed for chart
        chart_rows = 1  # Main price chart always
        if use_rsi and 'rsi' in df.columns:
            chart_rows += 1
        if use_macd and 'macd' in df.columns:
            chart_rows += 1
        
        # Create row heights dynamically
        row_heights = [0.6] + [0.2] * (chart_rows - 1) if chart_rows > 1 else [1.0]
        
        fig = make_subplots(
            rows=chart_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(['Price'] + 
                           (['RSI'] if use_rsi and 'rsi' in df.columns else []) +
                           (['MACD'] if use_macd and 'macd' in df.columns else []))
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Support/Resistance zones
        if support_zones and len(support_zones) > 0:
            for zone in support_zones[:3]:
                try:
                    # Handle dictionary format from find_support_resistance_zones
                    if isinstance(zone, dict):
                        level = zone['price']
                        strength = zone.get('strength', 'MEDIUM')
                        # Use orange for flipped levels, blue for normal support
                        line_color = "orange" if strength == 'FLIPPED' else "blue"
                        annotation = f"Support ${level:.2f}" if strength != 'FLIPPED' else f"Support ${level:.2f} (FLIPPED)"
                    elif isinstance(zone, (tuple, list)) and len(zone) >= 2:
                        level = zone[0]
                        line_color = "blue"
                        annotation = f"Support ${level:.2f}"
                    else:
                        level = float(zone)
                        line_color = "blue"
                        annotation = f"Support ${level:.2f}"
                    
                    fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color=line_color,
                        line_width=2,
                        opacity=0.6,
                        annotation_text=annotation,
                        annotation_position="right",
                        row=1, col=1
                    )
                except Exception as e:
                    if debug_mode:
                        st.write(f"Error plotting support: {e}")
        
        if resistance_zones and len(resistance_zones) > 0:
            for zone in resistance_zones[:3]:
                try:
                    # Handle dictionary format from find_support_resistance_zones
                    if isinstance(zone, dict):
                        level = zone['price']
                        strength = zone.get('strength', 'MEDIUM')
                        # Use orange for flipped levels, red for normal resistance
                        line_color = "orange" if strength == 'FLIPPED' else "red"
                        annotation = f"Resistance ${level:.2f}" if strength != 'FLIPPED' else f"Resistance ${level:.2f} (FLIPPED)"
                    elif isinstance(zone, (tuple, list)) and len(zone) >= 2:
                        level = zone[0]
                        line_color = "red"
                        annotation = f"Resistance ${level:.2f}"
                    else:
                        level = float(zone)
                        line_color = "red"
                        annotation = f"Resistance ${level:.2f}"
                    
                    fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color=line_color,
                        line_width=2,
                        opacity=0.6,
                        annotation_text=annotation,
                        annotation_position="right",
                        row=1, col=1
                    )
                except Exception as e:
                    if debug_mode:
                        st.write(f"Error plotting resistance: {e}")
        
        # Add Entry/Target/Stop Loss if not NEUTRAL
        if meeting_result['position'] != 'NEUTRAL':
            fig.add_hline(
                y=meeting_result['entry'],
                line_dash="solid",
                line_color="blue",
                line_width=2,
                annotation_text=f"Entry ${meeting_result['entry']:.2f}",
                annotation_position="left",
                row=1, col=1
            )
            
            fig.add_hline(
                y=meeting_result['target'],
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"Target ${meeting_result['target']:.2f}",
                annotation_position="left",
                row=1, col=1
            )
            
            fig.add_hline(
                y=meeting_result['stop_loss'],
                line_dash="solid",
                line_color="red",
                line_width=2,
                annotation_text=f"Stop ${meeting_result['stop_loss']:.2f}",
                annotation_position="left",
                row=1, col=1
            )
        
        # Add indicators if selected
        if use_sma and 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', 
                                     line=dict(color='orange', width=1)), row=1, col=1)
        if use_sma and 'sma_50' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', 
                                     line=dict(color='blue', width=1)), row=1, col=1)
        
        if use_bb and 'bb_upper' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper',
                                     line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower',
                                     line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
        
        # RSI in row 2 if enabled
        current_row = 2
        if use_rsi and 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple')),
                row=current_row, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            current_row += 1
        
        # MACD in next row if enabled
        if use_macd and 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='red')),
                row=current_row, col=1
            )
            # Add MACD histogram if available
            if 'macd_hist' in df.columns:
                colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
                fig.add_trace(
                    go.Bar(x=df['timestamp'], y=df['macd_hist'], name='Histogram', marker_color=colors),
                    row=current_row, col=1
                )
            current_row += 1
        
        fig.update_layout(
            title=f"{pair_display} - {timeframe_name}",
            xaxis_title="Time",
            yaxis_title="Price",
            height=800,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================== ADVANCED INDICATORS DISPLAY ====================
        st.markdown("---")
        st.markdown("### 🆕 Advanced Technical Indicators")
        
        if debug_mode:
            st.write(f"Debug - use_obv: {use_obv}, use_mfi: {use_mfi}, use_adx: {use_adx}, use_stoch: {use_stoch}, use_cci: {use_cci}")
            st.write(f"Debug - Columns in df: {df.columns.tolist()}")
        
        indicator_cols = st.columns(3)
        col_idx = 0
        indicators_displayed = 0
        
        if use_obv and 'obv' in df.columns:
            with indicator_cols[col_idx % 3]:
                obv_current = df['obv'].iloc[-1]
                obv_prev = df['obv'].iloc[-5] if len(df) > 5 else obv_current
                obv_change = obv_current - obv_prev
                
                if obv_current < 0:
                    pressure_type = "Selling"
                else:
                    pressure_type = "Buying"
                
                if obv_change > 0:
                    if obv_current < 0:
                        momentum = "Decreasing"
                        momentum_emoji = "📊"
                        trend_color = "normal"
                    else:
                        momentum = "Increasing"
                        momentum_emoji = "📈"
                        trend_color = "normal"
                elif obv_change < 0:
                    if obv_current < 0:
                        momentum = "Increasing"
                        momentum_emoji = "📉"
                        trend_color = "inverse"
                    else:
                        momentum = "Decreasing"
                        momentum_emoji = "📊"
                        trend_color = "inverse"
                else:
                    momentum = "Flat"
                    momentum_emoji = "➡️"
                    trend_color = "off"
                
                obv_status = f"{momentum_emoji} {pressure_type} - {momentum}"
                
                st.metric("OBV (Volume Flow)", 
                         f"{obv_current:,.0f}",
                         obv_status,
                         delta_color=trend_color)
                st.caption("Tracks cumulative buying/selling pressure")
            col_idx += 1
            indicators_displayed += 1
        
        if use_mfi and 'mfi' in df.columns:
            with indicator_cols[col_idx % 3]:
                mfi_current = df['mfi'].iloc[-1]
                mfi_status = "🔴 Overbought" if mfi_current > 80 else "🟢 Oversold" if mfi_current < 20 else "⚪ Neutral"
                
                st.metric("MFI (Money Flow)", 
                         f"{mfi_current:.1f}",
                         mfi_status)
                st.caption("Volume-weighted RSI")
            col_idx += 1
            indicators_displayed += 1
        
        if use_adx and 'adx' in df.columns:
            with indicator_cols[col_idx % 3]:
                adx_current = df['adx'].iloc[-1]
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]
                
                trend_strength = "💪 Strong" if adx_current > 25 else "😐 Weak"
                trend_dir = "🟢 Up" if plus_di > minus_di else "🔴 Down"
                
                st.metric("ADX (Trend Strength)", 
                         f"{adx_current:.1f}",
                         f"{trend_strength} | {trend_dir}")
                st.caption(f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}")
            col_idx += 1
            indicators_displayed += 1
        
        if use_stoch and 'stoch_k' in df.columns:
            with indicator_cols[col_idx % 3]:
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                stoch_status = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "⚪ Neutral"
                
                st.metric("Stochastic", 
                         f"{stoch_k:.1f}",
                         stoch_status)
                st.caption(f"%K: {stoch_k:.1f} | %D: {stoch_d:.1f}")
            col_idx += 1
            indicators_displayed += 1
        
        if use_cci and 'cci' in df.columns:
            with indicator_cols[col_idx % 3]:
                cci_current = df['cci'].iloc[-1]
                cci_status = "🔴 Overbought" if cci_current > 100 else "🟢 Oversold" if cci_current < -100 else "⚪ Neutral"
                
                st.metric("CCI (Cyclical)", 
                         f"{cci_current:.1f}",
                         cci_status)
                st.caption("Commodity Channel Index")
            col_idx += 1
            indicators_displayed += 1
        
        if debug_mode:
            st.write(f"Debug - Total indicators displayed: {indicators_displayed}")
        
        if indicators_displayed == 0:
            st.info("👆 Enable indicators in the sidebar to see advanced analysis")
        
        # ==================== TRADE TRACKING TABLE ====================
        if show_learning_dashboard:
            st.markdown("---")
            st.markdown("## 📊 Trade Tracking & Learning Dashboard")
            
            # Get all predictions
            all_predictions_df = get_all_recent_predictions(limit=200)
            
            if all_predictions_df is not None and len(all_predictions_df) > 0:
                
                # ==================== ACTIVE TRADES MONITORING SECTION ====================
                # Import active trade monitor (gracefully handle if module not yet deployed)
                try:
                    from active_trade_monitor import (
                        get_active_trades_for_monitoring,
                        calculate_trade_progress,
                        calculate_profit_pct,
                        get_exit_recommendation,
                        get_next_check_countdown,
                        format_countdown
                    )
                    
                    # Get active trades for monitoring
                    active_trades_df = get_active_trades_for_monitoring()
                    
                    if active_trades_df is not None and len(active_trades_df) > 0:
                        st.markdown("### 🎯 Active Trade Monitor")
                        st.caption("Auto-refreshes every 15 minutes with trend analysis")
                        
                        # Show countdown to next check
                        countdown_seconds = get_next_check_countdown()
                        st.info(f"⏱️ Next analysis in: {format_countdown(countdown_seconds)}")
                        
                        # Display each active trade as a card
                        for idx, trade in active_trades_df.iterrows():
                            # Fetch current price for this pair
                            from data_api import fetch_data
                            
                            symbol_base = trade['symbol'].replace('/USD', '').replace('-USD', '')
                            current_df, source = fetch_data(symbol_base, "💰 Cryptocurrency", {"binance": "1h", "okx": "1H", "limit": 100})
                            
                            if current_df is not None and len(current_df) > 0:
                                current_price = current_df['close'].iloc[-1]
                                
                                # Calculate metrics
                                progress_pct = calculate_trade_progress(
                                    trade['entry_price'],
                                    current_price,
                                    trade['target_price'],
                                    trade['position_type']
                                )
                                
                                profit_pct = calculate_profit_pct(
                                    trade['entry_price'],
                                    current_price,
                                    trade['position_type']
                                )
                                
                                # Get recommendation
                                action, message, alert_type = get_exit_recommendation(
                                    progress_pct,
                                    trade['position_type'],
                                    current_df
                                )
                                
                                # Display card
                                with st.container():
                                    # Card border color based on action
                                    if action == "EXIT":
                                        card_color = "🔴"
                                    else:
                                        card_color = "🟢"
                                    
                                    st.markdown(f"### {card_color} {trade['symbol']} - {trade['position_type']}")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Entry", f"${trade['entry_price']:.4f}")
                                        st.metric("Current", f"${current_price:.4f}")
                                    
                                    with col2:
                                        st.metric("Target", f"${trade['target_price']:.4f}")
                                        st.metric("Progress", f"{progress_pct:.1f}%")
                                    
                                    with col3:
                                        profit_color = "normal" if profit_pct >= 0 else "inverse"
                                        st.metric("Profit", f"{profit_pct:+.2f}%", delta=None)
                                        
                                        # Action button
                                        if action == "EXIT":
                                            st.error(f"**⚠️ RECOMMENDATION: EXIT**")
                                        else:
                                            st.success(f"**💪 RECOMMENDATION: HOLD**")
                                    
                                    # Show detailed message
                                    if alert_type == "success":
                                        st.success(message)
                                    elif alert_type == "warning":
                                        st.warning(message)
                                    elif alert_type == "error":
                                        st.error(message)
                                    else:
                                        st.info(message)
                                    
                                    st.markdown("---")
                        
                        # Auto-refresh every 15 minutes
                        if countdown_seconds <= 5:  # Refresh when countdown near zero
                            time.sleep(2)
                            st.rerun()
                
                except ImportError:
                    # Module not yet deployed - show placeholder
                    st.info("🔄 Active Trade Monitor will be available after deploying active_trade_monitor.py")
                
                # ==================== REST OF TRADE TRACKING ====================
                st.markdown("### 🎯 Recent Predictions & Trades")
                
                # Filter options
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    filter_status = st.selectbox(
                        "Filter by Status",
                        ["Active Trades", "All", "For Trading", "Analysis Only", "Closed"],
                        index=0,
                        help="Active Trades = For Trading + Analysis Only (not closed)"
                    )
                
                with col_filter2:
                    # Get unique symbols from DataFrame
                    unique_symbols = sorted(all_predictions_df['pair'].unique().tolist())
                    filter_symbol = st.selectbox(
                        "Filter by Symbol",
                        ["All"] + unique_symbols,
                        index=0
                    )
                
                with col_filter3:
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Newest First", "Oldest First", "Highest Confidence", "Largest Signal"],
                        index=0
                    )
                
                # Apply filters on DataFrame
                filtered_df = all_predictions_df.copy()
                
                if filter_status != "All":
                    if filter_status == "Active Trades":
                        # Show only will_trade (marked for trading)
                        filtered_df = filtered_df[filtered_df['status'] == 'will_trade']
                    elif filter_status == "For Trading":
                        filtered_df = filtered_df[filtered_df['status'] == 'will_trade']
                    elif filter_status == "Analysis Only":
                        filtered_df = filtered_df[filtered_df['status'] == 'analysis_only']
                    elif filter_status == "Closed":
                        filtered_df = filtered_df[filtered_df['status'] == 'completed']
                
                if filter_symbol != "All":
                    filtered_df = filtered_df[filtered_df['pair'] == filter_symbol]
                
                # Sort
                if sort_by == "Newest First":
                    filtered_df = filtered_df.sort_values('timestamp', ascending=False)
                elif sort_by == "Oldest First":
                    filtered_df = filtered_df.sort_values('timestamp', ascending=True)
                elif sort_by == "Highest Confidence":
                    filtered_df = filtered_df.sort_values('confidence', ascending=False)
                elif sort_by == "Largest Signal":
                    filtered_df['abs_signal'] = filtered_df['signal_strength'].abs()
                    filtered_df = filtered_df.sort_values('abs_signal', ascending=False)
                
                # Display predictions in a table
                if len(filtered_df) > 0:
                    # Show count
                    st.caption(f"📊 Showing {min(len(filtered_df), 50)} of {len(filtered_df)} total")
                    
                    for idx, pred in filtered_df.head(50).iterrows():  # Show top 50 (was 20)
                        with st.container():
                            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                            
                            with col1:
                                status_emoji = "✅" if pred['status'] == 'completed' else "📈" if pred['status'] == 'will_trade' else "⏳"
                                st.markdown(f"**{status_emoji} {pred['pair']}**")
                                st.caption(f"ID: {pred['id']} | {pred['timestamp'][:16]}")
                            
                            with col2:
                                pos_type = pred.get('position_type', 'NEUTRAL')
                                direction = "🟢 LONG" if pos_type == 'LONG' else "🔴 SHORT" if pos_type == 'SHORT' else "⚪ NEUTRAL"
                                st.markdown(f"**{direction}**")
                                entry_price = pred.get('actual_entry_price') or pred['current_price']
                                st.caption(f"Entry: ${entry_price:.2f}")
                            
                            with col3:
                                st.markdown(f"**Confidence: {pred['confidence']:.1f}%**")
                                signal = pred.get('signal_strength', 0)
                                st.caption(f"Signal: {signal:.1f}/10" if signal else "N/A")
                            
                            with col4:
                                target = pred.get('target_price')
                                stop = pred.get('stop_loss')
                                if target:
                                    st.markdown(f"**Target: ${target:.2f}**")
                                else:
                                    st.markdown("**Target: N/A**")
                                if stop:
                                    st.caption(f"Stop: ${stop:.2f}")
                                else:
                                    st.caption("Stop: N/A")
                            
                            with col5:
                                # Get position type, treating None as NEUTRAL
                                pos_type = pred.get('position_type')
                                
                                # Show close button if trade has entry price (regardless of position type)
                                if pred['status'] == 'completed':
                                    st.caption("Closed")
                                elif pred['status'] == 'will_trade':
                                    # Check if we have valid entry price
                                    entry_check = pred.get('actual_entry_price')
                                    if pd.isna(entry_check) or entry_check is None:
                                        st.caption("⚠️ No entry price")
                                    else:
                                        # Allow closing for ANY position (including NEUTRAL/manual trades)
                                        if st.button(f"Close Trade", key=f"close_{pred['id']}", use_container_width=True):
                                            st.session_state[f'closing_{pred["id"]}'] = True
                                            st.rerun()
                                        
                                        # Show close trade form
                                        if st.session_state.get(f'closing_{pred["id"]}', False):
                                            with st.form(key=f"form_close_{pred['id']}"):
                                                st.markdown(f"**📊 Close Trade #{pred['id']}**")
                                                
                                                exit_price = st.number_input(
                                                    "💰 Actual Exit Price ($)",
                                                    min_value=0.0,
                                                    value=float(current_price),
                                                    step=0.01,
                                                    help="Enter the actual price you exited at (you can modify this)",
                                                    key=f"exit_{pred['id']}"
                                                )
                                                
                                                exit_reason = st.selectbox(
                                                    "Exit Reason",
                                                    ["Target Hit", "Stop Loss Hit", "Manual Close", "Timeout"],
                                                    key=f"reason_{pred['id']}"
                                                )
                                                
                                                col_submit, col_cancel = st.columns(2)
                                                
                                                with col_submit:
                                                    submitted = st.form_submit_button("✅ Confirm", use_container_width=True)
                                                
                                                with col_cancel:
                                                    cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
                                                
                                                if submitted:
                                                    # Calculate P/L
                                                    entry = pred.get('actual_entry_price')
                                                    
                                                    # Validate entry price
                                                    if entry is None or pd.isna(entry):
                                                        st.error("❌ Cannot close trade: No valid entry price")
                                                    else:
                                                        pos_type_confirmed = pred.get('position_type', 'LONG')
                                                        
                                                        # If NEUTRAL or None, ask user or default to LONG for manual trades
                                                        if not pos_type_confirmed or pos_type_confirmed == 'NEUTRAL' or pd.isna(pos_type_confirmed):
                                                            pos_type_confirmed = 'LONG'  # Default to LONG for manual trades
                                                        
                                                        if pos_type_confirmed == 'LONG':
                                                            pl = exit_price - entry
                                                        else:  # SHORT
                                                            pl = entry - exit_price
                                                        
                                                        pl_pct = (pl / entry) * 100
                                                        
                                                        # Calculate prediction error
                                                        predicted = pred.get('predicted_price', entry)
                                                        prediction_error = abs(exit_price - predicted) / predicted * 100 if predicted and not pd.isna(predicted) else 0
                                                        
                                                        # Save trade result
                                                        try:
                                                            success = save_trade_result(
                                                                prediction_id=int(pred['id']),
                                                                entry_price=float(entry),
                                                                exit_price=float(exit_price),
                                                                profit_loss=float(pl),
                                                                profit_loss_pct=float(pl_pct),
                                                                prediction_error=float(prediction_error),
                                                                notes=exit_reason
                                                            )
                                                            
                                                            if success:
                                                                # 🆕 TRIGGER COMMITTEE LEARNING!
                                                                if COMMITTEE_AVAILABLE and committee:
                                                                    try:
                                                                        committee.learning_system.learn_from_trade(int(pred['id']))
                                                                    except Exception as learn_error:
                                                                        st.warning(f"⚠️ Trade closed but learning failed: {learn_error}")
                                                                
                                                                st.success(f"✅ Trade closed! P/L: ${pl:.2f} ({pl_pct:+.2f}%)")
                                                                st.session_state[f'closing_{pred["id"]}'] = False
                                                                time.sleep(1)
                                                                st.rerun()
                                                            else:
                                                                st.error("❌ Failed to close trade")
                                                        except Exception as e:
                                                            st.error(f"❌ Error: {str(e)}")
                                                
                                                if cancel:
                                                    st.session_state[f'closing_{pred["id"]}'] = False
                                                    st.rerun()
                                else:
                                    st.caption("Closed")
                            
                            st.markdown("---")
                else:
                    st.info("No predictions match the selected filters")
            else:
                st.info("📝 No predictions yet. Save a prediction to start tracking!")
            
            # ==================== CLOSED TRADES HISTORY ====================
            col_title, col_edit_btn = st.columns([3, 1])
            with col_title:
                st.markdown("### 💰 Closed Trades History")
            with col_edit_btn:
                if st.button("✏️ Edit Trade", key="edit_trade_btn"):
                    st.session_state['show_edit_form'] = not st.session_state.get('show_edit_form', False)
            
            # Edit Trade Form
            if st.session_state.get('show_edit_form', False):
                with st.form("edit_trade_form"):
                    st.markdown("#### 🔍 Search & Delete Trade")
                    
                    trade_id_input = st.number_input(
                        "Trade ID to Delete",
                        min_value=1,
                        value=59,
                        step=1,
                        help="Enter the ID of the trade you want to delete"
                    )
                    
                    col_search, col_cancel = st.columns(2)
                    
                    with col_search:
                        search_clicked = st.form_submit_button("🔍 Search & Preview", use_container_width=True)
                    
                    with col_cancel:
                        cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True)
                    
                    if cancel_clicked:
                        st.session_state['show_edit_form'] = False
                        st.rerun()
                    
                    if search_clicked:
                        try:
                            conn_search = sqlite3.connect(str(DB_PATH))
                            cursor_search = conn_search.cursor()
                            
                            # Get trade details
                            cursor_search.execute("""
                                SELECT 
                                    tr.id,
                                    tr.trade_date,
                                    p.pair,
                                    p.position_type,
                                    tr.entry_price,
                                    tr.exit_price,
                                    tr.profit_loss,
                                    tr.profit_loss_pct,
                                    tr.notes
                                FROM trade_results tr
                                LEFT JOIN predictions p ON tr.prediction_id = p.id
                                WHERE tr.id = ?
                            """, (trade_id_input,))
                            
                            trade = cursor_search.fetchone()
                            conn_search.close()
                            
                            if trade:
                                st.success("✅ Trade Found!")
                                
                                # Display trade details
                                st.markdown("**Trade Details:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.info(f"**ID:** {trade[0]}")
                                    st.info(f"**Symbol:** {trade[2] or 'N/A'}")
                                    st.info(f"**Position:** {trade[3] or 'N/A'}")
                                with col2:
                                    st.info(f"**Entry:** ${trade[4]:.2f}")
                                    st.info(f"**Exit:** ${trade[5]:.2f}")
                                with col3:
                                    pl_color = "🟢" if trade[6] > 0 else "🔴"
                                    st.info(f"**P/L:** {pl_color} ${trade[6]:.2f}")
                                    st.info(f"**P/L %:** {trade[7]:.2f}%")
                                
                                st.warning(f"**Exit Reason:** {trade[8]}")
                                
                                # Store trade ID for deletion
                                st.session_state['trade_to_delete'] = trade_id_input
                                
                            else:
                                st.error(f"❌ Trade ID {trade_id_input} not found!")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                # Delete button (outside form, only shows if trade found)
                if st.session_state.get('trade_to_delete'):
                    if st.button("🗑️ DELETE THIS TRADE", type="primary", use_container_width=True):
                        try:
                            trade_id_to_delete = st.session_state['trade_to_delete']
                            
                            conn_del = sqlite3.connect(str(DB_PATH))
                            cursor_del = conn_del.cursor()
                            
                            # Get P/L before deletion
                            cursor_del.execute("SELECT SUM(profit_loss) FROM trade_results")
                            total_before = cursor_del.fetchone()[0]
                            
                            # Delete the trade
                            cursor_del.execute("DELETE FROM trade_results WHERE id = ?", (trade_id_to_delete,))
                            conn_del.commit()
                            
                            # Get P/L after deletion
                            cursor_del.execute("SELECT SUM(profit_loss) FROM trade_results")
                            total_after = cursor_del.fetchone()[0]
                            
                            conn_del.close()
                            
                            # Clear session state
                            st.session_state['trade_to_delete'] = None
                            st.session_state['show_edit_form'] = False
                            
                            st.success(f"✅ Trade {trade_id_to_delete} deleted successfully!")
                            st.info(f"📊 Total P/L updated: ${total_before:.2f} → ${total_after:.2f}")
                            
                            time.sleep(2)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error deleting trade: {str(e)}")
                
                st.markdown("---")

            
            try:
                conn = sqlite3.connect(str(DB_PATH))
                trades_df = pd.read_sql_query("""
                    SELECT 
                        tr.id,
                        tr.trade_date,
                        COALESCE(p.pair, 'N/A') as symbol,
                        COALESCE(p.position_type, 'N/A') as position_type,
                        tr.entry_price,
                        tr.exit_price,
                        tr.profit_loss,
                        tr.profit_loss_pct,
                        tr.notes as exit_reason,
                        COALESCE(p.confidence, 0) as confidence,
                        COALESCE(p.signal_strength, 0) as signal_strength
                    FROM trade_results tr
                    LEFT JOIN predictions p ON tr.prediction_id = p.id
                    ORDER BY tr.trade_date DESC
                    LIMIT 100
                """, conn)
                conn.close()
                
                if len(trades_df) > 0:
                    # Add color coding
                    def color_pl(val):
                        if val > 0:
                            return 'background-color: #d4edda'
                        elif val < 0:
                            return 'background-color: #f8d7da'
                        return ''
                    
                    styled_df = trades_df.style.applymap(color_pl, subset=['profit_loss', 'profit_loss_pct'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary stats - Calculate from ALL trades, not just displayed 20
                    # Get complete stats from database
                    conn_stats = sqlite3.connect(str(DB_PATH))
                    cursor_stats = conn_stats.cursor()
                    
                    cursor_stats.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                            AVG(CASE WHEN profit_loss > 0 THEN profit_loss END) as avg_win,
                            AVG(CASE WHEN profit_loss < 0 THEN profit_loss END) as avg_loss,
                            SUM(profit_loss) as total_pl
                        FROM trade_results
                    """)
                    
                    stats = cursor_stats.fetchone()
                    conn_stats.close()
                    
                    total_all = stats[0] if stats[0] else 0
                    wins_all = stats[1] if stats[1] else 0
                    avg_win_all = stats[2] if stats[2] else 0
                    avg_loss_all = stats[3] if stats[3] else 0
                    total_pl_all = stats[4] if stats[4] else 0
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        win_rate = (wins_all / total_all * 100) if total_all > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins_all}/{total_all}")
                    
                    with col_stat2:
                        st.metric("Avg Win", f"${avg_win_all:.2f}")
                    
                    with col_stat3:
                        st.metric("Avg Loss", f"${avg_loss_all:.2f}")
                    
                    with col_stat4:
                        st.metric("Total P/L", f"${total_pl_all:.2f}", 
                                 "🟢" if total_pl_all > 0 else "🔴" if total_pl_all < 0 else "⚪")
                else:
                    st.info("No closed trades yet")
            
            except Exception as e:
                st.error(f"Error loading trade history: {str(e)}")
            
            # ==================== INDICATOR PERFORMANCE ====================
            st.markdown("### 🎯 Indicator Performance Analysis")
            
            try:
                conn = sqlite3.connect(str(DB_PATH))
                
                # Check if we have any closed trades
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trade_results")
                trade_count = cursor.fetchone()[0]
                
                # DEBUG: Show raw data
                with st.expander("🔍 Debug: Raw Indicator Data"):
                    debug_df = pd.read_sql_query("""
                        SELECT * FROM indicator_accuracy
                    """, conn)
                    st.dataframe(debug_df, use_container_width=True)
                
                # Get indicator data
                indicator_df = pd.read_sql_query("""
                    SELECT 
                        indicator_name,
                        correct_count,
                        wrong_count,
                        accuracy_rate,
                        weight_multiplier,
                        last_updated
                    FROM indicator_accuracy
                    ORDER BY accuracy_rate DESC
                """, conn)
                conn.close()
                
                if len(indicator_df) > 0:
                    # Show indicators that have been evaluated
                    st.dataframe(indicator_df, use_container_width=True)
                    
                    # Visual chart (only for indicators with data)
                    chart_df = indicator_df[indicator_df['correct_count'] + indicator_df['wrong_count'] > 0].copy()
                    
                    if len(chart_df) > 0:
                        fig_ind = go.Figure()
                        
                        fig_ind.add_trace(go.Bar(
                            x=chart_df['indicator_name'],
                            y=chart_df['accuracy_rate'] * 100,
                            name='Accuracy %',
                            marker_color=['green' if x > 0.6 else 'orange' if x > 0.5 else 'red' 
                                         for x in chart_df['accuracy_rate']]
                        ))
                        
                        fig_ind.update_layout(
                            title="Indicator Accuracy Rates",
                            xaxis_title="Indicator",
                            yaxis_title="Accuracy %",
                            height=400
                        )
                        
                        st.plotly_chart(fig_ind, use_container_width=True)
                    else:
                        st.info("📊 Indicator data initialized. Close trades or click 'Relearn from Past Trades' to see performance!")
                else:
                    # Show message - no indicators in table at all
                    if trade_count > 0:
                        st.warning(f"📊 You have {trade_count} closed trades!")
                        st.info("👆 Click the '🔄 Relearn from Past Trades' button in the sidebar to analyze them and populate indicator performance!")
                    else:
                        st.info("💡 No closed trades yet. Close some trades to see AI learning in action!")
                        st.caption("The system will automatically track which indicators are accurate as you trade.")
            
            except Exception as e:
                st.error(f"Error loading indicator performance: {str(e)}")
    
    else:
        st.error("❌ Could not generate predictions")

else:
    st.error("❌ Unable to fetch data. Please check symbol and try again.")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p><b>🚀 AI TRADING PLATFORM - COMMITTEE LEARNING SYSTEM</b></p>
    <p><b>⚠️ Educational purposes only. Not financial advice.</b></p>
</div>
""", unsafe_allow_html=True)
