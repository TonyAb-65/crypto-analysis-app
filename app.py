"""
Main Application - Streamlit UI
Imports functions from modularized files
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
    get_all_recent_predictions, save_trade_result, get_indicator_weights, DB_PATH
)
from utils import (
    should_retrain, trigger_ai_retraining, analyze_indicator_accuracy,
    backup_database, export_trades_to_csv
)
from news import (
    get_fear_greed_index, get_crypto_news_sentiment, analyze_news_sentiment_warning
)
from signals import (
    calculate_signal_strength, calculate_warning_signs, create_indicator_snapshot
)
from data_api import fetch_data, get_batch_data_binance
from support_resistance import find_support_resistance_zones
from indicators import calculate_technical_indicators
from ml_model import train_improved_model
from consultants import run_consultant_meeting

# Initialize database
init_database()

# ==================== STREAMLIT PAGE CONFIGURATION ====================

st.set_page_config(page_title="AI Trading Platform", layout="wide", page_icon="🤖")

st.title("🤖 AI Trading Analysis Platform - ENHANCED WITH SURGICAL FIXES")
st.markdown("*Crypto, Forex, Metals + AI ML Predictions + Trading Central Format + AI Learning + News Sentiment*")

if 'binance_blocked' not in st.session_state:
    st.session_state.binance_blocked = False

if st.session_state.binance_blocked:
    st.info("ℹ️ **Note:** Binance API is blocked in your region. Using OKX and backup APIs instead.")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**🕐 Last Updated:** {current_time}")

with st.expander("💾 Database Information", expanded=False):
    st.info(f"""
    **Database Location:** `{DB_PATH}`
    
    **File Exists:** {'✅ Yes' if DB_PATH.exists() else '❌ No'}
    
    **Note:** All your trade history and predictions are saved to this database file.
    """)

st.markdown("---")

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("⚙️ Configuration")

debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed API information")

# ==================== TOP MOVERS ====================
show_market_movers = st.sidebar.checkbox("📈 Show Top Movers", value=False,
                                        help="Display today's top gainers and losers")

@st.cache_data(ttl=300, show_spinner=False)
def get_market_movers():
    """Get top movers from popular cryptocurrencies"""
    from data_api import get_retry_session
    
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
            st.warning("Unable to load market movers")

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
use_obv = st.sidebar.checkbox("OBV (Volume)", value=False, help="On-Balance Volume - tracks volume flow")
use_mfi = st.sidebar.checkbox("MFI (14)", value=False, help="Money Flow Index - volume-weighted RSI")
use_adx = st.sidebar.checkbox("ADX (14)", value=False, help="Average Directional Index - trend strength")
use_stoch = st.sidebar.checkbox("Stochastic", value=False, help="Stochastic Oscillator - momentum indicator")
use_cci = st.sidebar.checkbox("CCI (20)", value=False, help="Commodity Channel Index - cyclical trends")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 AI Learning System")
show_learning_dashboard = st.sidebar.checkbox("📊 Show Trades Table on Page", value=False,
                                              help="✅ Enable to see your tracked trades table")

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
        
        # Trading Recommendations (Consultant Meeting)
        st.markdown("### 💰 Trading Recommendations")
        st.markdown("*Powered by AI/ML + Trading Central Format*")
        
        meeting_result = run_consultant_meeting(symbol, asset_type, current_price, warning_details)
        
        # Display meeting result
        st.markdown("## 🏢 CONSULTANT MEETING RESULT")
        
        if meeting_result['position'] == 'NEUTRAL':
            st.warning("⚪ NEUTRAL - DO NOT ENTER")
        elif meeting_result['position'] == 'LONG':
            st.success("🟢 LONG SIGNAL")
        else:
            st.error("🔴 SHORT SIGNAL")
        
        st.markdown(f"**Reasoning:** {meeting_result['reasoning']}")
        
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
                    # Get latest prediction for this symbol
                    conn = sqlite3.connect(str(DB_PATH))
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, current_price, position_type, target_price, stop_loss FROM predictions 
                        WHERE pair = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (symbol,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        pred_id, curr_price, pos_type, target, stop = result
                        mark_prediction_for_trading(
                            prediction_id=pred_id,
                            actual_entry_price=curr_price,
                            entry_timestamp=datetime.now().isoformat(),
                            position_type=pos_type,
                            target_price=target,
                            stop_loss=stop
                        )
                        st.success(f"✅ Marked for trading!")
                    else:
                        st.warning("⚠️ Save prediction first")
        
        # ==================== SUPPORT/RESISTANCE ZONES ====================
        st.markdown("---")
        st.markdown("### 🎯 Support & Resistance Zones")
        
        sr_zones = find_support_resistance_zones(df)
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
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
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
        
        # RSI
        if use_rsi and 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
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
        
        indicator_cols = st.columns(3)
        col_idx = 0
        
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
        
        if use_mfi and 'mfi' in df.columns:
            with indicator_cols[col_idx % 3]:
                mfi_current = df['mfi'].iloc[-1]
                mfi_status = "🔴 Overbought" if mfi_current > 80 else "🟢 Oversold" if mfi_current < 20 else "⚪ Neutral"
                
                st.metric("MFI (Money Flow)", 
                         f"{mfi_current:.1f}",
                         mfi_status)
                st.caption("Volume-weighted RSI")
            col_idx += 1
        
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
        
        if use_cci and 'cci' in df.columns:
            with indicator_cols[col_idx % 3]:
                cci_current = df['cci'].iloc[-1]
                cci_status = "🔴 Overbought" if cci_current > 100 else "🟢 Oversold" if cci_current < -100 else "⚪ Neutral"
                
                st.metric("CCI (Cyclical)", 
                         f"{cci_current:.1f}",
                         cci_status)
                st.caption("Commodity Channel Index")
            col_idx += 1
        
        # ==================== TRADE TRACKING TABLE ====================
        if show_learning_dashboard:
            st.markdown("---")
            st.markdown("## 📊 Trade Tracking & Learning Dashboard")
            
            # Get all predictions
            all_predictions_df = get_all_recent_predictions(limit=50)
            
            if all_predictions_df is not None and len(all_predictions_df) > 0:
                st.markdown("### 🎯 Recent Predictions & Trades")
                
                # Filter options
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    filter_status = st.selectbox(
                        "Filter by Status",
                        ["All", "Open", "Closed", "For Trading"],
                        index=0
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
                    if filter_status == "Open":
                        filtered_df = filtered_df[filtered_df['status'] != 'completed']
                    elif filter_status == "Closed":
                        filtered_df = filtered_df[filtered_df['status'] == 'completed']
                    elif filter_status == "For Trading":
                        filtered_df = filtered_df[filtered_df['status'] == 'will_trade']
                
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
                    for idx, pred in filtered_df.head(20).iterrows():  # Show top 20
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
                                if not pos_type or pos_type == 'NEUTRAL' or pd.isna(pos_type):
                                    st.caption("Closed" if pred['status'] == 'completed' else "N/A")
                                elif pred['status'] != 'completed':
                                    # Check if we have valid entry price
                                    entry_check = pred.get('actual_entry_price')
                                    if pd.isna(entry_check) or entry_check is None:
                                        st.caption("⚠️ No entry price")
                                    else:
                                        if st.button(f"Close Trade", key=f"close_{pred['id']}", use_container_width=True):
                                            st.session_state[f'closing_{pred["id"]}'] = True
                                            st.rerun()
                                        
                                        # Show close trade form
                                        if st.session_state.get(f'closing_{pred["id"]}', False):
                                            with st.form(key=f"form_close_{pred['id']}"):
                                                st.markdown(f"**Close Trade #{pred['id']}**")
                                                
                                                exit_price = st.number_input(
                                                    "Exit Price ($)",
                                                    min_value=0.0,
                                                    value=float(current_price),
                                                    step=0.01,
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
                                                                # Analyze indicator accuracy
                                                                analyze_indicator_accuracy(pred['id'])
                                                                
                                                                # Check if retraining needed
                                                                if should_retrain():
                                                                    with st.spinner("🧠 Retraining AI..."):
                                                                        trigger_ai_retraining()
                                                                
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
            st.markdown("### 💰 Closed Trades History")
            
            try:
                conn = sqlite3.connect(str(DB_PATH))
                trades_df = pd.read_sql_query("""
                    SELECT 
                        tr.id,
                        tr.trade_date,
                        p.pair as symbol,
                        p.position_type,
                        p.actual_entry_price as entry_price,
                        tr.exit_price,
                        tr.profit_loss,
                        tr.profit_loss_pct,
                        tr.notes as exit_reason,
                        p.confidence,
                        p.signal_strength
                    FROM trade_results tr
                    JOIN predictions p ON tr.prediction_id = p.id
                    ORDER BY tr.trade_date DESC
                    LIMIT 20
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
                    
                    # Summary stats
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        wins = len(trades_df[trades_df['profit_loss'] > 0])
                        total = len(trades_df)
                        win_rate = (wins / total * 100) if total > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}/{total}")
                    
                    with col_stat2:
                        avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean()
                        st.metric("Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "$0.00")
                    
                    with col_stat3:
                        avg_loss = trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean()
                        st.metric("Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "$0.00")
                    
                    with col_stat4:
                        total_pl = trades_df['profit_loss'].sum()
                        st.metric("Total P/L", f"${total_pl:.2f}", 
                                 "🟢" if total_pl > 0 else "🔴" if total_pl < 0 else "⚪")
                else:
                    st.info("No closed trades yet")
            
            except Exception as e:
                st.error(f"Error loading trade history: {str(e)}")
            
            # ==================== INDICATOR PERFORMANCE ====================
            st.markdown("### 🎯 Indicator Performance Analysis")
            
            try:
                conn = sqlite3.connect(str(DB_PATH))
                indicator_df = pd.read_sql_query("""
                    SELECT 
                        indicator_name,
                        correct_count,
                        wrong_count,
                        accuracy_rate,
                        weight_multiplier,
                        last_updated
                    FROM indicator_accuracy
                    WHERE correct_count + wrong_count > 0
                    ORDER BY accuracy_rate DESC
                """, conn)
                conn.close()
                
                if len(indicator_df) > 0:
                    st.dataframe(indicator_df, use_container_width=True)
                    
                    # Visual chart
                    fig_ind = go.Figure()
                    
                    fig_ind.add_trace(go.Bar(
                        x=indicator_df['indicator_name'],
                        y=indicator_df['accuracy_rate'] * 100,
                        name='Accuracy %',
                        marker_color=['green' if x > 0.6 else 'orange' if x > 0.5 else 'red' 
                                     for x in indicator_df['accuracy_rate']]
                    ))
                    
                    fig_ind.update_layout(
                        title="Indicator Accuracy Rates",
                        xaxis_title="Indicator",
                        yaxis_title="Accuracy %",
                        height=400
                    )
                    
                    st.plotly_chart(fig_ind, use_container_width=True)
                else:
                    st.info("No indicator performance data yet. Close some trades to see analysis!")
            
            except Exception as e:
                st.error(f"Error loading indicator performance: {str(e)}")
    
    else:
        st.error("❌ Could not generate predictions")

else:
    st.error("❌ Unable to fetch data. Please check symbol and try again.")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p><b>🚀 AI TRADING PLATFORM - MODULAR ARCHITECTURE</b></p>
    <p><b>⚠️ Educational purposes only. Not financial advice.</b></p>
</div>
""", unsafe_allow_html=True)
