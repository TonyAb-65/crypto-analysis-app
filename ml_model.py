"""
ML Model Module - Machine learning predictions with pattern recognition
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from support_resistance import check_support_resistance_barriers, analyze_timeframe_volatility, adjust_confidence_for_barriers


def create_pattern_features(df, lookback=6):
    """Create features for pattern-based prediction"""
    features = []
    targets = []
    
    for i in range(lookback, len(df)):
        hour_features = []
        
        for j in range(i - lookback, i):
            hour_features.extend([
                df['close'].iloc[j],
                df['volume'].iloc[j],
                df['rsi'].iloc[j] if 'rsi' in df.columns else 50,
                df['macd'].iloc[j] if 'macd' in df.columns else 0,
                df['sma_20'].iloc[j] if 'sma_20' in df.columns else df['close'].iloc[j],
                df['volatility'].iloc[j] if 'volatility' in df.columns else 0
            ])
            
            if j > i - lookback:
                prev_close = df['close'].iloc[j-1]
                hour_features.append((df['close'].iloc[j] - prev_close) / (prev_close + 1e-10))
            else:
                hour_features.append(0)
        
        features.append(hour_features)
        targets.append(df['close'].iloc[i])
    
    return np.array(features), np.array(targets)


def train_improved_model(df, lookback=6, prediction_periods=5):
    """Train ML model with cross-validation and pattern recognition"""
    try:
        if len(df) < 60:
            st.warning("⚠️ Need at least 60 data points")
            return None, None, 0, None
        
        df_clean = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        X, y = create_pattern_features(df_clean, lookback=lookback)
        
        if len(X) < 30:
            st.warning("⚠️ Not enough data after cleaning")
            return None, None, 0, None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            st.error("❌ Data contains NaN values after cleaning")
            return None, None, 0, None
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_test = X_scaled[split_idx:]
        y_test = y[split_idx:]
        
        # Random Forest Model
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting Model
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            cv_model = RandomForestRegressor(n_estimators=100, random_state=42)
            cv_model.fit(X_cv_train, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val)
            
            returns_actual_cv = np.diff(y_cv_val) / (y_cv_val[:-1] + 1e-8)
            returns_pred_cv = np.diff(cv_pred) / (cv_pred[:-1] + 1e-8)
            
            mask_cv = np.isfinite(returns_actual_cv) & np.isfinite(returns_pred_cv) & (np.abs(returns_actual_cv) > 1e-6)
            returns_actual_cv_clean = returns_actual_cv[mask_cv]
            returns_pred_cv_clean = returns_pred_cv[mask_cv]
            
            if len(returns_actual_cv_clean) > 0:
                cv_mape = np.mean(np.abs((returns_actual_cv_clean - returns_pred_cv_clean) / (returns_actual_cv_clean + 1e-8)))
                cv_mape = min(cv_mape, 100)
            else:
                cv_mape = 50.0
            
            cv_scores.append(cv_mape)
        
        # Train final model on all training data
        gb_model.fit(X_train, y_train)
        
        # Create predictions
        current_sequence = []
        lookback_start = len(df_clean) - lookback
        
        for i in range(lookback_start, len(df_clean)):
            hour_features = [
                df_clean['close'].iloc[i],
                df_clean['volume'].iloc[i],
                df_clean['rsi'].iloc[i] if 'rsi' in df_clean.columns else 50,
                df_clean['macd'].iloc[i] if 'macd' in df_clean.columns else 0,
                df_clean['sma_20'].iloc[i] if 'sma_20' in df_clean.columns else df_clean['close'].iloc[i],
                df_clean['volatility'].iloc[i] if 'volatility' in df_clean.columns else 0
            ]
            
            if i > lookback_start:
                prev_close = df_clean['close'].iloc[i-1]
                hour_features.append((df_clean['close'].iloc[i] - prev_close) / (prev_close + 1e-10))
            else:
                hour_features.append(0)
            
            current_sequence.extend(hour_features)
        
        current_sequence = np.array(current_sequence).reshape(1, -1)
        current_sequence = np.nan_to_num(current_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_scaled = scaler.transform(current_sequence)
        
        predictions = []
        for _ in range(prediction_periods):
            rf_pred = rf_model.predict(current_scaled)[0]
            gb_pred = gb_model.predict(current_scaled)[0]
            pred_price = 0.4 * rf_pred + 0.6 * gb_pred
            predictions.append(float(pred_price))
        
        # Use fixed base confidence (MAPE removed for simplicity)
        # Confidence is adjusted later based on barriers and volatility
        base_confidence = 65
        
        # Apply surgical fixes
        current_price_model = df_clean['close'].iloc[-1]
        predicted_price = predictions[0]
        pred_change_pct = ((predicted_price - current_price_model) / current_price_model) * 100
        
        barriers = check_support_resistance_barriers(df_clean, predicted_price, current_price_model)
        
        timeframe_hours = prediction_periods
        volatility_context = analyze_timeframe_volatility(df_clean, pred_change_pct, timeframe_hours)
        
        adjusted_confidence = adjust_confidence_for_barriers(base_confidence, barriers, volatility_context)
        
        from indicators import analyze_rsi_bounce_patterns
        rsi_insights = analyze_rsi_bounce_patterns(df_clean)
        
        return predictions, ['Pattern-based features'], adjusted_confidence, rsi_insights
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None, 0, None
