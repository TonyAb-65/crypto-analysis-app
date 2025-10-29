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
    
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    gb_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
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
        gb_pred = gb_model.predict(current_scaled)[0
