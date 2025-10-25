def train_models_by_regime(X, y, regimes, model_choice):
    """
    Train models for up / down / chop regimes.
    Cleans NaNs and enforces strict length equality before .fit().
    """
    models = {}
    MIN_SAMPLES = 50  # require enough samples per regime

    for label in ["up", "down", "chop"]:
        mask = (regimes == label)
        X_sub = X[mask]
        y_sub = y[mask]

        # must have enough samples for that regime
        if len(X_sub) < MIN_SAMPLES:
            models[label] = None
            continue

        # time-based split
        split_idx = int(len(X_sub) * 0.8)
        if split_idx < 1:
            split_idx = 1
        if split_idx >= len(X_sub):
            split_idx = len(X_sub) - 1

        X_train, X_test = X_sub[:split_idx], X_sub[split_idx:]
        y_train, y_test = y_sub[:split_idx], y_sub[split_idx:]

        # hard truncate to same raw length
        n_train = min(len(X_train), len(y_train))
        n_test = min(len(X_test), len(y_test))

        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

        # drop any rows in train that contain NaN or inf in X or y
        if n_train > 0:
            train_mask_good = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[train_mask_good]
            y_train = y_train[train_mask_good]

        # drop any rows in test that contain NaN or inf
        if n_test > 0:
            test_mask_good = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
            X_test = X_test[test_mask_good]
            y_test = y_test[test_mask_good]

        # after cleaning check minimum viable size
        if len(X_train) < 2 or len(y_train) < 2:
            models[label] = None
            continue

        # flatten y for sklearn
        y_train_flat = np.ravel(y_train)
        y_test_flat = np.ravel(y_test) if len(y_test) > 0 else np.array([])

        # FINAL guard: lengths must match exactly before fit
        if len(X_train) != len(y_train_flat):
            models[label] = None
            continue

        # choose / fit model
        if model_choice == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            model.fit(X_train, y_train_flat)

            if len(X_test) > 1:
                test_pred = model.predict(X_test)
                # avoid divide by zero in MAPE
                denom = np.where(y_test_flat == 0, 1e-9, y_test_flat)
                mape = np.mean(np.abs((y_test_flat - test_pred) / denom)) * 100
                fit_score = max(0, 100 - mape)
            else:
                fit_score = 50.0

            models[label] = {
                "type": "single",
                "model": model,
                "fit_score": fit_score
            }

        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6
            )
            model.fit(X_train, y_train_flat)

            if len(X_test) > 1:
                test_pred = model.predict(X_test)
                denom = np.where(y_test_flat == 0, 1e-9, y_test_flat)
                mape = np.mean(np.abs((y_test_flat - test_pred) / denom)) * 100
                fit_score = max(0, 100 - mape)
            else:
                fit_score = 50.0

            models[label] = {
                "type": "single",
                "model": model,
                "fit_score": fit_score
            }

        else:
            # Ensemble (Recommended)
            rf = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=8
            )
            gb = GradientBoostingRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=5
            )

            rf.fit(X_train, y_train_flat)
            gb.fit(X_train, y_train_flat)

            if len(X_test) > 1:
                rf_pred = rf.predict(X_test)
                gb_pred = gb.predict(X_test)

                denom = np.where(y_test_flat == 0, 1e-9, y_test_flat)

                rf_mape = np.mean(np.abs((y_test_flat - rf_pred) / denom)) * 100
                gb_mape = np.mean(np.abs((y_test_flat - gb_pred) / denom)) * 100
                fit_score = max(0, 100 - ((rf_mape + gb_mape) / 2))
            else:
                fit_score = 50.0

            models[label] = {
                "type": "ensemble",
                "rf": rf,
                "gb": gb,
                "fit_score": fit_score
            }

    return models
