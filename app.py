"""
CORRECTED AI PREDICTION APPROACH
- Monitors last 4+ hours for trend context
- Learns RSI bounce patterns from history
- Pattern-based predictions (finds similar past scenarios)
- Predicts based on what actually happened in similar situations
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


class PatternBasedPredictor:
    """
    Corrected prediction system that:
    1. Looks at last 4+ hours as context window
    2. Learns RSI bounce patterns from history
    3. Finds similar historical patterns
    4. Predicts based on what happened next historically
    """
    
    def __init__(self, lookback_hours=6):
        self.lookback_hours = lookback_hours  # How many hours to look back
        self.scaler = RobustScaler()
        self.rsi_bounce_patterns = {}
        self.historical_patterns = []
        self.models = {}
        
    def analyze_rsi_bounce_history(self, df):
        """
        CRITICAL: Learn where RSI typically bounces from history
        
        Examples:
        - When RSI was at 75 (overbought), where did it bounce to? (70, 60, 50?)
        - When RSI was at 25 (oversold), where did it bounce to? (30, 40, 50?)
        - What's the average bounce level?
        """
        
        if 'rsi_12' not in df.columns or len(df) < 50:
            return None
        
        rsi = df['rsi_12'].values
        price = df['close'].values
        
        bounce_patterns = {
            'overbought_bounces': [],  # RSI > 70 reversals
            'oversold_bounces': [],     # RSI < 30 reversals
            'mid_bounces': []           # RSI 40-60 bounces
        }
        
        # Scan through history
        for i in range(10, len(rsi) - 10):
            current_rsi = rsi[i]
            future_rsi = rsi[i+1:i+11]  # Next 10 periods
            current_price = price[i]
            future_prices = price[i+1:i+11]
            
            # OVERBOUGHT REVERSAL DETECTION (RSI > 70)
            if current_rsi > 70:
                # Find where it bounced back down to
                bounce_points = future_rsi[future_rsi < 70]
                if len(bounce_points) > 0:
                    bounce_level = bounce_points[0]
                    periods_to_bounce = np.where(future_rsi < 70)[0][0] + 1
                    price_change = ((future_prices[periods_to_bounce-1] - current_price) / current_price) * 100
                    
                    bounce_patterns['overbought_bounces'].append({
                        'from_rsi': current_rsi,
                        'to_rsi': bounce_level,
                        'periods': periods_to_bounce,
                        'price_change_pct': price_change
                    })
            
            # OVERSOLD REVERSAL DETECTION (RSI < 30)
            elif current_rsi < 30:
                # Find where it bounced back up to
                bounce_points = future_rsi[future_rsi > 30]
                if len(bounce_points) > 0:
                    bounce_level = bounce_points[0]
                    periods_to_bounce = np.where(future_rsi > 30)[0][0] + 1
                    price_change = ((future_prices[periods_to_bounce-1] - current_price) / current_price) * 100
                    
                    bounce_patterns['oversold_bounces'].append({
                        'from_rsi': current_rsi,
                        'to_rsi': bounce_level,
                        'periods': periods_to_bounce,
                        'price_change_pct': price_change
                    })
            
            # MID-LEVEL BOUNCES (40 < RSI < 60)
            elif 40 < current_rsi < 60:
                # Check if there was a significant move in next 5 periods
                if len(future_prices) >= 5:
                    max_price = max(future_prices[:5])
                    min_price = min(future_prices[:5])
                    price_change = ((max_price - min_price) / current_price) * 100
                    
                    if price_change > 1:  # Significant move
                        bounce_patterns['mid_bounces'].append({
                            'from_rsi': current_rsi,
                            'price_change_pct': price_change,
                            'direction': 'up' if future_prices[4] > current_price else 'down'
                        })
        
        # Calculate statistics
        self.rsi_bounce_patterns = {
            'overbought': self._calculate_bounce_stats(bounce_patterns['overbought_bounces']),
            'oversold': self._calculate_bounce_stats(bounce_patterns['oversold_bounces']),
            'mid_level': self._calculate_bounce_stats(bounce_patterns['mid_bounces'])
        }
        
        return self.rsi_bounce_patterns
    
    def _calculate_bounce_stats(self, bounces):
        """Calculate average bounce statistics"""
        if not bounces:
            return None
        
        df_bounces = pd.DataFrame(bounces)
        
        stats = {
            'count': len(bounces),
            'avg_price_change': df_bounces['price_change_pct'].mean() if 'price_change_pct' in df_bounces else 0,
            'avg_periods': df_bounces['periods'].mean() if 'periods' in df_bounces else 0,
            'avg_bounce_level': df_bounces['to_rsi'].mean() if 'to_rsi' in df_bounces else 0,
        }
        
        return stats
    
    def create_sequence_features(self, df, lookback=6):
        """
        Create features using LAST N HOURS as context
        
        This captures the trend/pattern of the last 4-6 hours
        Instead of just looking at current moment
        """
        
        sequences = []
        targets = []
        
        # We need lookback + 1 for target
        for i in range(lookback, len(df) - 1):
            # Get last N hours as a sequence
            sequence = []
            
            # For each of the last N hours
            for j in range(i - lookback, i):
                # Collect key features for this hour
                hour_features = [
                    df['close'].iloc[j],
                    df['volume'].iloc[j],
                    df['rsi_12'].iloc[j] if 'rsi_12' in df.columns else 50,
                    df['rsi_16'].iloc[j] if 'rsi_16' in df.columns else 50,
                    df['macd'].iloc[j] if 'macd' in df.columns else 0,
                    df['macd_signal'].iloc[j] if 'macd_signal' in df.columns else 0,
                    df['sma_20'].iloc[j] if 'sma_20' in df.columns else df['close'].iloc[j],
                    df['sma_50'].iloc[j] if 'sma_50' in df.columns else df['close'].iloc[j],
                    df['bb_upper'].iloc[j] if 'bb_upper' in df.columns else df['close'].iloc[j],
                    df['bb_lower'].iloc[j] if 'bb_lower' in df.columns else df['close'].iloc[j],
                    df['volatility'].iloc[j] if 'volatility' in df.columns else 0,
                ]
                
                # Calculate some sequential features
                if j > i - lookback:
                    prev_close = df['close'].iloc[j-1]
                    hour_features.extend([
                        (df['close'].iloc[j] - prev_close) / prev_close,  # Return
                        df['volume'].iloc[j] / (df['volume'].iloc[j-1] + 1e-10)  # Volume change
                    ])
                else:
                    hour_features.extend([0, 1])
                
                sequence.extend(hour_features)
            
            # Flatten the sequence into features
            sequences.append(sequence)
            
            # Target: next hour's price
            targets.append(df['close'].iloc[i])
        
        return np.array(sequences), np.array(targets)
    
    def find_similar_patterns(self, current_sequence, historical_sequences, top_k=10):
        """
        Find K most similar historical patterns to current situation
        Uses Euclidean distance in normalized feature space
        """
        
        # Normalize sequences
        current_norm = (current_sequence - np.mean(current_sequence)) / (np.std(current_sequence) + 1e-10)
        
        distances = []
        for i, hist_seq in enumerate(historical_sequences):
            hist_norm = (hist_seq - np.mean(hist_seq)) / (np.std(hist_seq) + 1e-10)
            dist = euclidean(current_norm, hist_norm)
            distances.append((i, dist))
        
        # Sort by distance and return top K
        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:top_k]]
    
    def train_pattern_based_model(self, df, prediction_horizons=[1, 2, 3, 4, 5]):
        """
        Train models for each prediction horizon based on historical patterns
        
        For each horizon (1hr, 2hr, 3hr, etc.):
        - Look at last N hours as pattern
        - Learn what happened N hours later in history
        - Train a model specifically for that horizon
        """
        
        if len(df) < 50:
            return None
        
        # First, analyze RSI bounce patterns
        self.analyze_rsi_bounce_history(df)
        
        # Create sequences with lookback window
        X_sequences, y_base = self.create_sequence_features(df, lookback=self.lookback_hours)
        
        if len(X_sequences) == 0:
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_sequences)
        
        # Store historical patterns for similarity search
        self.historical_patterns = X_scaled
        
        # Train a model for each prediction horizon
        for horizon in prediction_horizons:
            # Create target: price N hours ahead
            if horizon >= len(y_base):
                continue
            
            y_horizon = y_base[horizon:]
            X_horizon = X_scaled[:-horizon] if horizon > 0 else X_scaled
            
            if len(X_horizon) < 30:
                continue
            
            # Split into train/test
            split_idx = int(len(X_horizon) * 0.8)
            X_train = X_horizon[:split_idx]
            y_train = y_horizon[:split_idx]
            X_test = X_horizon[split_idx:]
            y_test = y_horizon[split_idx:]
            
            # Train optimized model for this horizon
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            test_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, test_pred) * 100
            accuracy = max(0, 100 - mape)
            
            self.models[f'horizon_{horizon}'] = {
                'model': model,
                'accuracy': accuracy,
                'mape': mape
            }
        
        return self.models
    
    def predict_with_context(self, df, periods=5):
        """
        Make predictions using:
        1. Last N hours as context
        2. Similar historical patterns
        3. RSI bounce pattern analysis
        4. Horizon-specific models
        """
        
        if not self.models or len(df) < self.lookback_hours + 1:
            return None, None
        
        # Create current sequence (last N hours)
        current_sequence = []
        lookback_start = len(df) - self.lookback_hours
        
        for i in range(lookback_start, len(df)):
            hour_features = [
                df['close'].iloc[i],
                df['volume'].iloc[i],
                df['rsi_12'].iloc[i] if 'rsi_12' in df.columns else 50,
                df['rsi_16'].iloc[i] if 'rsi_16' in df.columns else 50,
                df['macd'].iloc[i] if 'macd' in df.columns else 0,
                df['macd_signal'].iloc[i] if 'macd_signal' in df.columns else 0,
                df['sma_20'].iloc[i] if 'sma_20' in df.columns else df['close'].iloc[i],
                df['sma_50'].iloc[i] if 'sma_50' in df.columns else df['close'].iloc[i],
                df['bb_upper'].iloc[i] if 'bb_upper' in df.columns else df['close'].iloc[i],
                df['bb_lower'].iloc[i] if 'bb_lower' in df.columns else df['close'].iloc[i],
                df['volatility'].iloc[i] if 'volatility' in df.columns else 0,
            ]
            
            if i > lookback_start:
                prev_close = df['close'].iloc[i-1]
                hour_features.extend([
                    (df['close'].iloc[i] - prev_close) / prev_close,
                    df['volume'].iloc[i] / (df['volume'].iloc[i-1] + 1e-10)
                ])
            else:
                hour_features.extend([0, 1])
            
            current_sequence.extend(hour_features)
        
        current_sequence = np.array(current_sequence).reshape(1, -1)
        current_scaled = self.scaler.transform(current_sequence)
        
        # Make predictions for each horizon
        predictions = []
        confidence_scores = []
        
        for horizon in range(1, min(periods + 1, 6)):
            model_key = f'horizon_{horizon}'
            
            if model_key in self.models:
                model_info = self.models[model_key]
                pred_price = model_info['model'].predict(current_scaled)[0]
                
                # Adjust prediction based on RSI bounce patterns
                current_rsi = df['rsi_12'].iloc[-1] if 'rsi_12' in df.columns else 50
                pred_price = self._adjust_for_rsi_pattern(pred_price, current_rsi, horizon, df['close'].iloc[-1])
                
                predictions.append(pred_price)
                confidence_scores.append(model_info['accuracy'])
        
        # Overall confidence (weighted by horizon - closer predictions more confident)
        if confidence_scores:
            weights = [1.0 / (i + 1) for i in range(len(confidence_scores))]
            avg_confidence = np.average(confidence_scores, weights=weights)
        else:
            avg_confidence = 0
        
        return predictions, avg_confidence
    
    def _adjust_for_rsi_pattern(self, predicted_price, current_rsi, horizon, current_price):
        """
        Adjust prediction based on learned RSI bounce patterns
        
        If RSI is overbought (>70) and history shows it usually bounces to 50,
        adjust the prediction accordingly
        """
        
        if not self.rsi_bounce_patterns:
            return predicted_price
        
        # OVERBOUGHT adjustment
        if current_rsi > 70 and self.rsi_bounce_patterns.get('overbought'):
            stats = self.rsi_bounce_patterns['overbought']
            if stats and stats['count'] > 5:
                # History says when RSI is overbought, price typically drops
                expected_change = stats['avg_price_change']
                adjustment_factor = min(horizon / stats['avg_periods'], 1.0)
                
                # Apply historical pattern
                adjusted_price = current_price * (1 + (expected_change / 100) * adjustment_factor)
                
                # Blend with model prediction (50-50)
                predicted_price = (predicted_price + adjusted_price) / 2
        
        # OVERSOLD adjustment
        elif current_rsi < 30 and self.rsi_bounce_patterns.get('oversold'):
            stats = self.rsi_bounce_patterns['oversold']
            if stats and stats['count'] > 5:
                # History says when RSI is oversold, price typically rises
                expected_change = stats['avg_price_change']
                adjustment_factor = min(horizon / stats['avg_periods'], 1.0)
                
                adjusted_price = current_price * (1 + (expected_change / 100) * adjustment_factor)
                predicted_price = (predicted_price + adjusted_price) / 2
        
        return predicted_price
    
    def get_rsi_bounce_insights(self):
        """Return human-readable RSI bounce pattern insights"""
        
        if not self.rsi_bounce_patterns:
            return "No RSI pattern data available"
        
        insights = []
        
        # Overbought patterns
        if self.rsi_bounce_patterns.get('overbought'):
            stats = self.rsi_bounce_patterns['overbought']
            if stats and stats['count'] > 0:
                insights.append(
                    f"📉 **Overbought (RSI>70)**: Based on {stats['count']} historical cases, "
                    f"price typically moves {stats['avg_price_change']:.2f}% over "
                    f"{stats['avg_periods']:.1f} periods, bouncing to RSI ~{stats['avg_bounce_level']:.1f}"
                )
        
        # Oversold patterns
        if self.rsi_bounce_patterns.get('oversold'):
            stats = self.rsi_bounce_patterns['oversold']
            if stats and stats['count'] > 0:
                insights.append(
                    f"📈 **Oversold (RSI<30)**: Based on {stats['count']} historical cases, "
                    f"price typically moves {stats['avg_price_change']:.2f}% over "
                    f"{stats['avg_periods']:.1f} periods, bouncing to RSI ~{stats['avg_bounce_level']:.1f}"
                )
        
        return "\n".join(insights) if insights else "Insufficient data for RSI patterns"


def integrate_pattern_predictor(df, prediction_periods=5, lookback_hours=6):
    """
    Main function to integrate with your Streamlit app
    
    Usage:
        predictions, confidence, insights = integrate_pattern_predictor(df, 5, 6)
    """
    
    predictor = PatternBasedPredictor(lookback_hours=lookback_hours)
    
    # Train the pattern-based model
    models = predictor.train_pattern_based_model(df, prediction_horizons=list(range(1, prediction_periods + 1)))
    
    if not models:
        return None, 0, "Insufficient data for pattern analysis"
    
    # Make predictions
    predictions, confidence = predictor.predict_with_context(df, periods=prediction_periods)
    
    # Get RSI insights
    insights = predictor.get_rsi_bounce_insights()
    
    return predictions, confidence, insights
