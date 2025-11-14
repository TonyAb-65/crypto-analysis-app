# consultants.py - UPDATED TO WORK WITH YOUR DATABASE.PY
# Learning-enabled consultants + your existing run_consultant_meeting function

from database import DB_PATH
import sqlite3
from datetime import datetime
import json

# ============================================================================
# BASE CLASS: LEARNING CONSULTANT
# ============================================================================

class LearningConsultant:
    """
    Base class for all learning-enabled consultants.
    
    Each consultant:
    - Loads their historical performance from database
    - Loads signal weights learned from past trades
    - Applies weighted scoring to their signals
    - Tracks which signals work best
    """
    
    def __init__(self, name, specialty):
        """
        Initialize learning consultant
        
        Args:
            name: Consultant identifier ('C1', 'C2', 'C3', 'C4')
            specialty: Area of expertise (e.g., 'Technical Analysis')
        """
        self.name = name
        self.specialty = specialty
        self.performance = self._load_performance()
        self.signal_weights = self._load_signal_weights()
    
    def _load_performance(self):
        """Load consultant's historical performance from database"""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_weight, accuracy_rate, total_votes, correct_votes, current_streak
                FROM consultant_performance
                WHERE consultant_name = ?
            """, (self.name,))
            
            row = cursor.fetchone()
            
            if not row:
                # Initialize new consultant
                cursor.execute("""
                    INSERT INTO consultant_performance 
                    (consultant_name, specialty, current_weight, accuracy_rate, performance_history)
                    VALUES (?, ?, 1.0, 50.0, '[]')
                """, (self.name, self.specialty))
                conn.commit()
                conn.close()
                
                return {
                    'weight': 1.0,
                    'accuracy': 50.0,
                    'total_votes': 0,
                    'correct_votes': 0,
                    'current_streak': 0
                }
            
            conn.close()
            
            return {
                'weight': row[0],
                'accuracy': row[1],
                'total_votes': row[2],
                'correct_votes': row[3],
                'current_streak': row[4]
            }
        
        except Exception as e:
            print(f"⚠️ Error loading performance for {self.name}: {e}")
            return {
                'weight': 1.0,
                'accuracy': 50.0,
                'total_votes': 0,
                'correct_votes': 0,
                'current_streak': 0
            }
    
    def _load_signal_weights(self):
        """Load learned weights for each signal this consultant uses"""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT signal_name, signal_weight
                FROM signal_performance
                WHERE consultant_name = ?
            """, (self.name,))
            
            weights = {}
            for row in cursor.fetchall():
                weights[row[0]] = row[1]
            
            conn.close()
            
            return weights
        
        except Exception as e:
            print(f"⚠️ Error loading signal weights for {self.name}: {e}")
            return {}
    
    def _get_signal_weight(self, signal_name):
        """Get current learned weight for a signal"""
        return self.signal_weights.get(signal_name, 1.0)
    
    def _apply_signal(self, signal_name, base_score, description=None):
        """Apply a signal with its learned weight"""
        weight = self._get_signal_weight(signal_name)
        weighted_score = base_score * weight
        
        return {
            'signal': signal_name,
            'description': description or signal_name,
            'base_score': base_score,
            'weight': weight,
            'final_score': weighted_score,
            'consultant': self.name
        }
    
    def reload_performance(self):
        """Reload performance metrics from database"""
        self.performance = self._load_performance()
        self.signal_weights = self._load_signal_weights()
    
    def analyze(self, data, indicators, signals, **kwargs):
        """Must be implemented by child classes"""
        raise NotImplementedError("Child class must implement analyze() method")


# ============================================================================
# C1: TECHNICAL ANALYST (Learning-Enabled)
# ============================================================================

class C1_TechnicalAnalyst(LearningConsultant):
    """Technical Indicator Specialist with Learning"""
    
    def __init__(self):
        super().__init__(name='C1', specialty='Technical Analysis')
    
    def analyze(self, data, indicators, signals, **kwargs):
        """Analyze technical indicators with learned weights"""
        score = 0
        reasoning = []
        signals_used = []
        
        # RSI ANALYSIS
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            
            if rsi < 30:
                signal_result = self._apply_signal('rsi_oversold', 2.0, f"RSI oversold ({rsi:.1f})")
                score += signal_result['final_score']
                reasoning.append(f"🟢 RSI oversold ({rsi:.1f}) - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
            elif rsi > 70:
                signal_result = self._apply_signal('rsi_overbought', -2.0, f"RSI overbought ({rsi:.1f})")
                score += signal_result['final_score']
                reasoning.append(f"🔴 RSI overbought ({rsi:.1f}) - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
        
        # MACD ANALYSIS
        if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            
            if macd > macd_signal:
                signal_result = self._apply_signal('macd_bullish', 2.0, "MACD bullish")
                score += signal_result['final_score']
                reasoning.append(f"🟢 MACD bullish - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
            elif macd < macd_signal:
                signal_result = self._apply_signal('macd_bearish', -2.0, "MACD bearish")
                score += signal_result['final_score']
                reasoning.append(f"🔴 MACD bearish - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
        
        # Determine vote
        if score >= 3:
            vote, confidence = 'BUY', 'HIGH'
        elif score >= 1:
            vote, confidence = 'BUY', 'MEDIUM'
        elif score > 0:
            vote, confidence = 'BUY', 'LOW'
        elif score <= -3:
            vote, confidence = 'SELL', 'HIGH'
        elif score <= -1:
            vote, confidence = 'SELL', 'MEDIUM'
        elif score < 0:
            vote, confidence = 'SELL', 'LOW'
        else:
            vote, confidence = 'HOLD', 'LOW'
        
        return {
            'consultant': self.name,
            'specialty': self.specialty,
            'vote': vote,
            'confidence': confidence,
            'score': score,
            'consultant_weight': self.performance['weight'],
            'weighted_vote_power': score * self.performance['weight'],
            'reasoning': reasoning,
            'signals_used': signals_used,
            'accuracy_history': f"{self.performance['accuracy']:.1f}% over {self.performance['total_votes']} votes",
            'current_streak': self.performance['current_streak']
        }


# ============================================================================
# C2: SENTIMENT ANALYST (Learning-Enabled)
# ============================================================================

class C2_SentimentAnalyst(LearningConsultant):
    """News & Market Sentiment Specialist with Learning"""
    
    def __init__(self):
        super().__init__(name='C2', specialty='Market Sentiment')
    
    def analyze(self, data, indicators, signals, news_data=None, **kwargs):
        """Analyze news and sentiment with learned weights"""
        score = 0
        reasoning = []
        signals_used = []
        
        if not news_data:
            return {
                'consultant': self.name,
                'specialty': self.specialty,
                'vote': 'HOLD',
                'confidence': 'LOW',
                'score': 0,
                'consultant_weight': self.performance['weight'],
                'weighted_vote_power': 0,
                'reasoning': ['No news data available'],
                'signals_used': [],
                'accuracy_history': f"{self.performance['accuracy']:.1f}% over {self.performance['total_votes']} votes",
                'current_streak': self.performance['current_streak']
            }
        
        # Analyze sentiment
        sentiment = news_data.get('sentiment_score', 0)
        
        if sentiment > 0.5:
            signal_result = self._apply_signal('news_very_positive', 3.0, "Very positive news")
            score += signal_result['final_score']
            reasoning.append(f"🟢 Very positive news - Weight: {signal_result['weight']:.2f}x")
            signals_used.append(signal_result)
        elif sentiment < -0.5:
            signal_result = self._apply_signal('news_very_negative', -3.0, "Very negative news")
            score += signal_result['final_score']
            reasoning.append(f"🔴 Very negative news - Weight: {signal_result['weight']:.2f}x")
            signals_used.append(signal_result)
        
        # Determine vote
        if score >= 2:
            vote, confidence = 'BUY', 'HIGH'
        elif score >= 1:
            vote, confidence = 'BUY', 'MEDIUM'
        elif score > 0:
            vote, confidence = 'BUY', 'LOW'
        elif score <= -2:
            vote, confidence = 'SELL', 'HIGH'
        elif score <= -1:
            vote, confidence = 'SELL', 'MEDIUM'
        elif score < 0:
            vote, confidence = 'SELL', 'LOW'
        else:
            vote, confidence = 'HOLD', 'LOW'
        
        return {
            'consultant': self.name,
            'specialty': self.specialty,
            'vote': vote,
            'confidence': confidence,
            'score': score,
            'consultant_weight': self.performance['weight'],
            'weighted_vote_power': score * self.performance['weight'],
            'reasoning': reasoning,
            'signals_used': signals_used,
            'accuracy_history': f"{self.performance['accuracy']:.1f}% over {self.performance['total_votes']} votes",
            'current_streak': self.performance['current_streak']
        }


# ============================================================================
# C3: RISK MANAGER (Learning-Enabled)
# ============================================================================

class C3_RiskManager(LearningConsultant):
    """Risk Assessment Specialist with Learning"""
    
    def __init__(self):
        super().__init__(name='C3', specialty='Risk Management')
    
    def analyze(self, data, indicators, signals, risk_metrics=None, **kwargs):
        """Analyze risk factors with learned weights"""
        score = 0
        reasoning = []
        signals_used = []
        
        # Volatility analysis
        if indicators.get('atr') and indicators.get('close'):
            atr = indicators['atr']
            price = indicators['close']
            volatility_pct = (atr / price) * 100
            
            if volatility_pct > 5:
                signal_result = self._apply_signal('volatility_very_high', -2.0, f"Very high volatility ({volatility_pct:.1f}%)")
                score += signal_result['final_score']
                reasoning.append(f"🔴 Very high volatility - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
            elif volatility_pct < 1:
                signal_result = self._apply_signal('volatility_low', 1.0, f"Low volatility ({volatility_pct:.1f}%)")
                score += signal_result['final_score']
                reasoning.append(f"🟢 Low volatility - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
        
        # Determine vote
        if score >= 2:
            vote, confidence = 'BUY', 'HIGH'
        elif score >= 1:
            vote, confidence = 'BUY', 'MEDIUM'
        elif score > 0:
            vote, confidence = 'BUY', 'LOW'
        elif score <= -2:
            vote, confidence = 'SELL', 'HIGH'
        elif score <= -1:
            vote, confidence = 'SELL', 'MEDIUM'
        elif score < 0:
            vote, confidence = 'SELL', 'LOW'
        else:
            vote, confidence = 'HOLD', 'LOW'
        
        return {
            'consultant': self.name,
            'specialty': self.specialty,
            'vote': vote,
            'confidence': confidence,
            'score': score,
            'consultant_weight': self.performance['weight'],
            'weighted_vote_power': score * self.performance['weight'],
            'reasoning': reasoning,
            'signals_used': signals_used,
            'accuracy_history': f"{self.performance['accuracy']:.1f}% over {self.performance['total_votes']} votes",
            'current_streak': self.performance['current_streak']
        }


# ============================================================================
# C4: TREND ANALYST (Learning-Enabled)
# ============================================================================

class C4_TrendAnalyst(LearningConsultant):
    """Trend & Pattern Recognition Specialist with Learning"""
    
    def __init__(self):
        super().__init__(name='C4', specialty='Trend Analysis')
    
    def analyze(self, data, indicators, signals, patterns=None, **kwargs):
        """Analyze trends and patterns with learned weights"""
        score = 0
        reasoning = []
        signals_used = []
        
        # Moving average alignment
        if indicators.get('sma_20') and indicators.get('sma_50'):
            sma20 = indicators['sma_20']
            sma50 = indicators['sma_50']
            
            if sma20 > sma50:
                signal_result = self._apply_signal('bullish_ma', 2.0, "Bullish MA alignment")
                score += signal_result['final_score']
                reasoning.append(f"🟢 Bullish MAs - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
            else:
                signal_result = self._apply_signal('bearish_ma', -2.0, "Bearish MA alignment")
                score += signal_result['final_score']
                reasoning.append(f"🔴 Bearish MAs - Weight: {signal_result['weight']:.2f}x")
                signals_used.append(signal_result)
        
        # Determine vote
        if score >= 3:
            vote, confidence = 'BUY', 'HIGH'
        elif score >= 1:
            vote, confidence = 'BUY', 'MEDIUM'
        elif score > 0:
            vote, confidence = 'BUY', 'LOW'
        elif score <= -3:
            vote, confidence = 'SELL', 'HIGH'
        elif score <= -1:
            vote, confidence = 'SELL', 'MEDIUM'
        elif score < 0:
            vote, confidence = 'SELL', 'LOW'
        else:
            vote, confidence = 'HOLD', 'LOW'
        
        return {
            'consultant': self.name,
            'specialty': self.specialty,
            'vote': vote,
            'confidence': confidence,
            'score': score,
            'consultant_weight': self.performance['weight'],
            'weighted_vote_power': score * self.performance['weight'],
            'reasoning': reasoning,
            'signals_used': signals_used,
            'accuracy_history': f"{self.performance['accuracy']:.1f}% over {self.performance['total_votes']} votes",
            'current_streak': self.performance['current_streak']
        }


# ============================================================================
# YOUR EXISTING FUNCTION (FALLBACK)
# ============================================================================

def run_consultant_meeting(symbol, asset_type, current_price, warning_details):
    """
    YOUR ORIGINAL CONSULTANT MEETING FUNCTION
    Used as fallback when committee system not available
    """
    # Simple trading logic based on warnings
    if warning_details and isinstance(warning_details, dict):
        warning_count = sum(1 for w in warning_details.values() if isinstance(w, dict) and w.get('active'))
    else:
        warning_count = 0
    
    # Determine position based on warnings
    if warning_count >= 3:
        position = 'NEUTRAL'
        reasoning = f"Too many warnings ({warning_count}) - stay out"
    elif warning_count >= 1:
        position = 'LONG'  # or SHORT based on signals
        reasoning = f"Some warnings ({warning_count}) detected - cautious entry"
    else:
        position = 'LONG'
        reasoning = "Clear signals - standard entry"
    
    # Calculate entry/target/stop
    entry = current_price
    
    if asset_type == "💰 Cryptocurrency":
        target = entry * 1.03  # 3% target for crypto
        stop_loss = entry * 0.99  # 1% stop
    else:
        target = entry * 1.01  # 1% target for forex/metals
        stop_loss = entry * 0.995  # 0.5% stop
    
    return {
        'position': position,
        'reasoning': reasoning,
        'entry': entry,
        'target': target,
        'stop_loss': stop_loss
    }
