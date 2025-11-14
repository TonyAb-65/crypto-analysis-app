# consultants_with_learning.py - STEP 2: ENHANCED CONSULTANTS WITH LEARNING
# Complete implementation of learning-enabled consultants for committee system

from database_module import get_session, ConsultantPerformance, SignalPerformance
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
        """
        Load consultant's historical performance from database
        
        Returns:
            dict: Performance metrics including weight, accuracy, total votes
        """
        session = get_session()
        
        try:
            perf = session.query(ConsultantPerformance).filter(
                ConsultantPerformance.consultant_name == self.name
            ).first()
            
            if not perf:
                # Initialize new consultant if not exists
                perf = ConsultantPerformance(
                    consultant_name=self.name,
                    specialty=self.specialty,
                    current_weight=1.0,
                    accuracy_rate=50.0
                )
                session.add(perf)
                session.commit()
                print(f"✅ Initialized new consultant: {self.name}")
            
            return {
                'weight': perf.current_weight,
                'accuracy': perf.accuracy_rate,
                'total_votes': perf.total_votes,
                'correct_votes': perf.correct_votes,
                'current_streak': perf.current_streak
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
        
        finally:
            session.close()
    
    def _load_signal_weights(self):
        """
        Load learned weights for each signal this consultant uses
        
        Returns:
            dict: Mapping of signal_name -> weight (e.g., {'rsi_oversold': 1.25})
        """
        session = get_session()
        
        try:
            signals = session.query(SignalPerformance).filter(
                SignalPerformance.consultant_name == self.name
            ).all()
            
            weights = {}
            for signal in signals:
                weights[signal.signal_name] = signal.signal_weight
            
            if weights:
                print(f"📊 {self.name}: Loaded {len(weights)} learned signal weights")
            
            return weights
        
        except Exception as e:
            print(f"⚠️ Error loading signal weights for {self.name}: {e}")
            return {}
        
        finally:
            session.close()
    
    def _get_signal_weight(self, signal_name):
        """
        Get current learned weight for a signal
        
        Args:
            signal_name: Name of the signal (e.g., 'rsi_oversold')
        
        Returns:
            float: Signal weight (defaults to 1.0 if new/unknown signal)
        """
        return self.signal_weights.get(signal_name, 1.0)
    
    def _apply_signal(self, signal_name, base_score, description=None):
        """
        Apply a signal with its learned weight
        
        Args:
            signal_name: Signal identifier
            base_score: Original score value (e.g., +2.0 for bullish, -2.0 for bearish)
            description: Human-readable description (optional)
        
        Returns:
            dict: Signal details including weighted score
        """
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
        """Reload performance metrics from database (call after learning updates)"""
        self.performance = self._load_performance()
        self.signal_weights = self._load_signal_weights()
    
    def analyze(self, data, indicators, signals, **kwargs):
        """
        Analyze market data and provide recommendation.
        Must be implemented by child classes.
        
        Args:
            data: Market data (OHLCV)
            indicators: Technical indicators dictionary
            signals: Trading signals
            **kwargs: Additional context (varies by consultant)
        
        Returns:
            dict: Consultant's vote, confidence, reasoning, and signal details
        """
        raise NotImplementedError("Child class must implement analyze() method")


# ============================================================================
# C1: TECHNICAL ANALYST (Learning-Enabled)
# ============================================================================

class C1_TechnicalAnalyst(LearningConsultant):
    """
    Consultant 1: Technical Indicator Specialist
    
    Analyzes:
    - RSI (oversold/overbought)
    - MACD (bullish/bearish crossovers)
    - ADX (trend strength)
    - Stochastic (momentum)
    - Moving Averages (golden/death cross)
    - Bollinger Bands
    
    NOW WITH LEARNING:
    - Tracks which indicators are most accurate
    - Adjusts signal weights based on past performance
    - Example: If RSI oversold has 85% accuracy, its weight increases to 1.5x
    """
    
    def __init__(self):
        super().__init__(name='C1', specialty='Technical Analysis')
    
    def analyze(self, data, indicators, signals, **kwargs):
        """
        Analyze technical indicators with learned weights
        
        Args:
            data: Historical market data
            indicators: Dict of technical indicators
            signals: Trading signals from technical_indicators module
        
        Returns:
            dict: Vote, confidence, score, reasoning, and signals used
        """
        score = 0
        reasoning = []
        signals_used = []
        
        # ===== RSI ANALYSIS =====
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            
            if rsi < 30:
                # RSI Oversold - Bullish signal
                signal_result = self._apply_signal(
                    signal_name='rsi_oversold',
                    base_score=2.0,
                    description=f"RSI oversold ({rsi:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 RSI oversold ({rsi:.1f}) - "
                    f"Base: +{signal_result['base_score']}, "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif rsi > 70:
                # RSI Overbought - Bearish signal
                signal_result = self._apply_signal(
                    signal_name='rsi_overbought',
                    base_score=-2.0,
                    description=f"RSI overbought ({rsi:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 RSI overbought ({rsi:.1f}) - "
                    f"Base: {signal_result['base_score']}, "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif 30 <= rsi <= 45:
                # RSI Neutral-Bullish
                signal_result = self._apply_signal(
                    signal_name='rsi_neutral_bullish',
                    base_score=0.5,
                    description=f"RSI neutral-bullish ({rsi:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟡 RSI neutral-bullish ({rsi:.1f}) - "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif 55 <= rsi <= 70:
                # RSI Neutral-Bearish
                signal_result = self._apply_signal(
                    signal_name='rsi_neutral_bearish',
                    base_score=-0.5,
                    description=f"RSI neutral-bearish ({rsi:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟡 RSI neutral-bearish ({rsi:.1f}) - "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== MACD ANALYSIS =====
        if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_hist = indicators.get('macd_hist', 0)
            
            if macd > macd_signal and macd_hist > 0:
                # MACD Bullish
                signal_result = self._apply_signal(
                    signal_name='macd_bullish',
                    base_score=2.0,
                    description="MACD bullish crossover"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 MACD bullish (MACD: {macd:.2f} > Signal: {macd_signal:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif macd < macd_signal and macd_hist < 0:
                # MACD Bearish
                signal_result = self._apply_signal(
                    signal_name='macd_bearish',
                    base_score=-2.0,
                    description="MACD bearish crossover"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 MACD bearish (MACD: {macd:.2f} < Signal: {macd_signal:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== ADX TREND STRENGTH =====
        if indicators.get('adx') is not None:
            adx = indicators['adx']
            di_plus = indicators.get('di_plus', 0)
            di_minus = indicators.get('di_minus', 0)
            
            if adx > 25:  # Strong trend
                if di_plus > di_minus:
                    # Strong Uptrend
                    signal_result = self._apply_signal(
                        signal_name='adx_strong_uptrend',
                        base_score=2.0,
                        description=f"Strong uptrend (ADX {adx:.1f})"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 Strong uptrend (ADX {adx:.1f}, DI+ > DI-) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                else:
                    # Strong Downtrend
                    signal_result = self._apply_signal(
                        signal_name='adx_strong_downtrend',
                        base_score=-2.0,
                        description=f"Strong downtrend (ADX {adx:.1f})"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 Strong downtrend (ADX {adx:.1f}, DI- > DI+) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
            
            elif adx < 20:  # Weak/no trend
                signal_result = self._apply_signal(
                    signal_name='adx_weak_trend',
                    base_score=-0.5,
                    description=f"Weak trend (ADX {adx:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"⚠️ Weak trend (ADX {adx:.1f}) - ranging market - "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== STOCHASTIC OSCILLATOR =====
        if indicators.get('stoch_k') is not None:
            stoch = indicators['stoch_k']
            
            if stoch < 20:
                # Stochastic Oversold
                signal_result = self._apply_signal(
                    signal_name='stoch_oversold',
                    base_score=1.5,
                    description=f"Stochastic oversold ({stoch:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Stochastic oversold ({stoch:.1f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif stoch > 80:
                # Stochastic Overbought
                signal_result = self._apply_signal(
                    signal_name='stoch_overbought',
                    base_score=-1.5,
                    description=f"Stochastic overbought ({stoch:.1f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Stochastic overbought ({stoch:.1f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== MOVING AVERAGE CROSSOVERS =====
        if indicators.get('sma_20') and indicators.get('sma_50'):
            sma20 = indicators['sma_20']
            sma50 = indicators['sma_50']
            
            if sma20 > sma50:
                # Golden Cross (or bullish alignment)
                signal_result = self._apply_signal(
                    signal_name='golden_cross',
                    base_score=2.5,
                    description="Golden cross (SMA20 > SMA50)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Golden cross (SMA20 > SMA50) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            else:
                # Death Cross (or bearish alignment)
                signal_result = self._apply_signal(
                    signal_name='death_cross',
                    base_score=-2.5,
                    description="Death cross (SMA20 < SMA50)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Death cross (SMA20 < SMA50) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== BOLLINGER BANDS =====
        if indicators.get('bb_upper') and indicators.get('bb_lower') and indicators.get('close'):
            close = indicators['close']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            if close <= bb_lower:
                # Price at lower band - oversold
                signal_result = self._apply_signal(
                    signal_name='bb_lower_touch',
                    base_score=1.5,
                    description="Price at lower Bollinger Band"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 At lower BB - potential bounce - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif close >= bb_upper:
                # Price at upper band - overbought
                signal_result = self._apply_signal(
                    signal_name='bb_upper_touch',
                    base_score=-1.5,
                    description="Price at upper Bollinger Band"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 At upper BB - potential reversal - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== VOLUME ANALYSIS =====
        if indicators.get('volume') and indicators.get('volume_sma'):
            volume = indicators['volume']
            volume_avg = indicators['volume_sma']
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1
            
            if volume_ratio > 1.5:
                # High volume confirms move
                signal_result = self._apply_signal(
                    signal_name='volume_confirmation',
                    base_score=0.5,
                    description=f"High volume ({volume_ratio:.1f}x avg)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"📊 High volume confirmation ({volume_ratio:.1f}x) - "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== DETERMINE VOTE BASED ON SCORE =====
        if score >= 5:
            vote = 'BUY'
            confidence = 'HIGH'
        elif score >= 2:
            vote = 'BUY'
            confidence = 'MEDIUM'
        elif score >= 0.5:
            vote = 'BUY'
            confidence = 'LOW'
        elif score <= -5:
            vote = 'SELL'
            confidence = 'HIGH'
        elif score <= -2:
            vote = 'SELL'
            confidence = 'MEDIUM'
        elif score <= -0.5:
            vote = 'SELL'
            confidence = 'LOW'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
        # ===== RETURN RESULT =====
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
    """
    Consultant 2: News & Market Sentiment Specialist
    
    Analyzes:
    - News sentiment (positive/negative)
    - Social media sentiment
    - Market fear/greed
    - Breaking news impact
    
    NOW WITH LEARNING:
    - Learns which sentiment signals are reliable
    - Adjusts confidence based on past accuracy
    - Example: If positive news has 65% accuracy, weight adjusts accordingly
    """
    
    def __init__(self):
        super().__init__(name='C2', specialty='Market Sentiment')
    
    def analyze(self, data, indicators, signals, news_data=None, **kwargs):
        """
        Analyze news and sentiment with learned weights
        
        Args:
            data: Market data
            indicators: Technical indicators
            signals: Trading signals
            news_data: Dict containing sentiment data (optional)
        
        Returns:
            dict: Vote, confidence, reasoning
        """
        score = 0
        reasoning = []
        signals_used = []
        
        if not news_data:
            # No news data available
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
        
        # ===== NEWS SENTIMENT ANALYSIS =====
        if news_data.get('sentiment_score') is not None:
            sentiment = news_data['sentiment_score']  # Expected range: -1 to +1
            
            if sentiment > 0.5:
                # Very Positive News
                signal_result = self._apply_signal(
                    signal_name='news_very_positive',
                    base_score=3.0,
                    description=f"Very positive news sentiment ({sentiment:.2f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Very positive news ({sentiment:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sentiment > 0.2:
                # Positive News
                signal_result = self._apply_signal(
                    signal_name='news_positive',
                    base_score=1.5,
                    description=f"Positive news sentiment ({sentiment:.2f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Positive news ({sentiment:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sentiment < -0.5:
                # Very Negative News
                signal_result = self._apply_signal(
                    signal_name='news_very_negative',
                    base_score=-3.0,
                    description=f"Very negative news sentiment ({sentiment:.2f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Very negative news ({sentiment:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sentiment < -0.2:
                # Negative News
                signal_result = self._apply_signal(
                    signal_name='news_negative',
                    base_score=-1.5,
                    description=f"Negative news sentiment ({sentiment:.2f})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Negative news ({sentiment:.2f}) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== SOCIAL MEDIA SENTIMENT =====
        if news_data.get('social_sentiment'):
            social = news_data['social_sentiment']
            
            if social == 'bullish' or social == 'very_bullish':
                signal_result = self._apply_signal(
                    signal_name='social_bullish',
                    base_score=1.5,
                    description="Social media bullish"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Social media bullish - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif social == 'bearish' or social == 'very_bearish':
                signal_result = self._apply_signal(
                    signal_name='social_bearish',
                    base_score=-1.5,
                    description="Social media bearish"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Social media bearish - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== FEAR & GREED INDEX =====
        if news_data.get('fear_greed_index') is not None:
            fg_index = news_data['fear_greed_index']  # 0-100 scale
            
            if fg_index < 25:
                # Extreme Fear - Contrarian buy signal
                signal_result = self._apply_signal(
                    signal_name='extreme_fear',
                    base_score=2.0,
                    description=f"Extreme fear index ({fg_index})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Extreme fear ({fg_index}) - contrarian buy - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif fg_index > 75:
                # Extreme Greed - Contrarian sell signal
                signal_result = self._apply_signal(
                    signal_name='extreme_greed',
                    base_score=-2.0,
                    description=f"Extreme greed index ({fg_index})"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Extreme greed ({fg_index}) - contrarian sell - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== BREAKING NEWS IMPACT =====
        if news_data.get('breaking_news'):
            breaking = news_data['breaking_news']
            
            if breaking.get('impact') == 'positive':
                signal_result = self._apply_signal(
                    signal_name='breaking_news_positive',
                    base_score=2.5,
                    description="Positive breaking news"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Breaking news: {breaking.get('headline', 'N/A')[:50]} - "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif breaking.get('impact') == 'negative':
                signal_result = self._apply_signal(
                    signal_name='breaking_news_negative',
                    base_score=-2.5,
                    description="Negative breaking news"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Breaking news: {breaking.get('headline', 'N/A')[:50]} - "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== DETERMINE VOTE =====
        if score >= 3:
            vote = 'BUY'
            confidence = 'HIGH'
        elif score >= 1:
            vote = 'BUY'
            confidence = 'MEDIUM'
        elif score > 0:
            vote = 'BUY'
            confidence = 'LOW'
        elif score <= -3:
            vote = 'SELL'
            confidence = 'HIGH'
        elif score <= -1:
            vote = 'SELL'
            confidence = 'MEDIUM'
        elif score < 0:
            vote = 'SELL'
            confidence = 'LOW'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
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
    """
    Consultant 3: Risk Assessment Specialist
    
    Analyzes:
    - Volatility (ATR)
    - Support/Resistance proximity
    - Volume patterns
    - Risk/Reward ratios
    
    NOW WITH LEARNING:
    - Learns optimal volatility thresholds
    - Adjusts risk tolerance based on success rate
    - Example: If high volatility trades win 70%, become more risk-tolerant
    """
    
    def __init__(self):
        super().__init__(name='C3', specialty='Risk Management')
    
    def analyze(self, data, indicators, signals, risk_metrics=None, **kwargs):
        """
        Analyze risk factors with learned weights
        
        Args:
            data: Market data
            indicators: Technical indicators
            signals: Trading signals
            risk_metrics: Dict containing support/resistance, etc.
        
        Returns:
            dict: Vote, confidence, reasoning
        """
        score = 0
        reasoning = []
        signals_used = []
        
        # ===== VOLATILITY ANALYSIS (ATR) =====
        if indicators.get('atr') and indicators.get('close'):
            atr = indicators['atr']
            price = indicators['close']
            volatility_pct = (atr / price) * 100
            
            if volatility_pct > 5:
                # Very High Volatility - High Risk
                signal_result = self._apply_signal(
                    signal_name='volatility_very_high',
                    base_score=-2.0,
                    description=f"Very high volatility ({volatility_pct:.1f}%)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Very high volatility ({volatility_pct:.1f}%) - RISKY - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif volatility_pct > 3:
                # High Volatility
                signal_result = self._apply_signal(
                    signal_name='volatility_high',
                    base_score=-1.0,
                    description=f"High volatility ({volatility_pct:.1f}%)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"⚠️ High volatility ({volatility_pct:.1f}%) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif volatility_pct < 1:
                # Low Volatility - Safe environment
                signal_result = self._apply_signal(
                    signal_name='volatility_low',
                    base_score=1.0,
                    description=f"Low volatility ({volatility_pct:.1f}%)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Low volatility ({volatility_pct:.1f}%) - SAFE - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== SUPPORT/RESISTANCE PROXIMITY =====
        if risk_metrics:
            if risk_metrics.get('near_resistance'):
                # Price near resistance - reversal risk
                signal_result = self._apply_signal(
                    signal_name='near_resistance',
                    base_score=-1.5,
                    description="Price near resistance"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Near resistance - reversal risk - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            if risk_metrics.get('near_support'):
                # Price near support - good entry point
                signal_result = self._apply_signal(
                    signal_name='near_support',
                    base_score=1.5,
                    description="Price near support"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Near support - good entry - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            # Risk/Reward Ratio
            if risk_metrics.get('risk_reward_ratio'):
                rr_ratio = risk_metrics['risk_reward_ratio']
                
                if rr_ratio >= 2.0:
                    # Good risk/reward
                    signal_result = self._apply_signal(
                        signal_name='good_risk_reward',
                        base_score=1.5,
                        description=f"Good R/R ratio ({rr_ratio:.1f}:1)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 Good R/R ({rr_ratio:.1f}:1) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif rr_ratio < 1.0:
                    # Poor risk/reward
                    signal_result = self._apply_signal(
                        signal_name='poor_risk_reward',
                        base_score=-1.5,
                        description=f"Poor R/R ratio ({rr_ratio:.1f}:1)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 Poor R/R ({rr_ratio:.1f}:1) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
        
        # ===== VOLUME ANALYSIS =====
        if indicators.get('volume') and indicators.get('volume_sma'):
            volume = indicators['volume']
            volume_avg = indicators['volume_sma']
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1
            
            if volume_ratio > 2:
                # Volume spike - strong conviction
                signal_result = self._apply_signal(
                    signal_name='volume_spike',
                    base_score=1.0,
                    description=f"Volume spike ({volume_ratio:.1f}x)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"📊 Volume spike ({volume_ratio:.1f}x) - strong move - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif volume_ratio < 0.5:
                # Low volume - weak conviction
                signal_result = self._apply_signal(
                    signal_name='volume_low',
                    base_score=-0.5,
                    description=f"Low volume ({volume_ratio:.1f}x)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"⚠️ Low volume ({volume_ratio:.1f}x) - weak move - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== DETERMINE VOTE =====
        if score >= 2:
            vote = 'BUY'
            confidence = 'HIGH'
        elif score >= 1:
            vote = 'BUY'
            confidence = 'MEDIUM'
        elif score > 0:
            vote = 'BUY'
            confidence = 'LOW'
        elif score <= -2:
            vote = 'SELL'
            confidence = 'HIGH'
        elif score <= -1:
            vote = 'SELL'
            confidence = 'MEDIUM'
        elif score < 0:
            vote = 'SELL'
            confidence = 'LOW'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
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
    """
    Consultant 4: Trend & Pattern Recognition Specialist
    
    Analyzes:
    - Moving average alignment
    - Candlestick patterns
    - Price momentum
    - Trend strength
    
    NOW WITH LEARNING:
    - Learns which patterns are most reliable
    - Adjusts trend detection sensitivity
    - Example: If hammer patterns have 80% accuracy, increase weight
    """
    
    def __init__(self):
        super().__init__(name='C4', specialty='Trend Analysis')
    
    def analyze(self, data, indicators, signals, patterns=None, **kwargs):
        """
        Analyze trends and patterns with learned weights
        
        Args:
            data: Market data (OHLCV)
            indicators: Technical indicators
            signals: Trading signals
            patterns: Dict of detected candlestick patterns
        
        Returns:
            dict: Vote, confidence, reasoning
        """
        score = 0
        reasoning = []
        signals_used = []
        
        # ===== MOVING AVERAGE ALIGNMENT =====
        if indicators.get('sma_20') and indicators.get('sma_50') and indicators.get('sma_200'):
            sma20 = indicators['sma_20']
            sma50 = indicators['sma_50']
            sma200 = indicators['sma_200']
            
            if sma20 > sma50 > sma200:
                # Perfect bullish alignment
                signal_result = self._apply_signal(
                    signal_name='strong_uptrend_all_ma',
                    base_score=3.0,
                    description="Strong uptrend (all MAs aligned)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Strong uptrend (20>50>200 MAs) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sma20 < sma50 < sma200:
                # Perfect bearish alignment
                signal_result = self._apply_signal(
                    signal_name='strong_downtrend_all_ma',
                    base_score=-3.0,
                    description="Strong downtrend (all MAs aligned)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Strong downtrend (20<50<200 MAs) - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sma20 > sma50:
                # Bullish short-term
                signal_result = self._apply_signal(
                    signal_name='bullish_short_term',
                    base_score=1.5,
                    description="Bullish (SMA20 > SMA50)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🟢 Bullish short-term - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: +{signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
            
            elif sma20 < sma50:
                # Bearish short-term
                signal_result = self._apply_signal(
                    signal_name='bearish_short_term',
                    base_score=-1.5,
                    description="Bearish (SMA20 < SMA50)"
                )
                score += signal_result['final_score']
                reasoning.append(
                    f"🔴 Bearish short-term - "
                    f"Weight: {signal_result['weight']:.2f}x, "
                    f"Score: {signal_result['final_score']:.2f}"
                )
                signals_used.append(signal_result)
        
        # ===== CANDLESTICK PATTERNS =====
        if patterns:
            for pattern_name, pattern_signal in patterns.items():
                if pattern_signal == 'bullish':
                    signal_result = self._apply_signal(
                        signal_name=f'pattern_{pattern_name.lower()}',
                        base_score=1.5,
                        description=f"Bullish pattern: {pattern_name}"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 {pattern_name} pattern (bullish) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif pattern_signal == 'bearish':
                    signal_result = self._apply_signal(
                        signal_name=f'pattern_{pattern_name.lower()}',
                        base_score=-1.5,
                        description=f"Bearish pattern: {pattern_name}"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 {pattern_name} pattern (bearish) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
        
        # ===== PRICE MOMENTUM =====
        if data and len(data) >= 10:
            try:
                current_price = data[-1]['close']
                past_price = data[-10]['close']
                momentum = ((current_price - past_price) / past_price) * 100
                
                if momentum > 5:
                    # Strong bullish momentum
                    signal_result = self._apply_signal(
                        signal_name='strong_bullish_momentum',
                        base_score=2.0,
                        description=f"Strong bullish momentum (+{momentum:.1f}%)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 Strong momentum (+{momentum:.1f}% in 10 periods) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif momentum < -5:
                    # Strong bearish momentum
                    signal_result = self._apply_signal(
                        signal_name='strong_bearish_momentum',
                        base_score=-2.0,
                        description=f"Strong bearish momentum ({momentum:.1f}%)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 Strong momentum ({momentum:.1f}% in 10 periods) - "
                        f"Weight: {signal_result['weight']:.2f}x, "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif 2 < momentum <= 5:
                    # Moderate bullish momentum
                    signal_result = self._apply_signal(
                        signal_name='moderate_bullish_momentum',
                        base_score=1.0,
                        description=f"Moderate bullish momentum (+{momentum:.1f}%)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 Moderate momentum (+{momentum:.1f}%) - "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif -5 <= momentum < -2:
                    # Moderate bearish momentum
                    signal_result = self._apply_signal(
                        signal_name='moderate_bearish_momentum',
                        base_score=-1.0,
                        description=f"Moderate bearish momentum ({momentum:.1f}%)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 Moderate momentum ({momentum:.1f}%) - "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
            
            except (KeyError, IndexError, TypeError):
                pass
        
        # ===== TREND CONSISTENCY =====
        if data and len(data) >= 5:
            try:
                # Check if last 5 candles are consistently up or down
                closes = [candle['close'] for candle in data[-5:]]
                
                all_rising = all(closes[i] > closes[i-1] for i in range(1, len(closes)))
                all_falling = all(closes[i] < closes[i-1] for i in range(1, len(closes)))
                
                if all_rising:
                    signal_result = self._apply_signal(
                        signal_name='consistent_uptrend',
                        base_score=1.5,
                        description="Consistent uptrend (5 candles)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🟢 Consistent uptrend (5 rising candles) - "
                        f"Score: +{signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
                
                elif all_falling:
                    signal_result = self._apply_signal(
                        signal_name='consistent_downtrend',
                        base_score=-1.5,
                        description="Consistent downtrend (5 candles)"
                    )
                    score += signal_result['final_score']
                    reasoning.append(
                        f"🔴 Consistent downtrend (5 falling candles) - "
                        f"Score: {signal_result['final_score']:.2f}"
                    )
                    signals_used.append(signal_result)
            
            except (KeyError, IndexError, TypeError):
                pass
        
        # ===== DETERMINE VOTE =====
        if score >= 4:
            vote = 'BUY'
            confidence = 'HIGH'
        elif score >= 2:
            vote = 'BUY'
            confidence = 'MEDIUM'
        elif score > 0:
            vote = 'BUY'
            confidence = 'LOW'
        elif score <= -4:
            vote = 'SELL'
            confidence = 'HIGH'
        elif score <= -2:
            vote = 'SELL'
            confidence = 'MEDIUM'
        elif score < 0:
            vote = 'SELL'
            confidence = 'LOW'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
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
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LEARNING CONSULTANTS")
    print("=" * 60)
    
    # Initialize consultants
    c1 = C1_TechnicalAnalyst()
    c2 = C2_SentimentAnalyst()
    c3 = C3_RiskManager()
    c4 = C4_TrendAnalyst()
    
    print(f"\n✅ All consultants initialized!")
    print(f"  C1 ({c1.specialty}): {c1.performance['accuracy']:.1f}% accuracy, {c1.performance['weight']:.2f}x weight")
    print(f"  C2 ({c2.specialty}): {c2.performance['accuracy']:.1f}% accuracy, {c2.performance['weight']:.2f}x weight")
    print(f"  C3 ({c3.specialty}): {c3.performance['accuracy']:.1f}% accuracy, {c3.performance['weight']:.2f}x weight")
    print(f"  C4 ({c4.specialty}): {c4.performance['accuracy']:.1f}% accuracy, {c4.performance['weight']:.2f}x weight")
    
    # Example: Analyze market data
    example_indicators = {
        'rsi': 28.5,  # Oversold
        'macd': 0.5,
        'macd_signal': 0.3,
        'adx': 32,
        'di_plus': 28,
        'di_minus': 18,
        'stoch_k': 22,
        'sma_20': 50000,
        'sma_50': 49000,
        'sma_200': 48000,
        'close': 50100,
        'atr': 1500,
        'volume': 1000000,
        'volume_sma': 800000
    }
    
    print("\n📊 Example Analysis:")
    print("-" * 60)
    
    result = c1.analyze(None, example_indicators, None)
    print(f"\n{result['consultant']} ({result['specialty']}):")
    print(f"  Vote: {result['vote']} ({result['confidence']} confidence)")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Weight: {result['consultant_weight']:.2f}x")
    print(f"  Weighted Power: {result['weighted_vote_power']:.2f}")
    print(f"  History: {result['accuracy_history']}")
    print(f"  Reasoning:")
    for reason in result['reasoning'][:3]:  # Show first 3
        print(f"    {reason}")
    
    print("\n✅ Consultants ready for committee meetings!")
