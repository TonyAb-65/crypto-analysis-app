# committee_meeting.py - MODIFIED TO WORK WITH FUNCTION-BASED CONSULTANTS
# Uses existing consultant functions from consultants.py

from consultants import (
    consultant_c1_pattern_structure,
    consultant_c2_trend_momentum,
    consultant_c3_risk_warnings,
    consultant_c4_news_sentiment,
    run_consultant_meeting  # Keep original as fallback
)
from datetime import datetime
import json

# ============================================================================
# COMMITTEE MEETING ORCHESTRATOR (ADAPTED FOR FUNCTIONS)
# ============================================================================

class CommitteeMeeting:
    """
    Orchestrates committee decision-making using existing consultant functions.
    
    This version WRAPS your existing consultant functions instead of creating
    new consultant classes. It adds:
    - Performance tracking
    - Weighted voting
    - Learning capabilities
    
    WITHOUT changing your existing consultant logic.
    """
    
    def __init__(self, enable_learning=True):
        """
        Initialize committee with performance tracking
        
        Args:
            enable_learning: If True, tracks consultant performance
        """
        print("🏛️ Initializing Committee Meeting System (Function-Based)...")
        
        # Initialize learning system
        self.enable_learning = enable_learning
        if enable_learning:
            try:
                from committee_learning import CommitteeLearningSystem
                self.learning_system = CommitteeLearningSystem()
            except ImportError:
                print("⚠️ Committee learning not available - tracking disabled")
                self.enable_learning = False
                self.learning_system = None
        else:
            self.learning_system = None
        
        # Load consultant performance from database
        self.consultant_performance = self._load_performance()
        
        # Confidence multipliers (HIGH confidence votes count more)
        self.confidence_multipliers = {
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        print(f"✅ Committee initialized")
        print(f"  C1 (Pattern/Structure): {self.consultant_performance['C1']['weight']:.2f}x weight")
        print(f"  C2 (Momentum): {self.consultant_performance['C2']['weight']:.2f}x weight")
        print(f"  C3 (Risk): {self.consultant_performance['C3']['weight']:.2f}x weight")
        print(f"  C4 (Sentiment): {self.consultant_performance['C4']['weight']:.2f}x weight")
        print(f"  Learning: {'Enabled ✅' if self.enable_learning else 'Disabled ❌'}\n")
    
    def _load_performance(self):
        """Load consultant performance from database"""
        try:
            import sqlite3
            from database import DB_PATH
            
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Try to get performance from database
            cursor.execute("""
                SELECT consultant_name, accuracy_rate, current_weight, total_votes
                FROM consultant_performance
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            # Build performance dict
            performance = {}
            for name, accuracy, weight, total in results:
                performance[name] = {
                    'accuracy': accuracy,
                    'weight': weight,
                    'total_votes': total
                }
            
            # Fill in any missing consultants with defaults
            for name in ['C1', 'C2', 'C3', 'C4']:
                if name not in performance:
                    performance[name] = {
                        'accuracy': 50.0,
                        'weight': 1.0,
                        'total_votes': 0
                    }
            
            return performance
            
        except Exception as e:
            print(f"⚠️ Could not load performance, using defaults: {e}")
            # Return defaults
            return {
                'C1': {'accuracy': 50.0, 'weight': 1.0, 'total_votes': 0},
                'C2': {'accuracy': 50.0, 'weight': 1.0, 'total_votes': 0},
                'C3': {'accuracy': 50.0, 'weight': 1.0, 'total_votes': 0},
                'C4': {'accuracy': 50.0, 'weight': 1.0, 'total_votes': 0}
            }
    
    # ========================================================================
    # MAIN MEETING FUNCTION
    # ========================================================================
    
    def hold_meeting(self, data, indicators, signals, news_data=None, 
                     risk_metrics=None, patterns=None, symbol=None, 
                     current_price=None, market_type='crypto'):
        """
        🏛️ HOLD COMMITTEE MEETING
        
        Calls your existing consultant functions and applies weighted voting.
        
        Args:
            data: Historical market data (OHLCV DataFrame)
            indicators: Technical indicators dict
            signals: Trading signals dict
            news_data: News/sentiment data (optional)
            risk_metrics: Risk analysis data (optional)
            patterns: Candlestick patterns (optional)
            symbol: Trading pair (e.g., 'BTC/USD')
            current_price: Current market price
            market_type: 'crypto', 'forex', 'metals'
        
        Returns:
            dict: Complete committee decision with all details
        """
        
        print(f"\n{'='*70}")
        print(f"🏛️ COMMITTEE MEETING IN SESSION")
        if symbol:
            print(f"Symbol: {symbol} | Price: ${current_price:.2f}" if current_price else f"Symbol: {symbol}")
        print(f"{'='*70}\n")
        
        # Determine interval for C1
        interval = '1h'  # Default, can be made dynamic
        
        # ====================================================================
        # CALL EXISTING CONSULTANT FUNCTIONS
        # ====================================================================
        
        print("📊 Consultants analyzing...\n")
        
        # C1: Pattern & Structure
        print("🔧 C1 (Pattern/Structure) analyzing...")
        c1_result = consultant_c1_pattern_structure(data, symbol, interval)
        c1_vote = self._convert_c1_to_vote(c1_result)
        c1_vote['consultant_weight'] = self.consultant_performance['C1']['weight']
        self._print_vote(c1_vote)
        
        # C2: Momentum
        print("\n📈 C2 (Momentum) analyzing...")
        c2_result = consultant_c2_trend_momentum(data, symbol, c1_result)
        c2_vote = self._convert_c2_to_vote(c2_result)
        c2_vote['consultant_weight'] = self.consultant_performance['C2']['weight']
        self._print_vote(c2_vote)
        
        # C3: Risk
        print("\n⚖️ C3 (Risk) analyzing...")
        # Build warnings dict from indicators if needed
        warnings = self._build_warnings(indicators) if indicators else {}
        c3_result = consultant_c3_risk_warnings(data, symbol, warnings)
        c3_vote = self._convert_c3_to_vote(c3_result)
        c3_vote['consultant_weight'] = self.consultant_performance['C3']['weight']
        self._print_vote(c3_vote)
        
        # C4: Sentiment
        print("\n📰 C4 (Sentiment) analyzing...")
        c4_result = consultant_c4_news_sentiment(symbol, news_data)
        c4_vote = self._convert_c4_to_vote(c4_result)
        c4_vote['consultant_weight'] = self.consultant_performance['C4']['weight']
        self._print_vote(c4_vote)
        
        # ====================================================================
        # WEIGHTED VOTING
        # ====================================================================
        
        print(f"\n{'='*70}")
        print("🗳️ CALCULATING WEIGHTED VOTES")
        print(f"{'='*70}\n")
        
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        
        for vote in [c1_vote, c2_vote, c3_vote, c4_vote]:
            decision = vote['vote']
            weight = vote['consultant_weight']
            confidence = vote['confidence']
            
            # Apply confidence multiplier
            conf_mult = self.confidence_multipliers.get(confidence, 1.0)
            effective_weight = weight * conf_mult
            
            votes[decision] += effective_weight
            total_weight += effective_weight
            
            print(f"{vote['consultant']} → {decision} (weight: {effective_weight:.2f})")
        
        # Determine winner
        max_votes = max(votes.values())
        winner = [k for k, v in votes.items() if v == max_votes][0]
        consensus = (max_votes / total_weight * 100) if total_weight > 0 else 0
        
        # Determine strength
        if consensus >= 75:
            strength = "STRONG"
        elif consensus >= 60:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        print(f"\n{'='*70}")
        print(f"📊 FINAL DECISION: {strength} {winner}")
        print(f"Consensus: {consensus:.1f}%")
        print(f"{'='*70}\n")
        
        # Build result
        result = {
            'final_decision': winner,
            'decision_strength': strength,
            'consensus_level': consensus,
            'consensus_percentage': f"{consensus:.1f}%",
            'votes': votes,
            'total_weight': total_weight,
            'C1': c1_vote,
            'C2': c2_vote,
            'C3': c3_vote,
            'C4': c4_vote,
            'has_conflict': len([k for k, v in votes.items() if v > 0]) > 1,
            'conflicts': [],
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'current_price': current_price,
            'market_type': market_type,
            'summary': f"{strength} {winner} ({consensus:.1f}% consensus)",
            'summary_short': f"{strength} {winner}"
        }
        
        # Record decision if learning enabled
        if self.enable_learning and self.learning_system and symbol and current_price:
            try:
                decision_id = self.learning_system.record_decision(
                    committee_result=result,
                    symbol=symbol,
                    price=current_price,
                    market_type=market_type
                )
                result['decision_id'] = decision_id
                print(f"📝 Decision recorded for learning (ID: {decision_id})\n")
            except Exception as e:
                print(f"⚠️ Could not record decision: {e}")
        
        return result
    
    # ========================================================================
    # CONVERSION HELPERS (Function Results → Vote Format)
    # ========================================================================
    
    def _convert_c1_to_vote(self, c1_result):
        """Convert C1 function result to vote format"""
        signal = c1_result.get('signal', 'MID_RANGE')
        strength = c1_result.get('strength', 5)
        
        # C1 identifies location, not direction
        # So we interpret based on whether at support/resistance
        if 'SUPPORT' in signal:
            vote = 'BUY'  # At support = potential bounce
            confidence = 'HIGH' if strength >= 8 else 'MEDIUM'
        elif 'RESISTANCE' in signal:
            vote = 'SELL'  # At resistance = potential rejection
            confidence = 'HIGH' if strength >= 8 else 'MEDIUM'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
        return {
            'consultant': 'C1',
            'specialty': 'Pattern/Structure',
            'vote': vote,
            'confidence': confidence,
            'score': strength,
            'reasoning': [c1_result.get('reasoning', 'Pattern analysis')],
            'consultant_weight': 1.0,  # Will be overwritten
            'accuracy_history': f"{self.consultant_performance['C1']['accuracy']:.1f}% ({self.consultant_performance['C1']['total_votes']} votes)"
        }
    
    def _convert_c2_to_vote(self, c2_result):
        """Convert C2 function result to vote format"""
        signal = c2_result.get('signal', 'NO_CONFIRMATION')
        strength = c2_result.get('strength', 0)
        
        if 'BULLISH' in signal or 'REVERSAL_UP' in signal:
            vote = 'BUY'
            confidence = 'HIGH' if strength >= 8 else 'MEDIUM'
        elif 'BEARISH' in signal or 'REVERSAL_DOWN' in signal:
            vote = 'SELL'
            confidence = 'HIGH' if strength >= 8 else 'MEDIUM'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
        return {
            'consultant': 'C2',
            'specialty': 'Momentum',
            'vote': vote,
            'confidence': confidence,
            'score': strength,
            'reasoning': [c2_result.get('reasoning', 'Momentum analysis')],
            'consultant_weight': 1.0,
            'accuracy_history': f"{self.consultant_performance['C2']['accuracy']:.1f}% ({self.consultant_performance['C2']['total_votes']} votes)"
        }
    
    def _convert_c3_to_vote(self, c3_result):
        """Convert C3 function result to vote format"""
        signal = c3_result.get('signal', 'ACCEPTABLE')
        
        # C3 is risk manager - recommends caution
        if 'HIGH_RISK' in signal:
            vote = 'HOLD'  # Too risky
            confidence = 'HIGH'
        elif 'MODERATE' in signal:
            vote = 'HOLD'
            confidence = 'MEDIUM'
        else:
            vote = 'HOLD'  # C3 rarely votes BUY/SELL, mostly HOLD
            confidence = 'LOW'
        
        return {
            'consultant': 'C3',
            'specialty': 'Risk Management',
            'vote': vote,
            'confidence': confidence,
            'score': c3_result.get('strength', 5),
            'reasoning': [c3_result.get('reasoning', 'Risk assessment')],
            'consultant_weight': 1.0,
            'accuracy_history': f"{self.consultant_performance['C3']['accuracy']:.1f}% ({self.consultant_performance['C3']['total_votes']} votes)"
        }
    
    def _convert_c4_to_vote(self, c4_result):
        """Convert C4 function result to vote format"""
        signal = c4_result.get('signal', 'NO_NEWS')
        
        if 'BULLISH' in signal:
            vote = 'BUY'
            confidence = 'HIGH'
        elif 'BEARISH' in signal:
            vote = 'SELL'
            confidence = 'HIGH'
        else:
            vote = 'HOLD'
            confidence = 'LOW'
        
        return {
            'consultant': 'C4',
            'specialty': 'Sentiment',
            'vote': vote,
            'confidence': confidence,
            'score': c4_result.get('weight', 5),
            'reasoning': [c4_result.get('reasoning', 'Sentiment analysis')],
            'consultant_weight': 1.0,
            'accuracy_history': f"{self.consultant_performance['C4']['accuracy']:.1f}% ({self.consultant_performance['C4']['total_votes']} votes)"
        }
    
    def _build_warnings(self, indicators):
        """Build warnings dict from indicators"""
        warnings = {}
        
        if indicators:
            # Add warning logic here if needed
            # For now, return empty
            pass
        
        return warnings
    
    def _print_vote(self, vote):
        """Print consultant vote"""
        icon = "🟢" if vote['vote'] == 'BUY' else "🔴" if vote['vote'] == 'SELL' else "🟡"
        print(f"{icon} Vote: {vote['vote']} ({vote['confidence']} confidence)")
        print(f"  Weight: {vote['consultant_weight']:.2f}x")
        print(f"  History: {vote['accuracy_history']}")
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def reload_consultant_performance(self):
        """Reload consultant performance from database"""
        print("🔄 Reloading consultant performance...")
        self.consultant_performance = self._load_performance()
        print("✅ Performance reloaded")
    
    def get_consultant_rankings(self):
        """Get current consultant rankings"""
        rankings = []
        for name in ['C1', 'C2', 'C3', 'C4']:
            perf = self.consultant_performance[name]
            rankings.append({
                'rank': 0,  # Will be calculated
                'name': name,
                'specialty': {
                    'C1': 'Pattern/Structure',
                    'C2': 'Momentum',
                    'C3': 'Risk',
                    'C4': 'Sentiment'
                }[name],
                'accuracy': perf['accuracy'],
                'weight': perf['weight'],
                'total_votes': perf['total_votes'],
                'wins': 0,
                'losses': 0,
                'streak': 0
            })
        
        # Sort by accuracy
        rankings.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Assign ranks
        for i, r in enumerate(rankings, 1):
            r['rank'] = i
        
        return rankings


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def quick_committee_decision(symbol, indicators, signals, news_data=None, current_price=None):
    """Quick committee decision without learning"""
    committee = CommitteeMeeting(enable_learning=False)
    result = committee.hold_meeting(
        data=None,
        indicators=indicators,
        signals=signals,
        news_data=news_data,
        symbol=symbol,
        current_price=current_price
    )
    return result['final_decision']
