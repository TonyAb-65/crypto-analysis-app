# committee_meeting.py - STEP 4: COMMITTEE MEETING SYSTEM
# The orchestrator that brings all consultants together with weighted voting

from consultants import (
    C1_TechnicalAnalyst,
    C2_SentimentAnalyst, 
    C3_RiskManager,
    C4_TrendAnalyst
)
from committee_learning import CommitteeLearningSystem
from datetime import datetime
import json

# ============================================================================
# COMMITTEE MEETING ORCHESTRATOR
# ============================================================================

class CommitteeMeeting:
    """
    Orchestrates the entire committee decision-making process.
    
    This is the "boardroom" where all 4 consultants meet, discuss,
    vote, and reach a consensus using weighted voting based on their
    historical performance.
    
    Features:
    - Weighted voting (better consultants have more influence)
    - Confidence-based adjustments (HIGH confidence = 1.5x weight)
    - Conflict detection (alerts when consultants disagree)
    - Transparent reasoning (see why each consultant voted)
    - Automatic decision recording (for learning)
    """
    
    def __init__(self, enable_learning=True):
        """
        Initialize committee with all 4 consultants
        
        Args:
            enable_learning: If True, automatically records decisions for learning
        """
        print("🏛️ Initializing Committee Meeting System...")
        
        # Initialize all consultants
        self.c1 = C1_TechnicalAnalyst()
        self.c2 = C2_SentimentAnalyst()
        self.c3 = C3_RiskManager()
        self.c4 = C4_TrendAnalyst()
        
        # Initialize learning system
        self.enable_learning = enable_learning
        if enable_learning:
            self.learning_system = CommitteeLearningSystem()
        else:
            self.learning_system = None
        
        # Confidence multipliers (HIGH confidence votes count more)
        self.confidence_multipliers = {
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        print(f"✅ Committee initialized:")
        print(f"  C1 ({self.c1.specialty}): {self.c1.performance['weight']:.2f}x weight")
        print(f"  C2 ({self.c2.specialty}): {self.c2.performance['weight']:.2f}x weight")
        print(f"  C3 ({self.c3.specialty}): {self.c3.performance['weight']:.2f}x weight")
        print(f"  C4 ({self.c4.specialty}): {self.c4.performance['weight']:.2f}x weight")
        print(f"  Learning: {'Enabled ✅' if enable_learning else 'Disabled ❌'}\n")
    
    # ========================================================================
    # MAIN MEETING FUNCTION
    # ========================================================================
    
    def hold_meeting(self, data, indicators, signals, news_data=None, 
                     risk_metrics=None, patterns=None, symbol=None, 
                     current_price=None, market_type='crypto'):
        """
        🏛️ HOLD COMMITTEE MEETING
        
        The main function that orchestrates the entire decision-making process:
        1. Each consultant analyzes independently
        2. Consultants cast their votes
        3. Weighted voting determines final decision
        4. Conflict detection and consensus calculation
        5. Generate human-readable summary
        6. Optionally record decision for learning
        
        Args:
            data: Historical market data (OHLCV DataFrame)
            indicators: Technical indicators dict
            signals: Trading signals dict
            news_data: News/sentiment data (optional, for C2)
            risk_metrics: Risk analysis data (optional, for C3)
            patterns: Candlestick patterns (optional, for C4)
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
        
        # ====================================================================
        # STEP 1: EACH CONSULTANT ANALYZES INDEPENDENTLY
        # ====================================================================
        
        print("📊 Consultants analyzing market data...\n")
        
        # C1: Technical Analysis
        print("🔧 C1 (Technical Analyst) analyzing...")
        c1_result = self.c1.analyze(data, indicators, signals)
        self._print_consultant_vote(c1_result)
        
        # C2: Sentiment Analysis (if news data available)
        print("\n📰 C2 (Sentiment Analyst) analyzing...")
        if news_data:
            c2_result = self.c2.analyze(data, indicators, signals, news_data=news_data)
        else:
            c2_result = {
                'consultant': 'C2',
                'specialty': 'Market Sentiment',
                'vote': 'HOLD',
                'confidence': 'LOW',
                'score': 0,
                'consultant_weight': self.c2.performance['weight'],
                'weighted_vote_power': 0,
                'reasoning': ['No news data available'],
                'signals_used': [],
                'accuracy_history': f"{self.c2.performance['accuracy']:.1f}% over {self.c2.performance['total_votes']} votes",
                'current_streak': self.c2.performance['current_streak']
            }
        self._print_consultant_vote(c2_result)
        
        # C3: Risk Management
        print("\n⚖️ C3 (Risk Manager) analyzing...")
        c3_result = self.c3.analyze(data, indicators, signals, risk_metrics=risk_metrics)
        self._print_consultant_vote(c3_result)
        
        # C4: Trend Analysis
        print("\n📈 C4 (Trend Analyst) analyzing...")
        c4_result = self.c4.analyze(data, indicators, signals, patterns=patterns)
        self._print_consultant_vote(c4_result)
        
        # ====================================================================
        # STEP 2: WEIGHTED VOTING
        # ====================================================================
        
        print(f"\n{'='*70}")
        print("🗳️ CALCULATING WEIGHTED VOTES")
        print(f"{'='*70}\n")
        
        # Initialize vote tallies
        votes = {
            'BUY': 0,
            'SELL': 0,
            'HOLD': 0
        }
        
        total_weight = 0
        consultant_results = [c1_result, c2_result, c3_result, c4_result]
        
        # Calculate weighted votes
        for result in consultant_results:
            vote = result['vote']
            consultant_weight = result['consultant_weight']
            confidence = result['confidence']
            
            # Apply confidence multiplier
            confidence_multiplier = self.confidence_multipliers.get(confidence, 1.0)
            
            # Effective weight = consultant_weight × confidence_multiplier
            effective_weight = consultant_weight * confidence_multiplier
            
            # Add to vote tally
            votes[vote] += effective_weight
            total_weight += effective_weight
            
            # Print voting breakdown
            print(f"{result['consultant']} ({result['specialty']}):")
            print(f"  Vote: {vote} ({confidence} confidence)")
            print(f"  Base Weight: {consultant_weight:.2f}x")
            print(f"  Confidence Multiplier: {confidence_multiplier}x")
            print(f"  Effective Weight: {effective_weight:.2f}")
            print(f"  Vote Power: {effective_weight:.2f} points to {vote}")
            print()
        
        # ====================================================================
        # STEP 3: DETERMINE FINAL DECISION
        # ====================================================================
        
        print(f"{'='*70}")
        print("📊 VOTE TALLY")
        print(f"{'='*70}")
        print(f"BUY:  {votes['BUY']:.2f} points")
        print(f"SELL: {votes['SELL']:.2f} points")
        print(f"HOLD: {votes['HOLD']:.2f} points")
        print(f"Total Weight: {total_weight:.2f}")
        print(f"{'='*70}\n")
        
        # Find winner
        max_votes = max(votes.values())
        winner = [k for k, v in votes.items() if v == max_votes][0]
        
        # Calculate consensus level (percentage of votes for winner)
        consensus_level = (max_votes / total_weight * 100) if total_weight > 0 else 0
        
        # Determine decision strength
        if consensus_level >= 75:
            strength = "STRONG"
            strength_icon = "🟢" if winner == 'BUY' else "🔴" if winner == 'SELL' else "🟡"
        elif consensus_level >= 60:
            strength = "MODERATE"
            strength_icon = "🟡"
        else:
            strength = "WEAK"
            strength_icon = "⚠️"
        
        # ====================================================================
        # STEP 4: CONFLICT DETECTION
        # ====================================================================
        
        conflicts = self._detect_conflicts(consultant_results)
        has_conflict = len(conflicts) > 0
        
        if has_conflict:
            print("⚠️ CONFLICTS DETECTED:")
            for conflict in conflicts:
                print(f"  • {conflict}")
            print()
        
        # ====================================================================
        # STEP 5: BUILD RESULT
        # ====================================================================
        
        result = {
            # Final decision
            'final_decision': winner,
            'decision_strength': strength,
            'consensus_level': consensus_level,
            'consensus_percentage': f"{consensus_level:.1f}%",
            
            # Voting details
            'votes': votes,
            'total_weight': total_weight,
            
            # Individual consultant results
            'C1': c1_result,
            'C2': c2_result,
            'C3': c3_result,
            'C4': c4_result,
            
            # Conflict detection
            'has_conflict': has_conflict,
            'conflicts': conflicts,
            
            # Metadata
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'current_price': current_price,
            'market_type': market_type,
            'indicators': indicators,
            'learning_enabled': self.enable_learning
        }
        
        # Generate human-readable summary
        result['summary'] = self._generate_summary(result)
        result['summary_short'] = self._generate_short_summary(result)
        
        # ====================================================================
        # STEP 6: PRINT FINAL DECISION
        # ====================================================================
        
        print(f"{'='*70}")
        print(f"{strength_icon} FINAL COMMITTEE DECISION")
        print(f"{'='*70}")
        print(f"Decision: {strength} {winner}")
        print(f"Consensus: {consensus_level:.1f}%")
        
        if has_conflict:
            print(f"⚠️ Warning: Consultants disagree - consider waiting for clarity")
        
        print(f"{'='*70}\n")
        
        # ====================================================================
        # STEP 7: RECORD DECISION (if learning enabled)
        # ====================================================================
        
        if self.enable_learning and symbol and current_price:
            decision_id = self.learning_system.record_decision(
                committee_result=result,
                symbol=symbol,
                price=current_price,
                market_type=market_type
            )
            result['decision_id'] = decision_id
            print(f"📝 Decision recorded for learning (ID: {decision_id})\n")
        
        return result
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _print_consultant_vote(self, result):
        """Print a consultant's vote in a readable format"""
        vote_icon = "🟢" if result['vote'] == 'BUY' else "🔴" if result['vote'] == 'SELL' else "🟡"
        
        print(f"{vote_icon} Vote: {result['vote']} ({result['confidence']} confidence)")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Weight: {result['consultant_weight']:.2f}x")
        print(f"  History: {result['accuracy_history']}")
        
        # Print top 3 reasons
        if result['reasoning']:
            print(f"  Top Reasons:")
            for reason in result['reasoning'][:3]:
                print(f"    • {reason}")
    
    def _detect_conflicts(self, consultant_results):
        """
        Detect conflicts between consultants
        
        Returns:
            list: Conflict descriptions
        """
        conflicts = []
        
        # Get votes
        c1_vote = consultant_results[0]['vote']
        c2_vote = consultant_results[1]['vote']
        c3_vote = consultant_results[2]['vote']
        c4_vote = consultant_results[3]['vote']
        
        # Check for BUY vs SELL conflicts
        buy_votes = [v for v in [c1_vote, c2_vote, c3_vote, c4_vote] if v == 'BUY']
        sell_votes = [v for v in [c1_vote, c2_vote, c3_vote, c4_vote] if v == 'SELL']
        
        if len(buy_votes) > 0 and len(sell_votes) > 0:
            conflicts.append(f"Split opinion: {len(buy_votes)} want BUY, {len(sell_votes)} want SELL")
        
        # Check specific conflicts
        if c1_vote == 'BUY' and c3_vote == 'SELL':
            conflicts.append("Technical signals bullish but Risk Manager bearish")
        
        if c2_vote == 'BUY' and c4_vote == 'SELL':
            conflicts.append("Sentiment positive but Trend negative")
        
        if c1_vote == 'SELL' and c4_vote == 'BUY':
            conflicts.append("Technical signals bearish but Trend bullish")
        
        return conflicts
    
    def _generate_summary(self, result):
        """Generate detailed human-readable summary"""
        decision = result['final_decision']
        strength = result['decision_strength']
        consensus = result['consensus_level']
        
        lines = []
        
        # Header
        lines.append(f"{'='*70}")
        lines.append(f"COMMITTEE DECISION SUMMARY")
        lines.append(f"{'='*70}\n")
        
        # Decision
        icon = "🟢" if decision == 'BUY' else "🔴" if decision == 'SELL' else "🟡"
        lines.append(f"{icon} DECISION: {strength} {decision}")
        lines.append(f"Consensus: {consensus:.1f}%")
        
        if result['has_conflict']:
            lines.append(f"\n⚠️ WARNING: Consultants disagree")
            for conflict in result['conflicts']:
                lines.append(f"  • {conflict}")
        
        lines.append(f"\n{'-'*70}")
        lines.append("CONSULTANT VOTES:")
        lines.append(f"{'-'*70}")
        
        # Each consultant
        for consultant_name in ['C1', 'C2', 'C3', 'C4']:
            c_result = result[consultant_name]
            vote_icon = "✅" if c_result['vote'] == decision else "❌"
            
            lines.append(f"\n{vote_icon} {consultant_name} ({c_result['specialty']}):")
            lines.append(f"   Vote: {c_result['vote']} ({c_result['confidence']} confidence)")
            lines.append(f"   Weight: {c_result['consultant_weight']:.2f}x")
            lines.append(f"   Accuracy: {c_result['accuracy_history']}")
            
            # Top 2 reasons
            if c_result['reasoning']:
                lines.append(f"   Reasoning:")
                for reason in c_result['reasoning'][:2]:
                    lines.append(f"     • {reason}")
        
        lines.append(f"\n{'='*70}")
        
        return '\n'.join(lines)
    
    def _generate_short_summary(self, result):
        """Generate concise one-line summary"""
        decision = result['final_decision']
        strength = result['decision_strength']
        consensus = result['consensus_level']
        
        # Count votes
        buy_count = sum(1 for c in ['C1', 'C2', 'C3', 'C4'] if result[c]['vote'] == 'BUY')
        sell_count = sum(1 for c in ['C1', 'C2', 'C3', 'C4'] if result[c]['vote'] == 'SELL')
        hold_count = sum(1 for c in ['C1', 'C2', 'C3', 'C4'] if result[c]['vote'] == 'HOLD')
        
        vote_breakdown = f"{buy_count} BUY, {sell_count} SELL, {hold_count} HOLD"
        
        if result['has_conflict']:
            return f"⚠️ {strength} {decision} ({consensus:.0f}% consensus) - CONFLICTED - {vote_breakdown}"
        else:
            return f"✅ {strength} {decision} ({consensus:.0f}% consensus) - {vote_breakdown}"
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def reload_consultant_performance(self):
        """Reload all consultant performance from database (after learning)"""
        print("🔄 Reloading consultant performance...")
        self.c1.reload_performance()
        self.c2.reload_performance()
        self.c3.reload_performance()
        self.c4.reload_performance()
        print("✅ Performance reloaded for all consultants")
    
    def get_consultant_rankings(self):
        """Get current consultant rankings"""
        consultants = [
            ('C1', self.c1),
            ('C2', self.c2),
            ('C3', self.c3),
            ('C4', self.c4)
        ]
        
        # Sort by accuracy
        sorted_consultants = sorted(
            consultants,
            key=lambda x: x[1].performance['accuracy'],
            reverse=True
        )
        
        rankings = []
        for rank, (name, consultant) in enumerate(sorted_consultants, 1):
            rankings.append({
                'rank': rank,
                'name': name,
                'specialty': consultant.specialty,
                'accuracy': consultant.performance['accuracy'],
                'weight': consultant.performance['weight'],
                'total_votes': consultant.performance['total_votes'],
                'streak': consultant.performance['current_streak']
            })
        
        return rankings
    
    def print_committee_status(self):
        """Print current status of all consultants"""
        print(f"\n{'='*70}")
        print("COMMITTEE STATUS")
        print(f"{'='*70}\n")
        
        rankings = self.get_consultant_rankings()
        
        for rank_info in rankings:
            streak_icon = "🔥" if rank_info['streak'] > 0 else "❄️" if rank_info['streak'] < 0 else "➖"
            
            print(f"#{rank_info['rank']} {rank_info['name']} ({rank_info['specialty']}):")
            print(f"  Accuracy: {rank_info['accuracy']:.1f}%")
            print(f"  Weight: {rank_info['weight']:.2f}x")
            print(f"  Votes: {rank_info['total_votes']}")
            print(f"  Streak: {rank_info['streak']} {streak_icon}")
            print()
        
        print(f"{'='*70}\n")


# ============================================================================
# CONVENIENCE FUNCTION FOR QUICK MEETINGS
# ============================================================================

def quick_committee_decision(symbol, indicators, signals, 
                             news_data=None, current_price=None):
    """
    Convenience function for quick committee decisions
    
    Args:
        symbol: Trading pair
        indicators: Technical indicators dict
        signals: Trading signals dict
        news_data: Optional news data
        current_price: Current price
    
    Returns:
        str: Decision ('BUY', 'SELL', 'HOLD')
    """
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING COMMITTEE MEETING SYSTEM")
    print("=" * 70)
    
    # Initialize committee
    committee = CommitteeMeeting(enable_learning=True)
    
    # Example indicators (you would get these from your technical analysis)
    example_indicators = {
        'rsi': 28.5,
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_hist': 0.2,
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
        'volume_sma': 800000,
        'bb_upper': 51000,
        'bb_lower': 49000
    }
    
    example_signals = {
        'RSI': 'oversold',
        'MACD': 'bullish',
        'ADX': 'strong_uptrend'
    }
    
    # Hold meeting
    print("\n" + "="*70)
    print("EXAMPLE: Bitcoin Analysis")
    print("="*70 + "\n")
    
    result = committee.hold_meeting(
        data=None,
        indicators=example_indicators,
        signals=example_signals,
        symbol='BTC/USD',
        current_price=50100.0,
        market_type='crypto'
    )
    
    # Print results
    print("\n" + result['summary'])
    
    print("\n" + "="*70)
    print("SHORT SUMMARY:")
    print("="*70)
    print(result['summary_short'])
    
    print("\n✅ Committee meeting system working!")
    print("\nTo use in your app:")
    print("  1. Initialize: committee = CommitteeMeeting()")
    print("  2. Hold meeting: result = committee.hold_meeting(...)")
    print("  3. Use result['final_decision'] for trading")
    print("  4. System automatically records for learning!")
