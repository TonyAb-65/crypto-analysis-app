# committee_learning.py - STEP 3: COMMITTEE LEARNING SYSTEM
# The brain that makes consultants learn from every trade

from database_module import (
    get_session, ConsultantPerformance, SignalPerformance,
    CommitteeDecision, Trade, CommitteeMeetingLog
)
from datetime import datetime
import json

# ============================================================================
# COMMITTEE LEARNING SYSTEM
# ============================================================================

class CommitteeLearningSystem:
    """
    Manages learning for the entire committee system.
    
    This is the "teacher" that:
    1. Records every committee decision
    2. Links decisions to actual trades
    3. Evaluates which consultants were correct
    4. Updates consultant weights (0.5x to 3.0x)
    5. Updates signal weights (0.3x to 2.0x)
    6. Tracks performance history
    
    Called after every trade closes to trigger learning.
    """
    
    def __init__(self):
        """Initialize learning system with configuration"""
        # Learning rates (how fast weights change)
        self.consultant_learning_rate = 0.05  # 5% adjustment per trade
        self.signal_learning_rate = 0.05      # 5% adjustment per signal
        
        # Weight boundaries
        self.min_consultant_weight = 0.5   # Minimum consultant influence
        self.max_consultant_weight = 3.0   # Maximum consultant influence
        self.min_signal_weight = 0.3       # Minimum signal influence
        self.max_signal_weight = 2.0       # Maximum signal influence
        
        # Confidence multipliers
        self.confidence_multipliers = {
            'HIGH': 1.5,    # High confidence correct/wrong = 1.5x learning
            'MEDIUM': 1.0,  # Medium confidence = normal learning
            'LOW': 0.5      # Low confidence = 0.5x learning
        }
    
    # ========================================================================
    # DECISION RECORDING
    # ========================================================================
    
    def record_decision(self, committee_result, symbol, price, market_type='crypto'):
        """
        Store committee decision for future learning.
        Called immediately after committee meeting.
        
        Args:
            committee_result: Full result from committee.hold_meeting()
            symbol: Trading pair (e.g., 'BTC/USD')
            price: Current market price
            market_type: 'crypto', 'forex', 'metals'
        
        Returns:
            int: Decision ID (used to link trade later)
        """
        session = get_session()
        
        try:
            # Extract consultant votes
            c1_result = committee_result.get('C1', {})
            c2_result = committee_result.get('C2', {})
            c3_result = committee_result.get('C3', {})
            c4_result = committee_result.get('C4', {})
            
            # Create decision record
            decision = CommitteeDecision(
                symbol=symbol,
                market_type=market_type,
                price_at_decision=price,
                timestamp=datetime.utcnow(),
                
                # C1 vote details
                c1_vote=c1_result.get('vote', 'HOLD'),
                c1_confidence=c1_result.get('confidence', 'LOW'),
                c1_score=c1_result.get('score', 0),
                c1_weight=c1_result.get('consultant_weight', 1.0),
                c1_reasoning=json.dumps(c1_result) if c1_result else None,
                
                # C2 vote details
                c2_vote=c2_result.get('vote', 'HOLD'),
                c2_confidence=c2_result.get('confidence', 'LOW'),
                c2_score=c2_result.get('score', 0),
                c2_weight=c2_result.get('consultant_weight', 1.0),
                c2_reasoning=json.dumps(c2_result) if c2_result else None,
                
                # C3 vote details
                c3_vote=c3_result.get('vote', 'HOLD'),
                c3_confidence=c3_result.get('confidence', 'LOW'),
                c3_score=c3_result.get('score', 0),
                c3_weight=c3_result.get('consultant_weight', 1.0),
                c3_reasoning=json.dumps(c3_result) if c3_result else None,
                
                # C4 vote details
                c4_vote=c4_result.get('vote', 'HOLD'),
                c4_confidence=c4_result.get('confidence', 'LOW'),
                c4_score=c4_result.get('score', 0),
                c4_weight=c4_result.get('consultant_weight', 1.0),
                c4_reasoning=json.dumps(c4_result) if c4_result else None,
                
                # Committee final decision
                final_decision=committee_result.get('final_decision', 'HOLD'),
                consensus_level=committee_result.get('consensus_level', 0),
                total_weighted_votes=json.dumps(committee_result.get('votes', {})),
                
                # Recommendations (if provided)
                recommended_entry=committee_result.get('recommended_entry'),
                recommended_stop_loss=committee_result.get('recommended_stop_loss'),
                recommended_take_profit=committee_result.get('recommended_take_profit'),
                
                # Indicators snapshot
                indicators_snapshot=json.dumps(committee_result.get('indicators', {}))
            )
            
            session.add(decision)
            session.commit()
            
            decision_id = decision.id
            
            print(f"📝 Decision recorded: ID={decision_id}, {symbol}, {committee_result['final_decision']}")
            
            return decision_id
        
        except Exception as e:
            session.rollback()
            print(f"❌ Error recording decision: {e}")
            return None
        
        finally:
            session.close()
    
    def link_decision_to_trade(self, decision_id, trade_id):
        """
        Link a committee decision to an actual trade.
        Called when user executes the committee's recommendation.
        
        Args:
            decision_id: ID from record_decision()
            trade_id: ID of the trade in trades table
        
        Returns:
            bool: Success
        """
        session = get_session()
        
        try:
            decision = session.query(CommitteeDecision).filter(
                CommitteeDecision.id == decision_id
            ).first()
            
            if decision:
                decision.trade_id = trade_id
                session.commit()
                print(f"🔗 Linked decision {decision_id} to trade {trade_id}")
                return True
            else:
                print(f"⚠️ Decision {decision_id} not found")
                return False
        
        except Exception as e:
            session.rollback()
            print(f"❌ Error linking decision to trade: {e}")
            return False
        
        finally:
            session.close()
    
    # ========================================================================
    # LEARNING FROM TRADES
    # ========================================================================
    
    def learn_from_trade(self, trade_id):
        """
        🧠 MAIN LEARNING FUNCTION
        
        Called when a trade closes. This triggers the entire learning process:
        1. Find the committee decision for this trade
        2. Determine the outcome (WIN/LOSS)
        3. Evaluate which consultants were correct
        4. Update consultant weights
        5. Update signal weights
        6. Track performance history
        
        Args:
            trade_id: ID of the closed trade
        
        Returns:
            bool: Success
        """
        session = get_session()
        
        try:
            # Get the trade
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            
            if not trade:
                print(f"⚠️ Trade {trade_id} not found")
                return False
            
            # Get the committee decision for this trade
            decision = session.query(CommitteeDecision).filter(
                CommitteeDecision.trade_id == trade_id
            ).first()
            
            if not decision:
                print(f"⚠️ No committee decision found for trade {trade_id}")
                print(f"   Trade may not have been based on committee recommendation")
                return False
            
            # Determine outcome
            outcome = self._determine_outcome(trade)
            
            if not outcome:
                print(f"⚠️ Cannot determine outcome for trade {trade_id}")
                return False
            
            print(f"\n{'='*60}")
            print(f"🧠 LEARNING FROM TRADE #{trade_id}")
            print(f"{'='*60}")
            print(f"Symbol: {trade.symbol}")
            print(f"Type: {trade.trade_type}")
            print(f"Entry: ${trade.entry_price:.2f}")
            print(f"Exit: ${trade.exit_price:.2f}")
            print(f"Outcome: {outcome}")
            print(f"P&L: {trade.profit_loss_percentage:.2f}%")
            print(f"Committee Decision: {decision.final_decision}")
            print(f"-" * 60)
            
            # Update decision record with outcome
            decision.actual_outcome = outcome
            decision.outcome_determined_at = datetime.utcnow()
            decision.price_at_outcome = trade.exit_price
            decision.profit_loss_pct = trade.profit_loss_percentage
            
            # Calculate time to outcome
            time_diff = decision.outcome_determined_at - decision.timestamp
            decision.hours_to_outcome = time_diff.total_seconds() / 3600
            
            # Determine if committee decision was correct
            decision.was_correct = self._was_decision_correct(
                decision.final_decision,
                trade.trade_type,
                outcome
            )
            
            print(f"Committee was: {'✅ CORRECT' if decision.was_correct else '❌ WRONG'}")
            print(f"-" * 60)
            
            # Learn from each consultant's vote
            consultants_data = [
                ('C1', decision.c1_vote, decision.c1_confidence, decision.c1_reasoning),
                ('C2', decision.c2_vote, decision.c2_confidence, decision.c2_reasoning),
                ('C3', decision.c3_vote, decision.c3_confidence, decision.c3_reasoning),
                ('C4', decision.c4_vote, decision.c4_confidence, decision.c4_reasoning)
            ]
            
            for consultant_name, vote, confidence, reasoning_json in consultants_data:
                # Skip HOLD votes (neutral, no learning)
                if vote == 'HOLD':
                    print(f"{consultant_name}: HOLD vote - no learning")
                    continue
                
                # Determine if this consultant was correct
                was_correct = self._was_consultant_correct(
                    vote, trade.trade_type, outcome
                )
                
                # Update consultant performance
                self._update_consultant_performance(
                    consultant_name, was_correct, confidence
                )
                
                # Update signal performance
                if reasoning_json:
                    try:
                        reasoning = json.loads(reasoning_json)
                        signals_used = reasoning.get('signals_used', [])
                        
                        for signal in signals_used:
                            self._update_signal_performance(
                                consultant_name,
                                signal.get('signal'),
                                was_correct
                            )
                    except json.JSONDecodeError:
                        pass
            
            session.commit()
            
            print(f"{'='*60}")
            print(f"✅ Learning complete for trade #{trade_id}")
            print(f"{'='*60}\n")
            
            # Show updated performance summary
            self._print_learning_summary()
            
            return True
        
        except Exception as e:
            session.rollback()
            print(f"❌ Error learning from trade: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            session.close()
    
    def _determine_outcome(self, trade):
        """
        Determine if trade was WIN or LOSS
        
        Args:
            trade: Trade object
        
        Returns:
            str: 'WIN' or 'LOSS' or None
        """
        if trade.outcome:
            # Outcome already recorded
            return trade.outcome.upper()
        
        # Calculate based on P&L
        if trade.profit_loss_percentage is not None:
            if trade.profit_loss_percentage > 0:
                return 'WIN'
            elif trade.profit_loss_percentage < 0:
                return 'LOSS'
        
        # Fallback to profit_loss
        if trade.profit_loss is not None:
            if trade.profit_loss > 0:
                return 'WIN'
            elif trade.profit_loss < 0:
                return 'LOSS'
        
        return None
    
    def _was_decision_correct(self, decision, trade_type, outcome):
        """
        Check if committee decision was correct
        
        Args:
            decision: 'BUY', 'SELL', 'HOLD'
            trade_type: 'LONG', 'SHORT'
            outcome: 'WIN', 'LOSS'
        
        Returns:
            bool: True if correct
        """
        # For LONG trades: BUY decision + WIN = correct
        if trade_type == 'LONG':
            return decision == 'BUY' and outcome == 'WIN'
        
        # For SHORT trades: SELL decision + WIN = correct
        elif trade_type == 'SHORT':
            return decision == 'SELL' and outcome == 'WIN'
        
        return False
    
    def _was_consultant_correct(self, vote, trade_type, outcome):
        """
        Check if individual consultant was correct
        
        Args:
            vote: 'BUY', 'SELL', 'HOLD'
            trade_type: 'LONG', 'SHORT'
            outcome: 'WIN', 'LOSS'
        
        Returns:
            bool: True if correct
        """
        # LONG trade logic
        if trade_type == 'LONG':
            if vote == 'BUY' and outcome == 'WIN':
                return True  # Recommended buy, trade won
            elif vote == 'SELL' and outcome == 'LOSS':
                return True  # Recommended against it, trade lost
            else:
                return False
        
        # SHORT trade logic
        elif trade_type == 'SHORT':
            if vote == 'SELL' and outcome == 'WIN':
                return True  # Recommended sell/short, trade won
            elif vote == 'BUY' and outcome == 'LOSS':
                return True  # Recommended against it, trade lost
            else:
                return False
        
        return False
    
    # ========================================================================
    # CONSULTANT PERFORMANCE UPDATES
    # ========================================================================
    
    def _update_consultant_performance(self, consultant_name, was_correct, confidence):
        """
        Update consultant's performance metrics and dynamic weight
        
        Args:
            consultant_name: 'C1', 'C2', 'C3', or 'C4'
            was_correct: bool
            confidence: 'HIGH', 'MEDIUM', 'LOW'
        """
        session = get_session()
        
        try:
            # Get consultant performance record
            perf = session.query(ConsultantPerformance).filter(
                ConsultantPerformance.consultant_name == consultant_name
            ).first()
            
            if not perf:
                # Create new record
                perf = ConsultantPerformance(
                    consultant_name=consultant_name,
                    current_weight=1.0,
                    accuracy_rate=50.0
                )
                session.add(perf)
            
            # Store old values for comparison
            old_weight = perf.current_weight
            old_accuracy = perf.accuracy_rate
            
            # Update vote counts
            perf.total_votes += 1
            
            if was_correct:
                perf.correct_votes += 1
                
                # Update confidence-specific stats
                if confidence == 'HIGH':
                    perf.high_confidence_votes += 1
                    perf.high_confidence_correct += 1
                elif confidence == 'MEDIUM':
                    perf.medium_confidence_votes += 1
                    perf.medium_confidence_correct += 1
                elif confidence == 'LOW':
                    perf.low_confidence_votes += 1
                    perf.low_confidence_correct += 1
                
                # Update streak
                if perf.current_streak >= 0:
                    perf.current_streak += 1
                else:
                    perf.current_streak = 1
                
                if perf.current_streak > perf.best_streak:
                    perf.best_streak = perf.current_streak
            
            else:
                perf.wrong_votes += 1
                
                # Update confidence-specific stats
                if confidence == 'HIGH':
                    perf.high_confidence_votes += 1
                    perf.high_confidence_wrong += 1
                elif confidence == 'MEDIUM':
                    perf.medium_confidence_votes += 1
                elif confidence == 'LOW':
                    perf.low_confidence_votes += 1
                
                # Update streak
                if perf.current_streak <= 0:
                    perf.current_streak -= 1
                else:
                    perf.current_streak = -1
                
                if perf.current_streak < perf.worst_streak:
                    perf.worst_streak = perf.current_streak
            
            # Recalculate accuracy rate
            perf.accuracy_rate = (perf.correct_votes / perf.total_votes * 100) if perf.total_votes > 0 else 50.0
            
            # Recalculate high confidence accuracy
            if perf.high_confidence_votes > 0:
                perf.high_confidence_accuracy = (
                    perf.high_confidence_correct / perf.high_confidence_votes * 100
                )
            
            # ===== ADJUST CONSULTANT WEIGHT (THE LEARNING!) =====
            confidence_multiplier = self.confidence_multipliers.get(confidence, 1.0)
            
            if was_correct:
                # Increase weight (faster for high confidence correct predictions)
                learning_adjustment = 1 + (self.consultant_learning_rate * confidence_multiplier)
                new_weight = min(self.max_consultant_weight, old_weight * learning_adjustment)
                emoji = "📈"
            else:
                # Decrease weight (faster for high confidence wrong predictions)
                learning_adjustment = 1 - (self.consultant_learning_rate * confidence_multiplier)
                new_weight = max(self.min_consultant_weight, old_weight * learning_adjustment)
                emoji = "📉"
            
            perf.current_weight = new_weight
            perf.last_updated = datetime.utcnow()
            
            # Store in performance history (for charts/analysis)
            if not perf.performance_history:
                perf.performance_history = []
            
            perf.performance_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'accuracy': perf.accuracy_rate,
                'weight': new_weight,
                'total_votes': perf.total_votes,
                'streak': perf.current_streak,
                'was_correct': was_correct
            })
            
            # Keep only last 100 records (avoid database bloat)
            if len(perf.performance_history) > 100:
                perf.performance_history = perf.performance_history[-100:]
            
            session.commit()
            
            # Print update
            result_icon = "✅" if was_correct else "❌"
            streak_icon = "🔥" if perf.current_streak > 0 else "❄️"
            
            print(f"{consultant_name}: {result_icon} {confidence} confidence")
            print(f"  Accuracy: {old_accuracy:.1f}% → {perf.accuracy_rate:.1f}%")
            print(f"  Weight: {old_weight:.2f}x → {new_weight:.2f}x {emoji}")
            print(f"  Record: {perf.correct_votes}W-{perf.wrong_votes}L ({perf.total_votes} votes)")
            print(f"  Streak: {perf.current_streak} {streak_icon}")
        
        except Exception as e:
            session.rollback()
            print(f"❌ Error updating {consultant_name} performance: {e}")
        
        finally:
            session.close()
    
    # ========================================================================
    # SIGNAL PERFORMANCE UPDATES
    # ========================================================================
    
    def _update_signal_performance(self, consultant_name, signal_name, was_correct):
        """
        Update individual signal's performance and weight
        
        Args:
            consultant_name: 'C1', 'C2', 'C3', 'C4'
            signal_name: Signal identifier (e.g., 'rsi_oversold')
            was_correct: bool
        """
        if not signal_name:
            return
        
        session = get_session()
        
        try:
            # Get or create signal performance record
            signal_perf = session.query(SignalPerformance).filter(
                SignalPerformance.consultant_name == consultant_name,
                SignalPerformance.signal_name == signal_name
            ).first()
            
            if not signal_perf:
                signal_perf = SignalPerformance(
                    consultant_name=consultant_name,
                    signal_name=signal_name,
                    signal_weight=1.0,
                    accuracy_rate=50.0
                )
                session.add(signal_perf)
            
            old_weight = signal_perf.signal_weight
            
            # Update counts
            signal_perf.total_occurrences += 1
            
            if was_correct:
                signal_perf.correct_predictions += 1
            else:
                signal_perf.wrong_predictions += 1
            
            # Recalculate accuracy
            signal_perf.accuracy_rate = (
                signal_perf.correct_predictions / signal_perf.total_occurrences * 100
            ) if signal_perf.total_occurrences > 0 else 50.0
            
            # ===== ADJUST SIGNAL WEIGHT (THE LEARNING!) =====
            if was_correct:
                # Increase weight
                new_weight = min(
                    self.max_signal_weight,
                    old_weight * (1 + self.signal_learning_rate)
                )
            else:
                # Decrease weight
                new_weight = max(
                    self.min_signal_weight,
                    old_weight * (1 - self.signal_learning_rate)
                )
            
            signal_perf.signal_weight = new_weight
            signal_perf.last_updated = datetime.utcnow()
            
            session.commit()
            
            # Only print if significant weight change (>5%)
            weight_change_pct = abs(new_weight - old_weight) / old_weight * 100
            if weight_change_pct > 5:
                print(f"    Signal '{signal_name}': {old_weight:.2f}x → {new_weight:.2f}x "
                      f"({signal_perf.accuracy_rate:.1f}% accuracy)")
        
        except Exception as e:
            session.rollback()
            print(f"⚠️ Error updating signal '{signal_name}': {e}")
        
        finally:
            session.close()
    
    # ========================================================================
    # PERFORMANCE REPORTING
    # ========================================================================
    
    def _print_learning_summary(self):
        """Print current learning state after update"""
        session = get_session()
        
        try:
            print(f"\n{'='*60}")
            print(f"📊 COMMITTEE LEARNING SUMMARY")
            print(f"{'='*60}")
            
            # Get all consultants ordered by accuracy
            consultants = session.query(ConsultantPerformance).order_by(
                ConsultantPerformance.accuracy_rate.desc()
            ).all()
            
            for i, consultant in enumerate(consultants, 1):
                streak_icon = "🔥" if consultant.current_streak > 0 else "❄️" if consultant.current_streak < 0 else "➖"
                
                print(f"\n#{i} {consultant.consultant_name} ({consultant.specialty}):")
                print(f"  Accuracy: {consultant.accuracy_rate:.1f}% "
                      f"({consultant.correct_votes}W-{consultant.wrong_votes}L)")
                print(f"  Weight: {consultant.current_weight:.2f}x")
                print(f"  Streak: {consultant.current_streak} {streak_icon}")
                
                if consultant.high_confidence_votes > 0:
                    print(f"  High Confidence: {consultant.high_confidence_accuracy:.1f}% "
                          f"({consultant.high_confidence_correct}/{consultant.high_confidence_votes})")
                
                # Show top 3 signals for this consultant
                top_signals = session.query(SignalPerformance).filter(
                    SignalPerformance.consultant_name == consultant.consultant_name,
                    SignalPerformance.total_occurrences >= 3  # At least 3 uses
                ).order_by(
                    SignalPerformance.accuracy_rate.desc()
                ).limit(3).all()
                
                if top_signals:
                    print(f"  Top Signals:")
                    for sig in top_signals:
                        print(f"    • {sig.signal_name}: {sig.accuracy_rate:.1f}% "
                              f"(weight: {sig.signal_weight:.2f}x, n={sig.total_occurrences})")
            
            print(f"\n{'='*60}\n")
        
        except Exception as e:
            print(f"⚠️ Error printing summary: {e}")
        
        finally:
            session.close()
    
    def get_consultant_rankings(self):
        """
        Get consultants ranked by performance
        
        Returns:
            list: Consultant rankings with stats
        """
        session = get_session()
        
        try:
            consultants = session.query(ConsultantPerformance).order_by(
                ConsultantPerformance.accuracy_rate.desc()
            ).all()
            
            rankings = []
            for i, consultant in enumerate(consultants, 1):
                rankings.append({
                    'rank': i,
                    'name': consultant.consultant_name,
                    'specialty': consultant.specialty,
                    'accuracy': consultant.accuracy_rate,
                    'weight': consultant.current_weight,
                    'total_votes': consultant.total_votes,
                    'wins': consultant.correct_votes,
                    'losses': consultant.wrong_votes,
                    'streak': consultant.current_streak
                })
            
            return rankings
        
        finally:
            session.close()
    
    def get_top_signals(self, consultant_name=None, limit=10, min_occurrences=5):
        """
        Get top performing signals
        
        Args:
            consultant_name: Filter by consultant (optional)
            limit: Max results
            min_occurrences: Minimum times signal must appear
        
        Returns:
            list: Top signals with stats
        """
        session = get_session()
        
        try:
            query = session.query(SignalPerformance).filter(
                SignalPerformance.total_occurrences >= min_occurrences
            )
            
            if consultant_name:
                query = query.filter(SignalPerformance.consultant_name == consultant_name)
            
            signals = query.order_by(
                SignalPerformance.accuracy_rate.desc()
            ).limit(limit).all()
            
            results = []
            for sig in signals:
                results.append({
                    'consultant': sig.consultant_name,
                    'signal': sig.signal_name,
                    'accuracy': sig.accuracy_rate,
                    'weight': sig.signal_weight,
                    'occurrences': sig.total_occurrences,
                    'correct': sig.correct_predictions,
                    'wrong': sig.wrong_predictions
                })
            
            return results
        
        finally:
            session.close()
    
    def get_learning_history(self, consultant_name, limit=20):
        """
        Get performance history for a consultant
        
        Args:
            consultant_name: 'C1', 'C2', 'C3', 'C4'
            limit: Number of recent records
        
        Returns:
            list: Historical performance data
        """
        session = get_session()
        
        try:
            consultant = session.query(ConsultantPerformance).filter(
                ConsultantPerformance.consultant_name == consultant_name
            ).first()
            
            if consultant and consultant.performance_history:
                # Return last N records
                return consultant.performance_history[-limit:]
            
            return []
        
        finally:
            session.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COMMITTEE LEARNING SYSTEM")
    print("=" * 60)
    
    # Initialize learning system
    learning = CommitteeLearningSystem()
    
    print("\n✅ Learning system initialized!")
    print(f"  Consultant learning rate: {learning.consultant_learning_rate * 100}%")
    print(f"  Signal learning rate: {learning.signal_learning_rate * 100}%")
    print(f"  Weight range: {learning.min_consultant_weight}x to {learning.max_consultant_weight}x")
    
    # Example: Get current rankings
    print("\n📊 Current Rankings:")
    rankings = learning.get_consultant_rankings()
    for rank in rankings:
        print(f"  #{rank['rank']} {rank['name']}: {rank['accuracy']:.1f}% "
              f"({rank['weight']:.2f}x weight)")
    
    # Example: Get top signals
    print("\n🎯 Top Signals (All Consultants):")
    top_signals = learning.get_top_signals(limit=5, min_occurrences=1)
    for sig in top_signals:
        print(f"  {sig['consultant']}.{sig['signal']}: {sig['accuracy']:.1f}% "
              f"({sig['weight']:.2f}x weight)")
    
    print("\n✅ Learning system ready!")
    print("\nTo use:")
    print("  1. Record committee decision after meeting")
    print("  2. Link decision to trade when user executes")
    print("  3. Call learn_from_trade() when trade closes")
    print("  4. System automatically updates all weights!")
