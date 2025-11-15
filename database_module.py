"""
database_module.py - SQLAlchemy ORM models for Committee Learning System

This module provides SQLAlchemy models that map to the database tables
created by database.py. It works ALONGSIDE database.py (doesn't replace it).

Usage:
    from database_module import get_session, ConsultantPerformance, Trade
    
    session = get_session()
    consultants = session.query(ConsultantPerformance).all()
    session.close()
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path

# Get database path (same as database.py)
HOME = Path.home()
DB_PATH = HOME / 'trading_ai_learning.db'

# Create SQLAlchemy engine
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

# Create base class for models
Base = declarative_base()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ============================================================================
# SQLALCHEMY ORM MODELS
# ============================================================================

class ConsultantPerformance(Base):
    """
    Tracks performance of each consultant (C1, C2, C3, C4)
    Maps to: consultant_performance table
    """
    __tablename__ = 'consultant_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    consultant_name = Column(String, unique=True, nullable=False)
    specialty = Column(String)
    total_votes = Column(Integer, default=0)
    correct_votes = Column(Integer, default=0)
    wrong_votes = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=50.0)
    high_confidence_votes = Column(Integer, default=0)
    high_confidence_correct = Column(Integer, default=0)
    high_confidence_wrong = Column(Integer, default=0)
    high_confidence_accuracy = Column(Float, default=50.0)
    medium_confidence_votes = Column(Integer, default=0)
    medium_confidence_correct = Column(Integer, default=0)
    low_confidence_votes = Column(Integer, default=0)
    low_confidence_correct = Column(Integer, default=0)
    current_weight = Column(Float, default=1.0)
    current_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)
    worst_streak = Column(Integer, default=0)
    performance_history = Column(Text)  # JSON string
    created_at = Column(String, default=lambda: datetime.now().isoformat())
    last_updated = Column(String, default=lambda: datetime.now().isoformat())
    
    def __repr__(self):
        return f"<Consultant {self.consultant_name}: {self.accuracy_rate:.1f}% ({self.current_weight:.2f}x)>"


class SignalPerformance(Base):
    """
    Tracks performance of individual signals used by consultants
    Maps to: signal_performance table
    """
    __tablename__ = 'signal_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    consultant_name = Column(String, nullable=False)
    signal_name = Column(String, nullable=False)
    total_occurrences = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    wrong_predictions = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=50.0)
    signal_weight = Column(Float, default=1.0)
    last_updated = Column(String, default=lambda: datetime.now().isoformat())
    
    def __repr__(self):
        return f"<Signal {self.consultant_name}.{self.signal_name}: {self.accuracy_rate:.1f}%>"


class CommitteeDecision(Base):
    """
    Records every committee decision for learning purposes
    Maps to: committee_decisions table
    """
    __tablename__ = 'committee_decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    market_type = Column(String)
    price_at_decision = Column(Float, nullable=False)
    timestamp = Column(String, nullable=False)
    
    # C1 vote details
    c1_vote = Column(String)
    c1_confidence = Column(String)
    c1_score = Column(Float)
    c1_weight = Column(Float)
    c1_reasoning = Column(Text)
    
    # C2 vote details
    c2_vote = Column(String)
    c2_confidence = Column(String)
    c2_score = Column(Float)
    c2_weight = Column(Float)
    c2_reasoning = Column(Text)
    
    # C3 vote details
    c3_vote = Column(String)
    c3_confidence = Column(String)
    c3_score = Column(Float)
    c3_weight = Column(Float)
    c3_reasoning = Column(Text)
    
    # C4 vote details
    c4_vote = Column(String)
    c4_confidence = Column(String)
    c4_score = Column(Float)
    c4_weight = Column(Float)
    c4_reasoning = Column(Text)
    
    # Committee final decision
    final_decision = Column(String, nullable=False)
    consensus_level = Column(Float)
    total_weighted_votes = Column(Text)  # JSON string
    
    # Recommendations
    recommended_entry = Column(Float)
    recommended_stop_loss = Column(Float)
    recommended_take_profit = Column(Float)
    
    # Indicators snapshot
    indicators_snapshot = Column(Text)  # JSON string
    
    # Link to trade
    trade_id = Column(Integer, ForeignKey('trade_results.id'))
    
    # Outcome (filled in after trade closes)
    actual_outcome = Column(String)  # 'WIN' or 'LOSS'
    outcome_determined_at = Column(String)
    price_at_outcome = Column(Float)
    profit_loss_pct = Column(Float)
    hours_to_outcome = Column(Float)
    was_correct = Column(Integer)  # 1 = correct, 0 = wrong
    
    def __repr__(self):
        return f"<Decision {self.id}: {self.symbol} {self.final_decision} @ {self.price_at_decision:.2f}>"


class Trade(Base):
    """
    Maps to existing trade_results table
    This allows committee_learning to link decisions to trades
    """
    __tablename__ = 'trade_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    trade_date = Column(String, nullable=False)
    profit_loss = Column(Float, nullable=False)
    profit_loss_pct = Column(Float, nullable=False)
    prediction_error = Column(Float, nullable=False)
    notes = Column(Text)
    predicted_entry_price = Column(Float)
    predicted_exit_price = Column(Float)
    entry_slippage = Column(Float)
    exit_slippage = Column(Float)
    
    # NOTE: Don't add columns that don't exist in actual database!
    # outcome column removed - it doesn't exist in your trade_results table
    
    def __repr__(self):
        return f"<Trade {self.id}: ${self.profit_loss:.2f} ({self.profit_loss_pct:.2f}%)>"


class CommitteeMeetingLog(Base):
    """
    Logs committee meetings for analysis
    Maps to: committee_meeting_logs table
    """
    __tablename__ = 'committee_meeting_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, ForeignKey('committee_decisions.id'), nullable=False)
    timestamp = Column(String, nullable=False)
    meeting_summary = Column(Text)
    conflicts_detected = Column(Text)
    consensus_reached = Column(Integer)  # 1 = yes, 0 = no
    
    def __repr__(self):
        return f"<MeetingLog {self.id}: Decision {self.decision_id}>"


class SignalCorrelation(Base):
    """
    Tracks which signals work well together
    Maps to: signal_correlations table
    """
    __tablename__ = 'signal_correlations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_1 = Column(String, nullable=False)
    signal_2 = Column(String, nullable=False)
    times_both_correct = Column(Integer, default=0)
    times_both_wrong = Column(Integer, default=0)
    times_conflicted = Column(Integer, default=0)
    correlation_strength = Column(Float, default=0)
    last_updated = Column(String, default=lambda: datetime.now().isoformat())
    
    def __repr__(self):
        return f"<Correlation {self.signal_1} + {self.signal_2}: {self.correlation_strength:.2f}>"


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session():
    """
    Get a new SQLAlchemy session
    
    Usage:
        session = get_session()
        # Do database operations
        session.close()
    
    Returns:
        SQLAlchemy Session object
    """
    return SessionLocal()


def init_models():
    """
    Initialize models (create tables if they don't exist)
    This is called automatically when module is imported
    
    Note: Tables are already created by database.py's init_database()
    This just ensures the ORM mappings are ready
    """
    # Don't create tables - they already exist from database.py
    # Just ensure metadata is loaded
    Base.metadata.create_all(bind=engine, checkfirst=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_consultant_performance(consultant_name):
    """
    Get performance for a specific consultant
    
    Args:
        consultant_name: 'C1', 'C2', 'C3', or 'C4'
    
    Returns:
        ConsultantPerformance object or None
    """
    session = get_session()
    try:
        consultant = session.query(ConsultantPerformance).filter(
            ConsultantPerformance.consultant_name == consultant_name
        ).first()
        return consultant
    finally:
        session.close()


def get_all_consultants():
    """
    Get all consultant performance records
    
    Returns:
        List of ConsultantPerformance objects
    """
    session = get_session()
    try:
        consultants = session.query(ConsultantPerformance).all()
        return consultants
    finally:
        session.close()


def get_recent_decisions(limit=10):
    """
    Get recent committee decisions
    
    Args:
        limit: Number of decisions to retrieve
    
    Returns:
        List of CommitteeDecision objects
    """
    session = get_session()
    try:
        decisions = session.query(CommitteeDecision).order_by(
            CommitteeDecision.id.desc()
        ).limit(limit).all()
        return decisions
    finally:
        session.close()


def get_decision_by_trade(trade_id):
    """
    Get committee decision for a specific trade
    
    Args:
        trade_id: Trade ID
    
    Returns:
        CommitteeDecision object or None
    """
    session = get_session()
    try:
        decision = session.query(CommitteeDecision).filter(
            CommitteeDecision.trade_id == trade_id
        ).first()
        return decision
    finally:
        session.close()


# ============================================================================
# INITIALIZATION
# ============================================================================

# DON'T auto-initialize on import - causes conflicts with existing database
# Models will be created when explicitly needed
# 
# To manually initialize: call init_models() after importing
#
# try:
#     init_models()
#     print("✅ database_module.py loaded - SQLAlchemy models ready")
# except Exception as e:
#     print(f"⚠️ database_module.py: Could not initialize models - {e}")

print("✅ database_module.py loaded - use init_models() to create tables if needed")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING database_module.py")
    print("="*70)
    
    # Test 1: Get session
    print("\n1. Testing get_session()...")
    session = get_session()
    print(f"   ✅ Session created: {session}")
    session.close()
    
    # Test 2: Query consultants
    print("\n2. Testing consultant queries...")
    consultants = get_all_consultants()
    if consultants:
        print(f"   ✅ Found {len(consultants)} consultants:")
        for c in consultants:
            print(f"      {c}")
    else:
        print("   ⚠️ No consultants found (database not initialized yet)")
    
    # Test 3: Get specific consultant
    print("\n3. Testing get_consultant_performance('C1')...")
    c1 = get_consultant_performance('C1')
    if c1:
        print(f"   ✅ C1 found: {c1}")
        print(f"      Accuracy: {c1.accuracy_rate:.1f}%")
        print(f"      Weight: {c1.current_weight:.2f}x")
        print(f"      Total votes: {c1.total_votes}")
    else:
        print("   ⚠️ C1 not found (database not initialized yet)")
    
    # Test 4: Get recent decisions
    print("\n4. Testing get_recent_decisions()...")
    decisions = get_recent_decisions(limit=5)
    if decisions:
        print(f"   ✅ Found {len(decisions)} recent decisions:")
        for d in decisions:
            print(f"      {d}")
    else:
        print("   ℹ️ No decisions recorded yet")
    
    # Test 5: Get trades
    print("\n5. Testing Trade model...")
    session = get_session()
    trades = session.query(Trade).limit(3).all()
    if trades:
        print(f"   ✅ Found {len(trades)} trades:")
        for t in trades:
            print(f"      {t}")
    else:
        print("   ℹ️ No trades found")
    session.close()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)
