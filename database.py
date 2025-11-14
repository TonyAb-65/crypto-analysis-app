# database_module.py - COMPLETE VERSION WITH LEARNING SYSTEM

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# ============================================================================
# ORIGINAL TABLES (Your existing system)
# ============================================================================

class Trade(Base):
    """Historical completed trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # LONG/SHORT
    
    # Prices
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    
    # Timing
    entry_time = Column(DateTime, default=datetime.utcnow, index=True)
    exit_time = Column(DateTime, index=True)
    
    # Position details
    quantity = Column(Float)
    
    # Results
    profit_loss = Column(Float)
    profit_loss_percentage = Column(Float)
    outcome = Column(String(10), index=True)  # 'win', 'loss'
    exit_type = Column(String(50))  # 'stop_loss', 'take_profit', 'manual'
    
    # Technical data
    indicators_at_entry = Column(JSON)
    indicators_at_exit = Column(JSON)
    
    # ML confidence
    model_confidence = Column(Float)
    
    # Notes
    notes = Column(Text)
    
    # NEW: Prediction tracking (from your earlier system)
    predicted_entry_price = Column(Float, nullable=True)
    predicted_exit_price = Column(Float, nullable=True)
    actual_entry_price = Column(Float, nullable=True)
    actual_exit_price = Column(Float, nullable=True)
    entry_slippage_pct = Column(Float, nullable=True)
    exit_slippage_pct = Column(Float, nullable=True)
    broker_execution_quality = Column(String(20), nullable=True)


class ActivePosition(Base):
    """Currently open positions being monitored"""
    __tablename__ = 'active_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)
    
    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    
    # Current status
    current_price = Column(Float)
    quantity = Column(Float)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Monitoring
    timeframe = Column(String(10), default='1H')
    last_check_time = Column(DateTime, index=True)
    current_recommendation = Column(String(10))
    
    # Technical snapshot
    indicators_snapshot = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # OBV monitoring
    last_obv_slope = Column(Float)
    
    # Alerts
    monitoring_alerts = Column(JSON)


class MarketData(Base):
    """Historical market data snapshots"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # OHLCV
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    
    # Calculated indicators
    indicators = Column(JSON)


# ============================================================================
# NEW: COMMITTEE LEARNING TABLES
# ============================================================================

class ConsultantPerformance(Base):
    """Track each consultant's prediction accuracy over time"""
    __tablename__ = 'consultant_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    consultant_name = Column(String(50), nullable=False, unique=True, index=True)  # 'C1', 'C2', 'C3', 'C4'
    specialty = Column(String(100))  # 'Technical Analysis', 'Sentiment', 'Risk', 'Trend'
    
    # Performance metrics
    total_votes = Column(Integer, default=0)
    correct_votes = Column(Integer, default=0)
    wrong_votes = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=50.0)  # Percentage
    
    # Confidence tracking
    high_confidence_votes = Column(Integer, default=0)
    high_confidence_correct = Column(Integer, default=0)
    high_confidence_wrong = Column(Integer, default=0)
    high_confidence_accuracy = Column(Float, default=50.0)
    
    medium_confidence_votes = Column(Integer, default=0)
    medium_confidence_correct = Column(Integer, default=0)
    
    low_confidence_votes = Column(Integer, default=0)
    low_confidence_correct = Column(Integer, default=0)
    
    # Dynamic weight (starts at 1.0, range: 0.5x to 3.0x)
    current_weight = Column(Float, default=1.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Historical performance tracking (JSON array of snapshots)
    performance_history = Column(JSON, default=list)
    
    # Win/Loss streaks
    current_streak = Column(Integer, default=0)  # Positive = winning streak, Negative = losing streak
    best_streak = Column(Integer, default=0)
    worst_streak = Column(Integer, default=0)


class SignalPerformance(Base):
    """Track performance of individual signals used by each consultant"""
    __tablename__ = 'signal_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    consultant_name = Column(String(50), nullable=False, index=True)
    signal_name = Column(String(100), nullable=False, index=True)  # 'rsi_oversold', 'macd_bullish', etc.
    signal_description = Column(Text)  # Human-readable description
    
    # Performance metrics
    total_occurrences = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    wrong_predictions = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=50.0)
    
    # Signal weight (how much this signal contributes to score)
    signal_weight = Column(Float, default=1.0)  # Starts at 1.0, range: 0.3x to 2.0x
    base_score = Column(Float, default=1.0)  # The original score value
    
    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Context tracking
    market_conditions = Column(JSON)  # Track when this signal works best (volatility, trend, etc.)
    
    # Unique constraint: One record per consultant+signal combination
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )


class CommitteeDecision(Base):
    """Store each committee meeting decision for future learning"""
    __tablename__ = 'committee_decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Link to actual trade (if user executes the recommendation)
    trade_id = Column(Integer, nullable=True, index=True)
    
    # Decision metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False)
    market_type = Column(String(20))
    
    # Market context at decision time
    price_at_decision = Column(Float, nullable=False)
    indicators_snapshot = Column(JSON)
    
    # C1 (Technical Analyst) vote
    c1_vote = Column(String(10))  # 'BUY', 'SELL', 'HOLD'
    c1_confidence = Column(String(10))  # 'HIGH', 'MEDIUM', 'LOW'
    c1_score = Column(Float)
    c1_weight = Column(Float)  # Weight at time of decision
    c1_reasoning = Column(JSON)  # Full reasoning with signals
    
    # C2 (Sentiment Analyst) vote
    c2_vote = Column(String(10))
    c2_confidence = Column(String(10))
    c2_score = Column(Float)
    c2_weight = Column(Float)
    c2_reasoning = Column(JSON)
    
    # C3 (Risk Manager) vote
    c3_vote = Column(String(10))
    c3_confidence = Column(String(10))
    c3_score = Column(Float)
    c3_weight = Column(Float)
    c3_reasoning = Column(JSON)
    
    # C4 (Trend Analyst) vote
    c4_vote = Column(String(10))
    c4_confidence = Column(String(10))
    c4_score = Column(Float)
    c4_weight = Column(Float)
    c4_reasoning = Column(JSON)
    
    # Committee final decision
    final_decision = Column(String(10), index=True)  # 'BUY', 'SELL', 'HOLD'
    consensus_level = Column(Float)  # 0.0 to 100.0 (percentage of agreement)
    total_weighted_votes = Column(JSON)  # {'BUY': 5.2, 'SELL': 1.8, 'HOLD': 0.5}
    
    # Recommended action details (if BUY or SELL)
    recommended_entry = Column(Float)
    recommended_stop_loss = Column(Float)
    recommended_take_profit = Column(Float)
    
    # Outcome tracking (filled in later when position closes)
    actual_outcome = Column(String(10), nullable=True, index=True)  # 'WIN', 'LOSS', None
    outcome_determined_at = Column(DateTime, nullable=True)
    price_at_outcome = Column(Float, nullable=True)
    profit_loss_pct = Column(Float, nullable=True)
    
    # Was the committee decision correct?
    was_correct = Column(Boolean, nullable=True)
    
    # How long until outcome was known
    hours_to_outcome = Column(Float, nullable=True)


class CommitteeMeetingLog(Base):
    """Detailed log of committee discussions (for debugging/analysis)"""
    __tablename__ = 'committee_meeting_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, index=True)  # Links to CommitteeDecision
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Full meeting transcript
    meeting_summary = Column(Text)
    
    # Disagreements and conflicts
    had_conflict = Column(Boolean, default=False)
    conflict_description = Column(Text)
    
    # Which consultants agreed/disagreed
    agreement_matrix = Column(JSON)  # e.g., {"C1-C2": "agree", "C1-C3": "disagree"}
    
    # Market conditions during meeting
    volatility = Column(Float)
    trend_strength = Column(Float)
    volume_level = Column(String(20))  # 'low', 'normal', 'high'


# ============================================================================
# NEW: SIGNAL CORRELATION TRACKING
# ============================================================================

class SignalCorrelation(Base):
    """Track which signals work well together"""
    __tablename__ = 'signal_correlations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Signal pair
    signal_1_consultant = Column(String(50), index=True)
    signal_1_name = Column(String(100), index=True)
    signal_2_consultant = Column(String(50), index=True)
    signal_2_name = Column(String(100), index=True)
    
    # Performance when both signals are present
    both_present_count = Column(Integer, default=0)
    both_present_wins = Column(Integer, default=0)
    both_present_accuracy = Column(Float, default=50.0)
    
    # Synergy score (how much better they perform together vs. individually)
    synergy_score = Column(Float, default=0.0)  # Positive = good combo, Negative = bad combo
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# EXISTING TABLES FROM YOUR SYSTEM (if you have these)
# ============================================================================

class ModelPerformance(Base):
    """ML model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    
    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training data
    training_samples = Column(Integer)
    
    # Model version
    model_version = Column(String(50))


class IndicatorPerformance(Base):
    """Track individual indicator accuracy"""
    __tablename__ = 'indicator_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_name = Column(String(50), nullable=False, unique=True)
    
    # Performance
    total_signals = Column(Integer, default=0)
    correct_signals = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=50.0)
    
    # Weight
    current_weight = Column(Float, default=1.0)
    
    last_updated = Column(DateTime, default=datetime.utcnow)


class DivergenceEvent(Base):
    """Track divergence detection and resolution"""
    __tablename__ = 'divergence_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # Divergence details
    indicator = Column(String(50), nullable=False)  # 'RSI', 'MACD', 'OBV'
    divergence_type = Column(String(20), nullable=False)  # 'bullish', 'bearish'
    
    # Detection
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    detection_price = Column(Float)
    
    # Resolution
    resolved_at = Column(DateTime, nullable=True)
    resolution_price = Column(Float, nullable=True)
    resolution_candles = Column(Integer, nullable=True)
    resolution_outcome = Column(String(20), nullable=True)  # 'successful', 'failed'
    
    # Status
    status = Column(String(20), default='active', index=True)  # 'active', 'resolved'


class DivergenceStats(Base):
    """Statistical analysis of divergence timing"""
    __tablename__ = 'divergence_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator = Column(String(50), nullable=False)
    divergence_type = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Statistics
    total_occurrences = Column(Integer, default=0)
    successful_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Timing stats
    avg_candles_to_resolution = Column(Float)
    min_candles_to_resolution = Column(Integer)
    max_candles_to_resolution = Column(Integer)
    
    # Speed classification
    speed_classification = Column(String(20))  # 'FAST', 'ACTIONABLE', 'SLOW'
    
    last_updated = Column(DateTime, default=datetime.utcnow)


class WhaleActivity(Base):
    """Track whale movements and smart money"""
    __tablename__ = 'whale_activity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Activity type
    activity_type = Column(String(50))  # 'accumulation', 'distribution'
    
    # Details
    volume_spike = Column(Float)
    price_at_detection = Column(Float)
    obv_slope = Column(Float)
    
    # Outcome
    was_correct = Column(Boolean, nullable=True)


# ============================================================================
# DATABASE CONNECTION AND SESSION MANAGEMENT
# ============================================================================

# Global engine and session factory
_engine = None
_SessionFactory = None


def get_engine():
    """Get or create database engine"""
    global _engine
    
    if _engine is None:
        # Try to get DATABASE_URL from environment
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            # Default to SQLite for development
            database_url = 'sqlite:///trading_platform.db'
            print("⚠️ DATABASE_URL not set, using SQLite: trading_platform.db")
        
        # Create engine
        if database_url.startswith('sqlite'):
            _engine = create_engine(
                database_url,
                connect_args={'check_same_thread': False},
                echo=False  # Set to True for SQL debugging
            )
        else:
            # PostgreSQL or other databases
            _engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                echo=False
            )
        
        print(f"✅ Database engine created: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    
    return _engine


def get_session_factory():
    """Get or create session factory"""
    global _SessionFactory
    
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine)
    
    return _SessionFactory


def get_session():
    """Get a new database session"""
    SessionFactory = get_session_factory()
    return SessionFactory()


def init_db():
    """Initialize database - create all tables"""
    try:
        engine = get_engine()
        
        print("🔧 Creating database tables...")
        Base.metadata.create_all(engine)
        
        print("✅ Database tables created successfully!")
        
        # Initialize consultant performance records if they don't exist
        _initialize_consultants()
        
        # Run migrations for existing databases
        _run_migrations()
        
        return engine
    
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise


def _initialize_consultants():
    """Initialize consultant performance records"""
    session = get_session()
    
    try:
        consultants = [
            {'name': 'C1', 'specialty': 'Technical Analysis'},
            {'name': 'C2', 'specialty': 'Market Sentiment'},
            {'name': 'C3', 'specialty': 'Risk Management'},
            {'name': 'C4', 'specialty': 'Trend Analysis'}
        ]
        
        for consultant in consultants:
            existing = session.query(ConsultantPerformance).filter(
                ConsultantPerformance.consultant_name == consultant['name']
            ).first()
            
            if not existing:
                new_consultant = ConsultantPerformance(
                    consultant_name=consultant['name'],
                    specialty=consultant['specialty'],
                    current_weight=1.0,
                    accuracy_rate=50.0
                )
                session.add(new_consultant)
                print(f"  ✅ Initialized {consultant['name']}: {consultant['specialty']}")
        
        session.commit()
    
    except Exception as e:
        session.rollback()
        print(f"⚠️ Error initializing consultants: {e}")
    
    finally:
        session.close()


def _run_migrations():
    """Run database migrations for existing databases"""
    from sqlalchemy import inspect, text
    
    engine = get_engine()
    inspector = inspect(engine)
    
    print("🔄 Checking for database migrations...")
    
    # Migration 1: Add timeframe to active_positions
    if 'active_positions' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('active_positions')]
        
        if 'timeframe' not in columns:
            try:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE active_positions ADD COLUMN timeframe VARCHAR(10) DEFAULT '1H'"))
                print("  ✅ Added 'timeframe' column to active_positions")
            except Exception as e:
                print(f"  ⚠️ Migration 1 failed: {e}")
    
    # Migration 2: Add prediction tracking columns to trades
    if 'trades' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('trades')]
        
        new_columns = [
            ('predicted_entry_price', 'FLOAT'),
            ('predicted_exit_price', 'FLOAT'),
            ('actual_entry_price', 'FLOAT'),
            ('actual_exit_price', 'FLOAT'),
            ('entry_slippage_pct', 'FLOAT'),
            ('exit_slippage_pct', 'FLOAT'),
            ('broker_execution_quality', 'VARCHAR(20)')
        ]
        
        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"))
                    print(f"  ✅ Added '{col_name}' column to trades")
                except Exception as e:
                    print(f"  ⚠️ Could not add '{col_name}': {e}")
    
    # Migration 3: Add indexes for performance
    try:
        with engine.begin() as conn:
            # Check existing indexes
            existing_indexes = inspector.get_indexes('trades')
            index_names = [idx['name'] for idx in existing_indexes]
            
            if 'idx_symbol_outcome' not in index_names:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_symbol_outcome ON trades(symbol, outcome)"))
                print("  ✅ Added index: idx_symbol_outcome")
            
            if 'idx_entry_exit_time' not in index_names:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entry_exit_time ON trades(entry_time, exit_time)"))
                print("  ✅ Added index: idx_entry_exit_time")
    
    except Exception as e:
        print(f"  ⚠️ Index creation failed: {e}")
    
    print("✅ Migrations complete!")


def drop_all_tables():
    """⚠️ WARNING: Drop all tables (use only for testing!)"""
    engine = get_engine()
    print("⚠️ DROPPING ALL TABLES...")
    Base.metadata.drop_all(engine)
    print("✅ All tables dropped")


def reset_database():
    """⚠️ WARNING: Reset entire database (drop and recreate)"""
    drop_all_tables()
    init_db()
    print("✅ Database reset complete")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_consultant_performance(consultant_name):
    """Get performance metrics for a specific consultant"""
    session = get_session()
    
    try:
        perf = session.query(ConsultantPerformance).filter(
            ConsultantPerformance.consultant_name == consultant_name
        ).first()
        
        if perf:
            return {
                'name': perf.consultant_name,
                'specialty': perf.specialty,
                'accuracy': perf.accuracy_rate,
                'weight': perf.current_weight,
                'total_votes': perf.total_votes,
                'correct': perf.correct_votes,
                'wrong': perf.wrong_votes,
                'current_streak': perf.current_streak
            }
        
        return None
    
    finally:
        session.close()


def get_all_consultants_performance():
    """Get performance metrics for all consultants"""
    session = get_session()
    
    try:
        consultants = session.query(ConsultantPerformance).order_by(
            ConsultantPerformance.accuracy_rate.desc()
        ).all()
        
        results = []
        for perf in consultants:
            results.append({
                'name': perf.consultant_name,
                'specialty': perf.specialty,
                'accuracy': perf.accuracy_rate,
                'weight': perf.current_weight,
                'total_votes': perf.total_votes,
                'correct': perf.correct_votes,
                'wrong': perf.wrong_votes
            })
        
        return results
    
    finally:
        session.close()


def get_top_signals(consultant_name=None, limit=10):
    """Get top performing signals"""
    session = get_session()
    
    try:
        query = session.query(SignalPerformance)
        
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
                'occurrences': sig.total_occurrences
            })
        
        return results
    
    finally:
        session.close()


def get_committee_decision_history(limit=20):
    """Get recent committee decisions"""
    session = get_session()
    
    try:
        decisions = session.query(CommitteeDecision).order_by(
            CommitteeDecision.timestamp.desc()
        ).limit(limit).all()
        
        results = []
        for dec in decisions:
            results.append({
                'id': dec.id,
                'symbol': dec.symbol,
                'decision': dec.final_decision,
                'consensus': dec.consensus_level,
                'timestamp': dec.timestamp,
                'outcome': dec.actual_outcome,
                'was_correct': dec.was_correct
            })
        
        return results
    
    finally:
        session.close()


# ============================================================================
# INITIALIZATION ON IMPORT
# ============================================================================

if __name__ == "__main__":
    # If run directly, initialize database
    print("=" * 60)
    print("TRADING PLATFORM - DATABASE INITIALIZATION")
    print("=" * 60)
    init_db()
    print("\n✅ Database ready!")
    
    # Show consultant status
    print("\n📊 Consultant Status:")
    consultants = get_all_consultants_performance()
    for c in consultants:
        print(f"  {c['name']} ({c['specialty']}): {c['accuracy']:.1f}% accuracy, {c['weight']:.2f}x weight")
