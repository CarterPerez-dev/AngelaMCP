"""
Database models for AngelaMCP.

This defines all the database tables and relationships using SQLAlchemy.
I'm implementing a production-grade schema with proper indexes, constraints,
and relationships for multi-agent collaboration tracking.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Text, DateTime, Float, Integer, Boolean, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func


Base = declarative_base()


class Conversation(Base):
    """
    Represents a conversation session with multiple agents.
    
    This is the top-level container for all interactions in a session.
    """
    __tablename__ = "conversations"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Basic information
    title: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(50), 
        default="active",
        index=True
    )  # active, completed, archived, failed
    
    # Collaboration configuration
    collaboration_strategy: Mapped[Optional[str]] = mapped_column(
        String(50),
        index=True
    )  # debate, parallel, consensus, single_agent
    
    participants: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )  # List of agent names participating
    
    # Quality metrics
    consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    
    # Metadata and preferences
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", 
        JSON, 
        default=dict
    )
    user_preferences: Mapped[Dict[str, Any]] = mapped_column(
        JSON, 
        default=dict
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    task_executions: Mapped[List["TaskExecution"]] = relationship(
        "TaskExecution",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    debate_sessions: Mapped[List["DebateSession"]] = relationship(
        "DebateSession",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_conversations_status_created", "status", "created_at"),
        Index("idx_conversations_strategy", "collaboration_strategy"),
        CheckConstraint(
            "status IN ('active', 'completed', 'archived', 'failed')",
            name="check_conversation_status"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id[:8]}, title={self.title}, status={self.status})>"


class Message(Base):
    """
    Individual messages within a conversation.
    
    Stores all interactions between users and agents.
    """
    __tablename__ = "messages"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to conversation
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True
    )
    
    # Message details
    role: Mapped[str] = mapped_column(
        String(50),
        index=True
    )  # user, assistant, system
    
    content: Mapped[str] = mapped_column(Text)
    
    # Agent information
    agent_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        index=True
    )  # claude, openai, gemini
    
    agent_name: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Performance metrics
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON, 
        default=dict
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        index=True
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_messages_conversation_created", "conversation_id", "created_at"),
        Index("idx_messages_agent_type", "agent_type"),
        Index("idx_messages_role", "role"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="check_message_role"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id[:8]}, role={self.role}, agent={self.agent_type})>"


class TaskExecution(Base):
    """
    Records of task executions and their results.
    
    Tracks the performance and outcomes of orchestrated tasks.
    """
    __tablename__ = "task_executions"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to conversation (optional)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        index=True
    )
    
    # Task details
    task_description: Mapped[str] = mapped_column(Text)
    task_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        index=True
    )  # general, code_generation, code_review, etc.
    
    # Execution configuration
    strategy: Mapped[str] = mapped_column(
        String(50),
        index=True
    )  # single_agent, parallel, debate, consensus
    
    # Results
    success: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True
    )
    final_solution: Mapped[Optional[str]] = mapped_column(Text)
    
    # Performance metrics
    execution_time_ms: Mapped[float] = mapped_column(Float)
    consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    
    # Agent participation
    participating_agents: Mapped[List[str]] = mapped_column(JSON, default=list)
    agent_responses: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    cost_breakdown: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_type: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON, 
        default=dict
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="task_executions"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_task_executions_strategy_success", "strategy", "success"),
        Index("idx_task_executions_created", "created_at"),
        Index("idx_task_executions_type", "task_type"),
        CheckConstraint(
            "execution_time_ms >= 0",
            name="check_positive_execution_time"
        ),
        CheckConstraint(
            "consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 1)",
            name="check_consensus_score_range"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<TaskExecution(id={self.id[:8]}, strategy={self.strategy}, success={self.success})>"


class DebateSession(Base):
    """
    Records of structured debate sessions between agents.
    
    Tracks the full debate process including rounds, proposals, and outcomes.
    """
    __tablename__ = "debate_sessions"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to conversation (optional)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        index=True
    )
    
    # Debate details
    topic: Mapped[str] = mapped_column(Text)
    participating_agents: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Results
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    final_consensus: Mapped[Optional[str]] = mapped_column(Text)
    consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Performance metrics
    total_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    
    # Debate structure
    debate_rounds: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    voting_results: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON, 
        default=dict
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="debate_sessions"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_debate_sessions_success", "success"),
        Index("idx_debate_sessions_created", "created_at"),
        CheckConstraint(
            "rounds_completed >= 0",
            name="check_non_negative_rounds"
        ),
        CheckConstraint(
            "consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 1)",
            name="check_debate_consensus_score_range"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<DebateSession(id={self.id[:8]}, topic={self.topic[:50]}, success={self.success})>"


class AgentPerformance(Base):
    """
    Agent performance metrics and statistics.
    
    Tracks individual agent performance over time.
    """
    __tablename__ = "agent_performance"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Agent identification
    agent_name: Mapped[str] = mapped_column(
        String(100),
        index=True
    )
    agent_type: Mapped[str] = mapped_column(
        String(50),
        index=True
    )
    
    # Performance metrics (aggregated over time period)
    time_period: Mapped[str] = mapped_column(
        String(20),
        index=True
    )  # daily, weekly, monthly
    
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True
    )
    
    # Request statistics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    avg_execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    avg_confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    
    # Quality metrics
    avg_consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    debate_wins: Mapped[int] = mapped_column(Integer, default=0)
    debate_participations: Mapped[int] = mapped_column(Integer, default=0)
    
    # Task type breakdown
    task_type_distribution: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Error breakdown
    error_distribution: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_agent_performance_agent_period", "agent_name", "time_period", "period_start"),
        Index("idx_agent_performance_type", "agent_type"),
        UniqueConstraint(
            "agent_name", "time_period", "period_start",
            name="uq_agent_performance_period"
        ),
        CheckConstraint(
            "total_requests >= 0",
            name="check_non_negative_requests"
        ),
        CheckConstraint(
            "successful_requests + failed_requests <= total_requests",
            name="check_request_sum"
        ),
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def win_rate(self) -> float:
        """Calculate debate win rate."""
        if self.debate_participations == 0:
            return 0.0
        return self.debate_wins / self.debate_participations
    
    def __repr__(self) -> str:
        return f"<AgentPerformance(agent={self.agent_name}, period={self.time_period}, success_rate={self.success_rate:.2f})>"


class SystemMetrics(Base):
    """
    System-wide metrics and health statistics.
    
    Tracks overall platform performance and usage.
    """
    __tablename__ = "system_metrics"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Time period
    metric_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True
    )
    metric_type: Mapped[str] = mapped_column(
        String(50),
        index=True
    )  # daily, hourly, realtime
    
    # Usage statistics
    total_conversations: Mapped[int] = mapped_column(Integer, default=0)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    total_task_executions: Mapped[int] = mapped_column(Integer, default=0)
    total_debate_sessions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    avg_response_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    success_rate: Mapped[Optional[float]] = mapped_column(Float)
    system_uptime_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Resource usage
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    database_size_mb: Mapped[Optional[float]] = mapped_column(Float)
    
    # Strategy distribution
    strategy_distribution: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Agent distribution
    agent_usage_distribution: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    error_types: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Additional metrics
    custom_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index("idx_system_metrics_date_type", "metric_date", "metric_type"),
        UniqueConstraint(
            "metric_date", "metric_type",
            name="uq_system_metrics_date_type"
        ),
        CheckConstraint(
            "success_rate IS NULL OR (success_rate >= 0 AND success_rate <= 1)",
            name="check_system_success_rate"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<SystemMetrics(date={self.metric_date.date()}, type={self.metric_type})>"


# View models for complex queries (using SQL views)
class ConversationSummary(Base):
    """
    Materialized view for conversation summaries.
    
    Provides aggregated conversation statistics for reporting.
    """
    __tablename__ = "conversation_summaries"
    
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50))
    strategy: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Aggregated metrics
    message_count: Mapped[int] = mapped_column(Integer)
    agent_count: Mapped[int] = mapped_column(Integer)
    task_execution_count: Mapped[int] = mapped_column(Integer)
    debate_session_count: Mapped[int] = mapped_column(Integer)
    
    # Performance aggregates
    avg_execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    avg_consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_minutes: Mapped[Optional[float]] = mapped_column(Float)
    
    def __repr__(self) -> str:
        return f"<ConversationSummary(id={self.conversation_id[:8]}, messages={self.message_count})>"
