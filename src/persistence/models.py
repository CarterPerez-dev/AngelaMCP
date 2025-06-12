"""
Database models for AngelaMCP.

This defines all the database tables and relationships.
I'm using SQLAlchemy with async support for production-grade persistence.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Text, DateTime, Float, Integer, Boolean, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func


Base = declarative_base()


class Conversation(Base):
    """
    Represents a conversation session with multiple agents.
    """
    __tablename__ = "conversations"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Basic info
    title: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="active")  # active, completed, archived
    
    # Collaboration details
    collaboration_strategy: Mapped[Optional[str]] = mapped_column(String(50))  # debate, parallel, consensus
    participants: Mapped[List[str]] = mapped_column(JSON, default=list)  # List of agent names
    consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    user_preferences: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
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
    
    # Indexes
    __table_args__ = (
        Index("idx_conversations_status", "status"),
        Index("idx_conversations_created_at", "created_at"),
        Index("idx_conversations_strategy", "collaboration_strategy"),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title}, status={self.status})>"


class Message(Base):
    """
    Individual messages within a conversation.
    """
    __tablename__ = "messages"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Message details
    role: Mapped[str] = mapped_column(String(50))  # user, assistant, system
    agent_type: Mapped[Optional[str]] = mapped_column(String(50))  # claude, openai, gemini
    agent_name: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Content
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(50), default="text")  # text, code, json
    
    # Agent response metadata
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    token_usage: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Context
    task_type: Mapped[Optional[str]] = mapped_column(String(50))
    agent_role: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    error_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index("idx_messages_conversation_id", "conversation_id"),
        Index("idx_messages_agent_type", "agent_type"),
        Index("idx_messages_created_at", "created_at"),
        Index("idx_messages_role", "role"),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, agent={self.agent_type}, role={self.role})>"


class TaskExecution(Base):
    """
    Records of task executions and their results.
    """
    __tablename__ = "task_executions"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    conversation_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    
    # Task details
    task_description: Mapped[str] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(String(50))
    complexity_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Execution details
    strategy_used: Mapped[str] = mapped_column(String(50))  # single_agent, parallel, debate, consensus
    participants: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Results
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    final_solution: Mapped[Optional[str]] = mapped_column(Text)
    consensus_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Performance metrics
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    constraints: Mapped[List[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    error_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    conversation: Mapped[Optional[Conversation]] = relationship("TaskExecution", back_populates="task_executions")
    debate_rounds: Mapped[List["DebateRound"]] = relationship(
        "DebateRound",
        back_populates="task_execution",
        cascade="all, delete-orphan"
    )
    agent_responses: Mapped[List["AgentResponse"]] = relationship(
        "AgentResponse",
        back_populates="task_execution", 
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_task_executions_conversation_id", "conversation_id"),
        Index("idx_task_executions_strategy", "strategy_used"),
        Index("idx_task_executions_success", "success"),
        Index("idx_task_executions_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<TaskExecution(id={self.id}, strategy={self.strategy_used}, success={self.success})>"


class DebateRound(Base):
    """
    Individual rounds in a debate session.
    """
    __tablename__ = "debate_rounds"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    task_execution_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("task_executions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Round details
    round_number: Mapped[int] = mapped_column(Integer)
    topic: Mapped[str] = mapped_column(Text)
    participants: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Results
    round_summary: Mapped[Optional[str]] = mapped_column(Text)
    consensus_reached: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Performance
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)

    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    task_execution: Mapped[TaskExecution] = relationship("TaskExecution", back_populates="debate_rounds")
    agent_responses: Mapped[List["AgentResponse"]] = relationship(
        "AgentResponse",
        back_populates="debate_round",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_debate_rounds_task_execution_id", "task_execution_id"),
        Index("idx_debate_rounds_round_number", "round_number"),
        UniqueConstraint("task_execution_id", "round_number", name="uq_task_round"),
    )
    
    def __repr__(self) -> str:
        return f"<DebateRound(id={self.id}, round={self.round_number}, consensus={self.consensus_reached})>"


class AgentResponse(Base):
    """
    Individual agent responses within tasks and debates.
    """
    __tablename__ = "agent_responses"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    task_execution_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("task_executions.id", ondelete="CASCADE")
    )
    debate_round_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("debate_rounds.id", ondelete="CASCADE")
    )
    
    # Agent details
    agent_type: Mapped[str] = mapped_column(String(50))  # claude, openai, gemini
    agent_name: Mapped[str] = mapped_column(String(100))
    
    # Response details
    response_type: Mapped[str] = mapped_column(String(50))  # proposal, critique, vote, analysis
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    
    # Performance metrics
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    token_usage: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    
    # Context
    task_type: Mapped[Optional[str]] = mapped_column(String(50))
    agent_role: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    error_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task_execution: Mapped[Optional[TaskExecution]] = relationship("TaskExecution", back_populates="agent_responses")
    debate_round: Mapped[Optional[DebateRound]] = relationship("DebateRound", back_populates="agent_responses")
    
    # Indexes
    __table_args__ = (
        Index("idx_agent_responses_task_execution_id", "task_execution_id"),
        Index("idx_agent_responses_debate_round_id", "debate_round_id"),
        Index("idx_agent_responses_agent_type", "agent_type"),
        Index("idx_agent_responses_response_type", "response_type"),
        Index("idx_agent_responses_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AgentResponse(id={self.id}, agent={self.agent_type}, type={self.response_type})>"


class SessionMetrics(Base):
    """
    Performance and usage metrics for monitoring.
    """
    __tablename__ = "session_metrics"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Session details
    session_id: Mapped[str] = mapped_column(String(255))
    metric_type: Mapped[str] = mapped_column(String(50))  # performance, usage, error, cost
    
    # Metric data
    metric_name: Mapped[str] = mapped_column(String(100))
    metric_value: Mapped[float] = mapped_column(Float)
    metric_unit: Mapped[str] = mapped_column(String(50))  # ms, tokens, usd, count
    
    # Context
    agent_type: Mapped[Optional[str]] = mapped_column(String(50))
    task_type: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    
    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_session_metrics_session_id", "session_id"),
        Index("idx_session_metrics_type", "metric_type"),
        Index("idx_session_metrics_name", "metric_name"),
        Index("idx_session_metrics_timestamp", "timestamp"),
        Index("idx_session_metrics_agent_type", "agent_type"),
    )
    
    def __repr__(self) -> str:
        return f"<SessionMetrics(id={self.id}, type={self.metric_type}, name={self.metric_name})>"
