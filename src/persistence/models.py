"""
Database models for AngelaMCP.

This module defines all SQLAlchemy models for the multi-agent collaboration platform.
I'm implementing comprehensive models with proper relationships, indexes, and validation.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class Conversation(Base):
    """
    Represents a conversation session between the user and multiple AI agents.
    
    I'm tracking the overall conversation state, metadata, and session information
    to enable conversation persistence and resume functionality.
    """
    __tablename__ = "conversations"
    
    # Primary key and identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        unique=True, 
        nullable=False,
        default=uuid.uuid4
    )
    
    # Timing information
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Status and configuration
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active"
    )
    
    # Metadata and settings
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",  # Avoid keyword conflict
        JSON,
        nullable=True
    )
    
    # Cost tracking
    total_cost_usd: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Configuration
    agent_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
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
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'completed', 'failed', 'paused')",
            name="valid_conversation_status"
        ),
        CheckConstraint(
            "total_cost_usd >= 0",
            name="non_negative_cost"
        ),
        CheckConstraint(
            "total_tokens >= 0",
            name="non_negative_tokens"
        ),
        Index("idx_conversation_session_id", "session_id"),
        Index("idx_conversation_status", "status"),
        Index("idx_conversation_started_at", "started_at"),
        Index("idx_conversation_last_activity", "last_activity_at"),
    )


class Message(Base):
    """
    Represents individual messages within a conversation.
    
    I'm storing all agent messages, user inputs, and system messages with
    proper typing, cost tracking, and execution metadata.
    """
    __tablename__ = "messages"
    
    # Primary key and relationships
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Message identification
    agent_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )
    
    # Message content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # Execution metadata
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True
    )
    
    # Cost and performance tracking
    cost_usd: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    execution_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "agent_type IN ('claude_code', 'openai', 'gemini', 'system', 'user')",
            name="valid_agent_type"
        ),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="valid_message_role"
        ),
        CheckConstraint(
            "cost_usd >= 0",
            name="non_negative_message_cost"
        ),
        CheckConstraint(
            "tokens_used >= 0",
            name="non_negative_message_tokens"
        ),
        CheckConstraint(
            "execution_time_ms >= 0",
            name="non_negative_execution_time"
        ),
        CheckConstraint(
            "retry_count >= 0",
            name="non_negative_retry_count"
        ),
        Index("idx_message_conversation_id", "conversation_id"),
        Index("idx_message_agent_type", "agent_type"),
        Index("idx_message_created_at", "created_at"),
        Index("idx_message_conversation_created", "conversation_id", "created_at"),
    )


class TaskExecution(Base):
    """
    Represents a task execution involving multiple agents.
    
    I'm tracking collaborative tasks, debates, voting results, and the complete
    decision-making process across all participating agents.
    """
    __tablename__ = "task_executions"
    
    # Primary key and relationships
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Task identification
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    task_description: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # Execution status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending"
    )
    
    # Input and output data
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Collaboration data
    debate_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    voting_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Execution metadata
    requires_collaboration: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    participating_agents: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True
    )
    primary_agent: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    
    # Results and consensus
    final_result: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    consensus_reached: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    # Cost tracking
    total_cost_usd: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="task_executions"
    )
    agent_proposals: Mapped[List["AgentProposal"]] = relationship(
        "AgentProposal",
        back_populates="task_execution",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="valid_task_status"
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="valid_confidence_score"
        ),
        CheckConstraint(
            "total_cost_usd >= 0",
            name="non_negative_task_cost"
        ),
        CheckConstraint(
            "total_tokens >= 0",
            name="non_negative_task_tokens"
        ),
        CheckConstraint(
            "retry_count >= 0",
            name="non_negative_task_retry_count"
        ),
        Index("idx_task_conversation_id", "conversation_id"),
        Index("idx_task_status", "status"),
        Index("idx_task_type", "task_type"),
        Index("idx_task_started_at", "started_at"),
        Index("idx_task_requires_collaboration", "requires_collaboration"),
    )


class AgentProposal(Base):
    """
    Represents individual agent proposals during collaborative tasks.
    
    I'm storing each agent's contribution to the debate process, including
    initial proposals, critiques, rebuttals, and final solutions.
    """
    __tablename__ = "agent_proposals"
    
    # Primary key and relationships
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    task_execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("task_executions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Agent and proposal information
    agent_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    proposal_phase: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    proposal_content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # Proposal metadata
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    reasoning: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True
    )
    
    # Cost tracking
    cost_usd: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    execution_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Voting and ranking
    vote_weight: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    final_rank: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    selected_as_final: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Relationships
    task_execution: Mapped["TaskExecution"] = relationship(
        "TaskExecution",
        back_populates="agent_proposals"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "agent_type IN ('claude_code', 'openai', 'gemini')",
            name="valid_proposal_agent_type"
        ),
        CheckConstraint(
            "proposal_phase IN ('initial', 'critique', 'rebuttal', 'final')",
            name="valid_proposal_phase"
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="valid_proposal_confidence"
        ),
        CheckConstraint(
            "cost_usd >= 0",
            name="non_negative_proposal_cost"
        ),
        CheckConstraint(
            "tokens_used >= 0",
            name="non_negative_proposal_tokens"
        ),
        CheckConstraint(
            "execution_time_ms >= 0",
            name="non_negative_proposal_execution_time"
        ),
        CheckConstraint(
            "vote_weight >= 0",
            name="non_negative_vote_weight"
        ),
        CheckConstraint(
            "final_rank >= 0",
            name="non_negative_final_rank"
        ),
        # Ensure only one proposal per agent per phase per task
        UniqueConstraint(
            "task_execution_id", "agent_type", "proposal_phase",
            name="unique_agent_phase_proposal"
        ),
        Index("idx_proposal_task_execution_id", "task_execution_id"),
        Index("idx_proposal_agent_type", "agent_type"),
        Index("idx_proposal_phase", "proposal_phase"),
        Index("idx_proposal_created_at", "created_at"),
        Index("idx_proposal_selected", "selected_as_final"),
    )


class SystemMetrics(Base):
    """
    Tracks system-wide metrics and performance data.
    
    I'm implementing comprehensive metrics tracking for monitoring system
    performance, API usage, costs, and operational health.
    """
    __tablename__ = "system_metrics"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Metric identification
    metric_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    
    # Metric values
    value: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    unit: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    
    # Contextual information
    tags: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True
    )
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True
    )
    
    # Timing
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "metric_type IN ('cost', 'performance', 'usage', 'error', 'system')",
            name="valid_metric_type"
        ),
        Index("idx_metrics_type_name", "metric_type", "metric_name"),
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_type_timestamp", "metric_type", "timestamp"),
    )
