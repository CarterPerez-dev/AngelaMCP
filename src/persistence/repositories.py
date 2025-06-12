"""
Repository pattern implementation for AngelaMCP.
Provides data access layer for all database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from sqlalchemy import desc, func, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import (
    Conversation, Message, TaskExecution, AgentProposal,
    TaskStatus, AgentType, MessageRole
)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def commit(self) -> None:
        """Commit current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback current transaction."""
        await self.session.rollback()
    
    async def refresh(self, instance) -> None:
        """Refresh instance from database."""
        await self.session.refresh(instance)


class ConversationRepository(BaseRepository):
    """Repository for conversation management."""
    
    async def create_conversation(
        self,
        session_id: UUID,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.session.add(conversation)
        await self.commit()
        await self.refresh(conversation)
        
        self.logger.info(f"Created conversation {conversation.id} for session {session_id}")
        return conversation
    
    async def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation by ID."""
        result = await self.session.get(Conversation, conversation_id)
        return result
    
    async def get_active_conversation(self, session_id: UUID) -> Optional[Conversation]:
        """Get active conversation for session."""
        result = await self.session.execute(
            self.session.query(Conversation)
            .filter(
                and_(
                    Conversation.session_id == session_id,
                    Conversation.ended_at.is_(None)
                )
            )
            .order_by(desc(Conversation.started_at))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def end_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """End a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.ended_at = datetime.utcnow()
            conversation.status = "completed"
            await self.commit()
            
        return conversation
    
    async def get_conversation_history(
        self,
        session_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """Get conversation history."""
        query = self.session.query(Conversation)
        
        if session_id:
            query = query.filter(Conversation.session_id == session_id)
            
        query = query.order_by(desc(Conversation.started_at))
        query = query.limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_conversation_metadata(
        self,
        conversation_id: UUID,
        metadata: Dict[str, Any]
    ) -> Optional[Conversation]:
        """Update conversation metadata."""
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.metadata.update(metadata)
            await self.commit()
            
        return conversation


class MessageRepository(BaseRepository):
    """Repository for message management."""
    
    async def create_message(
        self,
        conversation_id: UUID,
        agent_type: AgentType,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Create a new message."""
        message = Message(
            conversation_id=conversation_id,
            agent_type=agent_type,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.session.add(message)
        await self.commit()
        await self.refresh(message)
        
        self.logger.debug(f"Created message {message.id} from {agent_type} in conversation {conversation_id}")
        return message
    
    async def get_message(self, message_id: UUID) -> Optional[Message]:
        """Get message by ID."""
        return await self.session.get(Message, message_id)
    
    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
        agent_type: Optional[AgentType] = None
    ) -> List[Message]:
        """Get messages for a conversation."""
        query = self.session.query(Message).filter(Message.conversation_id == conversation_id)
        
        if agent_type:
            query = query.filter(Message.agent_type == agent_type)
            
        query = query.order_by(Message.created_at)
        query = query.limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_recent_messages(
        self,
        session_id: Optional[UUID] = None,
        limit: int = 20,
        agent_type: Optional[AgentType] = None
    ) -> List[Message]:
        """Get recent messages across conversations."""
        query = self.session.query(Message)
        
        if session_id:
            query = query.join(Conversation).filter(Conversation.session_id == session_id)
            
        if agent_type:
            query = query.filter(Message.agent_type == agent_type)
            
        query = query.order_by(desc(Message.created_at))
        query = query.limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_message_metadata(
        self,
        message_id: UUID,
        metadata: Dict[str, Any]
    ) -> Optional[Message]:
        """Update message metadata."""
        message = await self.get_message(message_id)
        if message:
            message.metadata.update(metadata)
            await self.commit()
            
        return message
    
    async def search_messages(
        self,
        query: str,
        conversation_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[Message]:
        """Search messages by content."""
        search_query = self.session.query(Message).filter(
            Message.content.ilike(f"%{query}%")
        )
        
        if conversation_id:
            search_query = search_query.filter(Message.conversation_id == conversation_id)
            
        search_query = search_query.order_by(desc(Message.created_at))
        search_query = search_query.limit(limit)
        
        result = await self.session.execute(search_query)
        return result.scalars().all()


class TaskExecutionRepository(BaseRepository):
    """Repository for task execution management."""
    
    async def create_task_execution(
        self,
        conversation_id: UUID,
        task_type: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskExecution:
        """Create a new task execution."""
        task_execution = TaskExecution(
            conversation_id=conversation_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            input_data=input_data,
            metadata=metadata or {}
        )
        
        self.session.add(task_execution)
        await self.commit()
        await self.refresh(task_execution)
        
        self.logger.info(f"Created task execution {task_execution.id} of type {task_type}")
        return task_execution
    
    async def get_task_execution(self, task_id: UUID) -> Optional[TaskExecution]:
        """Get task execution by ID."""
        return await self.session.get(TaskExecution, task_id)
    
    async def update_task_status(
        self,
        task_id: UUID,
        status: TaskStatus,
        output_data: Optional[Dict[str, Any]] = None
    ) -> Optional[TaskExecution]:
        """Update task execution status."""
        task = await self.get_task_execution(task_id)
        if task:
            task.status = status
            if output_data:
                task.output_data = output_data
                
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = datetime.utcnow()
                
            await self.commit()
            
        return task
    
    async def get_conversation_tasks(
        self,
        conversation_id: UUID,
        status: Optional[TaskStatus] = None
    ) -> List[TaskExecution]:
        """Get task executions for a conversation."""
        query = self.session.query(TaskExecution).filter(
            TaskExecution.conversation_id == conversation_id
        )
        
        if status:
            query = query.filter(TaskExecution.status == status)
            
        query = query.order_by(desc(TaskExecution.started_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_active_tasks(self) -> List[TaskExecution]:
        """Get all active task executions."""
        result = await self.session.execute(
            self.session.query(TaskExecution).filter(
                TaskExecution.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
            ).order_by(TaskExecution.started_at)
        )
        return result.scalars().all()
    
    async def update_task_metadata(
        self,
        task_id: UUID,
        metadata: Dict[str, Any]
    ) -> Optional[TaskExecution]:
        """Update task execution metadata."""
        task = await self.get_task_execution(task_id)
        if task:
            task.metadata.update(metadata)
            await self.commit()
            
        return task
    
    async def get_task_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get task execution metrics."""
        query = self.session.query(TaskExecution)
        
        if start_date:
            query = query.filter(TaskExecution.started_at >= start_date)
        if end_date:
            query = query.filter(TaskExecution.started_at <= end_date)
            
        # Get status counts
        status_counts = await self.session.execute(
            query.with_entities(
                TaskExecution.status,
                func.count(TaskExecution.id)
            ).group_by(TaskExecution.status)
        )
        
        # Get average execution time
        avg_time = await self.session.execute(
            query.filter(TaskExecution.completed_at.isnot(None))
            .with_entities(
                func.avg(
                    func.extract('epoch', TaskExecution.completed_at - TaskExecution.started_at)
                )
            )
        )
        
        return {
            "status_counts": dict(status_counts.fetchall()),
            "average_execution_time": avg_time.scalar() or 0,
            "total_tasks": await self.session.execute(
                query.with_entities(func.count(TaskExecution.id))
            ).scalar()
        }


class AgentProposalRepository(BaseRepository):
    """Repository for agent proposal management."""
    
    async def create_proposal(
        self,
        task_execution_id: UUID,
        agent_type: AgentType,
        proposal_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentProposal:
        """Create a new agent proposal."""
        proposal = AgentProposal(
            task_execution_id=task_execution_id,
            agent_type=agent_type,
            proposal_content=proposal_content,
            metadata=metadata or {}
        )
        
        self.session.add(proposal)
        await self.commit()
        await self.refresh(proposal)
        
        self.logger.info(f"Created proposal {proposal.id} from {agent_type}")
        return proposal
    
    async def get_proposal(self, proposal_id: UUID) -> Optional[AgentProposal]:
        """Get proposal by ID."""
        return await self.session.get(AgentProposal, proposal_id)
    
    async def get_task_proposals(self, task_execution_id: UUID) -> List[AgentProposal]:
        """Get all proposals for a task."""
        result = await self.session.execute(
            self.session.query(AgentProposal)
            .filter(AgentProposal.task_execution_id == task_execution_id)
            .order_by(AgentProposal.created_at)
        )
        return result.scalars().all()
    
    async def add_critique(
        self,
        proposal_id: UUID,
        critique_data: Dict[str, Any]
    ) -> Optional[AgentProposal]:
        """Add critique data to a proposal."""
        proposal = await self.get_proposal(proposal_id)
        if proposal:
            if not proposal.critique_data:
                proposal.critique_data = {}
            proposal.critique_data.update(critique_data)
            await self.commit()
            
        return proposal
    
    async def increment_vote_count(self, proposal_id: UUID) -> Optional[AgentProposal]:
        """Increment vote count for a proposal."""
        proposal = await self.get_proposal(proposal_id)
        if proposal:
            proposal.vote_count += 1
            await self.commit()
            
        return proposal
    
    async def get_winning_proposal(self, task_execution_id: UUID) -> Optional[AgentProposal]:
        """Get the proposal with highest vote count."""
        result = await self.session.execute(
            self.session.query(AgentProposal)
            .filter(AgentProposal.task_execution_id == task_execution_id)
            .order_by(desc(AgentProposal.vote_count))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def update_proposal_metadata(
        self,
        proposal_id: UUID,
        metadata: Dict[str, Any]
    ) -> Optional[AgentProposal]:
        """Update proposal metadata."""
        proposal = await self.get_proposal(proposal_id)
        if proposal:
            proposal.metadata.update(metadata)
            await self.commit()
            
        return proposal


class RepositoryManager:
    """Central manager for all repositories."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.tasks = TaskExecutionRepository(session)
        self.proposals = AgentProposalRepository(session)
        self.logger = logging.getLogger(__name__)
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Get counts before deletion
        old_conversations = await self.session.execute(
            self.session.query(func.count(Conversation.id))
            .filter(Conversation.started_at < cutoff_date)
        )
        old_messages = await self.session.execute(
            self.session.query(func.count(Message.id))
            .join(Conversation)
            .filter(Conversation.started_at < cutoff_date)
        )
        
        # Delete old data (cascading deletes will handle related records)
        await self.session.execute(
            self.session.query(Conversation)
            .filter(Conversation.started_at < cutoff_date)
            .delete()
        )
        
        await self.session.commit()
        
        cleaned_counts = {
            "conversations": old_conversations.scalar(),
            "messages": old_messages.scalar()
        }
        
        self.logger.info(f"Cleaned up old data: {cleaned_counts}")
        return cleaned_counts
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        # Count active conversations
        active_conversations = await self.session.execute(
            self.session.query(func.count(Conversation.id))
            .filter(Conversation.ended_at.is_(None))
        )
        
        # Count total messages
        total_messages = await self.session.execute(
            self.session.query(func.count(Message.id))
        )
        
        # Count active tasks
        active_tasks = await self.session.execute(
            self.session.query(func.count(TaskExecution.id))
            .filter(TaskExecution.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING]))
        )
        
        # Get agent activity
        agent_activity = await self.session.execute(
            self.session.query(
                Message.agent_type,
                func.count(Message.id)
            )
            .filter(Message.created_at >= datetime.utcnow() - timedelta(days=1))
            .group_by(Message.agent_type)
        )
        
        return {
            "active_conversations": active_conversations.scalar(),
            "total_messages": total_messages.scalar(),
            "active_tasks": active_tasks.scalar(),
            "agent_activity_24h": dict(agent_activity.fetchall())
        }