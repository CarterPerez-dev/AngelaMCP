"""
Session and Task Manager for AngelaMCP.
Manages session lifecycle, task queues, resource allocation, and performance monitoring.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import weakref

from src.models.database import Conversation, TaskExecution, TaskStatus, AgentType
from src.persistence.repositories import RepositoryManager
from src.persistence.database import DatabaseManager
from src.utils.metrics import MetricsCollector
from config.settings import settings


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    IDLE = "idle"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    conversation_id: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    task_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskQueueItem:
    """Item in the task queue."""
    task_id: str
    conversation_id: str
    task_type: str
    priority: int
    input_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    retries: int = 0
    max_retries: int = 3


class ResourcePool:
    """Manages resource allocation for agents and tasks."""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Set[str] = set()
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.lock = asyncio.Lock()
        
    async def acquire_task_slot(self, task_id: str, agent_type: str) -> bool:
        """Try to acquire a task slot for execution."""
        async with self.lock:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                return False
                
            self.active_tasks.add(task_id)
            self.agent_load[agent_type] += 1
            return True
    
    async def release_task_slot(self, task_id: str, agent_type: str):
        """Release a task slot after completion."""
        async with self.lock:
            self.active_tasks.discard(task_id)
            self.agent_load[agent_type] = max(0, self.agent_load[agent_type] - 1)
    
    def get_load_info(self) -> Dict[str, Any]:
        """Get current resource load information."""
        return {
            "active_tasks": len(self.active_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "utilization": len(self.active_tasks) / self.max_concurrent_tasks * 100,
            "agent_load": dict(self.agent_load)
        }


class SessionManager:
    """Manages active sessions and their lifecycle."""
    
    def __init__(self, repository_manager: RepositoryManager, metrics: MetricsCollector):
        self.repository_manager = repository_manager
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
        # Active sessions
        self.active_sessions: Dict[str, SessionInfo] = {}
        
        # Session cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the session manager."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
        self.logger.info("Session manager started")
    
    async def stop(self):
        """Stop the session manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Session manager stopped")
    
    async def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        
        # Create conversation in database
        conversation = await self.repository_manager.conversations.create_conversation(
            session_id=uuid.UUID(session_id),
            metadata=metadata or {}
        )
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            conversation_id=str(conversation.id),
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = session_info
        
        self.logger.info(f"Created session {session_id} with conversation {conversation.id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        return self.active_sessions.get(session_id)
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.utcnow()
    
    async def increment_task_count(self, session_id: str):
        """Increment task count for session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].task_count += 1
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a session."""
        session_info = self.active_sessions.get(session_id)
        if not session_info:
            return False
            
        # End conversation in database
        if session_info.conversation_id:
            await self.repository_manager.conversations.end_conversation(
                uuid.UUID(session_info.conversation_id)
            )
        
        # Update session status
        session_info.status = SessionStatus.TERMINATED
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        self.logger.info(f"Terminated session {session_id}")
        return True
    
    async def _cleanup_sessions(self):
        """Background task to cleanup idle sessions."""
        while self._running:
            try:
                now = datetime.utcnow()
                timeout = timedelta(seconds=settings.session_timeout)
                
                # Find sessions to cleanup
                sessions_to_cleanup = []
                for session_id, session_info in self.active_sessions.items():
                    if now - session_info.last_activity > timeout:
                        sessions_to_cleanup.append(session_id)
                
                # Cleanup idle sessions
                for session_id in sessions_to_cleanup:
                    await self.terminate_session(session_id)
                    self.logger.info(f"Cleaned up idle session {session_id}")
                
                # Check if we're over the session limit
                if len(self.active_sessions) > settings.max_concurrent_sessions:
                    # Remove oldest sessions
                    sessions_by_age = sorted(
                        self.active_sessions.items(),
                        key=lambda x: x[1].last_activity
                    )
                    
                    excess_count = len(self.active_sessions) - settings.max_concurrent_sessions
                    for session_id, _ in sessions_by_age[:excess_count]:
                        await self.terminate_session(session_id)
                        self.logger.info(f"Terminated session {session_id} due to session limit")
                
                await asyncio.sleep(settings.session_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        now = datetime.utcnow()
        
        active_count = len(self.active_sessions)
        idle_count = 0
        
        for session_info in self.active_sessions.values():
            if now - session_info.last_activity > timedelta(minutes=5):
                idle_count += 1
        
        return {
            "active_sessions": active_count,
            "idle_sessions": idle_count,
            "max_sessions": settings.max_concurrent_sessions,
            "utilization": active_count / settings.max_concurrent_sessions * 100 if settings.max_concurrent_sessions > 0 else 0
        }


class TaskQueue:
    """Priority-based task queue with retry logic."""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
        # Task queues by priority (higher number = higher priority)
        self.queues: Dict[int, deque] = defaultdict(lambda: deque())
        self.pending_tasks: Dict[str, TaskQueueItem] = {}
        self.failed_tasks: Dict[str, TaskQueueItem] = {}
        
        # Queue statistics
        self.processed_count = 0
        self.failed_count = 0
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def enqueue(self, task_item: TaskQueueItem):
        """Add task to queue."""
        async with self.lock:
            self.queues[task_item.priority].append(task_item)
            self.pending_tasks[task_item.task_id] = task_item
            
        self.logger.debug(f"Enqueued task {task_item.task_id} with priority {task_item.priority}")
    
    async def dequeue(self) -> Optional[TaskQueueItem]:
        """Get next task from queue (highest priority first)."""
        async with self.lock:
            # Check queues from highest to lowest priority
            for priority in sorted(self.queues.keys(), reverse=True):
                if self.queues[priority]:
                    task_item = self.queues[priority].popleft()
                    return task_item
            
            return None
    
    async def mark_completed(self, task_id: str):
        """Mark task as completed."""
        async with self.lock:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
                self.processed_count += 1
    
    async def mark_failed(self, task_id: str, error: str = None):
        """Mark task as failed and handle retry logic."""
        async with self.lock:
            if task_id not in self.pending_tasks:
                return
                
            task_item = self.pending_tasks[task_id]
            task_item.retries += 1
            
            if task_item.retries <= task_item.max_retries:
                # Re-queue for retry with lower priority
                task_item.priority = max(0, task_item.priority - 1)
                self.queues[task_item.priority].append(task_item)
                self.logger.info(f"Re-queued task {task_id} for retry {task_item.retries}/{task_item.max_retries}")
            else:
                # Move to failed tasks
                del self.pending_tasks[task_id]
                self.failed_tasks[task_id] = task_item
                self.failed_count += 1
                self.logger.error(f"Task {task_id} failed permanently after {task_item.retries} retries")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        total_pending = sum(len(queue) for queue in self.queues.values())
        
        return {
            "pending_tasks": total_pending,
            "processed_tasks": self.processed_count,
            "failed_tasks": self.failed_count,
            "queue_sizes_by_priority": {
                priority: len(queue) for priority, queue in self.queues.items()
            }
        }


class TaskManager:
    """Manages task lifecycle and execution coordination."""
    
    def __init__(
        self,
        repository_manager: RepositoryManager,
        session_manager: SessionManager,
        metrics: MetricsCollector
    ):
        self.repository_manager = repository_manager
        self.session_manager = session_manager
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
        # Task management components
        self.task_queue = TaskQueue(metrics)
        self.resource_pool = ResourcePool(settings.parallel_task_limit)
        
        # Task execution tracking
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Task processor
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the task manager."""
        if self._running:
            return
            
        self._running = True
        self._processor_task = asyncio.create_task(self._process_tasks())
        self.logger.info("Task manager started")
    
    async def stop(self):
        """Stop the task manager."""
        self._running = False
        
        # Cancel processor
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active executions
        for task in self.active_executions.values():
            task.cancel()
        
        if self.active_executions:
            await asyncio.gather(*self.active_executions.values(), return_exceptions=True)
        
        self.logger.info("Task manager stopped")
    
    async def submit_task(
        self,
        session_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit a new task for execution."""
        # Get or create session
        session_info = await self.session_manager.get_session(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found")
        
        # Create task execution in database
        task_execution = await self.repository_manager.tasks.create_task_execution(
            conversation_id=uuid.UUID(session_info.conversation_id),
            task_type=task_type,
            input_data=input_data
        )
        
        # Create queue item
        task_item = TaskQueueItem(
            task_id=str(task_execution.id),
            conversation_id=session_info.conversation_id,
            task_type=task_type,
            priority=priority,
            input_data=input_data
        )
        
        # Enqueue task
        await self.task_queue.enqueue(task_item)
        
        # Update session activity
        await self.session_manager.update_session_activity(session_id)
        await self.session_manager.increment_task_count(session_id)
        
        self.logger.info(f"Submitted task {task_execution.id} of type {task_type}")
        return str(task_execution.id)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task execution status."""
        task_execution = await self.repository_manager.tasks.get_task_execution(uuid.UUID(task_id))
        if not task_execution:
            return None
            
        return {
            "task_id": str(task_execution.id),
            "status": task_execution.status.value,
            "task_type": task_execution.task_type,
            "started_at": task_execution.started_at.isoformat() if task_execution.started_at else None,
            "completed_at": task_execution.completed_at.isoformat() if task_execution.completed_at else None,
            "output_data": task_execution.output_data
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task execution."""
        # Cancel if currently executing
        if task_id in self.active_executions:
            self.active_executions[task_id].cancel()
            del self.active_executions[task_id]
        
        # Update database
        await self.repository_manager.tasks.update_task_status(
            uuid.UUID(task_id),
            TaskStatus.CANCELLED
        )
        
        self.logger.info(f"Cancelled task {task_id}")
        return True
    
    async def _process_tasks(self):
        """Background task processor."""
        while self._running:
            try:
                # Get next task from queue
                task_item = await self.task_queue.dequeue()
                if not task_item:
                    await asyncio.sleep(1)  # No tasks available
                    continue
                
                # Try to acquire resources
                if not await self.resource_pool.acquire_task_slot(task_item.task_id, "orchestrator"):
                    # Re-queue task if no resources available
                    await self.task_queue.enqueue(task_item)
                    await asyncio.sleep(5)  # Wait before retrying
                    continue
                
                # Start task execution
                execution_task = asyncio.create_task(
                    self._execute_task(task_item)
                )
                self.active_executions[task_item.task_id] = execution_task
                
                # Don't await here - let it run in background
                
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(10)  # Wait on error
    
    async def _execute_task(self, task_item: TaskQueueItem):
        """Execute a single task."""
        task_id = task_item.task_id
        
        try:
            # Update task status to running
            await self.repository_manager.tasks.update_task_status(
                uuid.UUID(task_id),
                TaskStatus.RUNNING
            )
            
            # Record start time for metrics
            start_time = datetime.utcnow()
            
            # Here you would integrate with the actual orchestrator
            # For now, we'll simulate task execution
            await asyncio.sleep(1)  # Simulate work
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Mark task as completed
            await self.repository_manager.tasks.update_task_status(
                uuid.UUID(task_id),
                TaskStatus.COMPLETED,
                {"execution_time": execution_time}
            )
            
            await self.task_queue.mark_completed(task_id)
            
            # Record metrics
            await self.metrics.record_timing("task_execution", execution_time)
            
            self.logger.info(f"Completed task {task_id} in {execution_time:.2f}s")
            
        except asyncio.CancelledError:
            # Task was cancelled
            await self.repository_manager.tasks.update_task_status(
                uuid.UUID(task_id),
                TaskStatus.CANCELLED
            )
            self.logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # Task failed
            await self.repository_manager.tasks.update_task_status(
                uuid.UUID(task_id),
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            
            await self.task_queue.mark_failed(task_id, str(e))
            await self.metrics.record_error("task_execution", str(e))
            
            self.logger.error(f"Task {task_id} failed: {e}")
            
        finally:
            # Release resources
            await self.resource_pool.release_task_slot(task_id, "orchestrator")
            
            # Remove from active executions
            if task_id in self.active_executions:
                del self.active_executions[task_id]
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        return {
            "sessions": self.session_manager.get_session_stats(),
            "task_queue": self.task_queue.get_queue_stats(),
            "resources": self.resource_pool.get_load_info(),
            "active_executions": len(self.active_executions)
        }


class OrchestrationManager:
    """Main orchestration manager that coordinates all components."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        metrics: MetricsCollector
    ):
        self.db_manager = db_manager
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
        # Create repository manager
        self.repository_manager = None  # Will be set during initialization
        
        # Create managers
        self.session_manager = None
        self.task_manager = None
        
        self._running = False
    
    async def initialize(self):
        """Initialize the orchestration manager."""
        if self._running:
            return
        
        # Create repository manager with database session
        session = await self.db_manager.get_session()
        self.repository_manager = RepositoryManager(session)
        
        # Create and start managers
        self.session_manager = SessionManager(self.repository_manager, self.metrics)
        self.task_manager = TaskManager(
            self.repository_manager,
            self.session_manager,
            self.metrics
        )
        
        await self.session_manager.start()
        await self.task_manager.start()
        
        self._running = True
        self.logger.info("Orchestration manager initialized")
    
    async def cleanup(self):
        """Cleanup the orchestration manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop managers
        if self.task_manager:
            await self.task_manager.stop()
        
        if self.session_manager:
            await self.session_manager.stop()
        
        self.logger.info("Orchestration manager cleaned up")
    
    async def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """Create a new session."""
        if not self._running:
            raise RuntimeError("Orchestration manager not initialized")
        
        return await self.session_manager.create_session(metadata)
    
    async def submit_task(
        self,
        session_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit a task for execution."""
        if not self._running:
            raise RuntimeError("Orchestration manager not initialized")
        
        return await self.task_manager.submit_task(session_id, task_type, input_data, priority)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self._running:
            return {"status": "not_initialized"}
        
        stats = self.task_manager.get_manager_stats()
        
        return {
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "sessions": stats["sessions"],
            "task_queue": stats["task_queue"],
            "resources": stats["resources"],
            "active_executions": stats["active_executions"]
        }