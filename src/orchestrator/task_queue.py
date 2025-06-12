"""
Async task queue for AngelaMCP orchestration.

This module provides an async task queue system for managing concurrent
agent operations and task execution with proper prioritization and resource management.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from datetime import datetime, timedelta
from collections import defaultdict

from src.utils import get_logger, monitor_performance
from src.utils import OrchestrationError
from config import settings


class TaskStatus(str, Enum):
    """Task status in the queue."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class QueuedTask:
    """A task in the execution queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable[..., Awaitable[Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Results
    result: Any = None
    error: Optional[Exception] = None
    
    # Context
    agent_name: Optional[str] = None
    operation_type: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def wait_time(self) -> float:
        """Get time spent waiting in queue."""
        start_time = self.started_at or datetime.now()
        return (start_time - self.created_at).total_seconds()


class QueueError(Exception):
    """Exception raised by task queue operations."""
    pass


class AsyncTaskQueue:
    """
    Async task queue for managing concurrent agent operations.
    
    Provides priority-based task scheduling, resource management,
    and monitoring for AngelaMCP orchestration.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = None,
        default_timeout: float = None,
        max_queue_size: int = None
    ):
        self.max_concurrent_tasks = max_concurrent_tasks or settings.parallel_task_limit
        self.default_timeout = default_timeout or settings.task_execution_timeout
        self.max_queue_size = max_queue_size or settings.task_queue_max_size
        
        self.logger = get_logger("orchestrator.task_queue")
        
        # Queue storage
        self._queues: Dict[TaskPriority, List[QueuedTask]] = {
            priority: [] for priority in TaskPriority
        }
        self._running_tasks: Dict[str, QueuedTask] = {}
        self._completed_tasks: Dict[str, QueuedTask] = {}
        
        # Semaphore for controlling concurrency
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Task tracking
        self._task_futures: Dict[str, asyncio.Task] = {}
        self._metrics = {
            "tasks_queued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time": 0.0
        }
        
        # Queue management
        self._shutdown = False
        self._worker_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the task queue worker."""
        if self._worker_task and not self._worker_task.done():
            self.logger.warning("Task queue already running")
            return
        
        self.logger.info("Starting async task queue")
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self) -> None:
        """Stop the task queue and wait for running tasks to complete."""
        self.logger.info("Stopping async task queue")
        self._shutdown = True
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending tasks
        await self._cancel_all_pending_tasks()
        
        # Wait for running tasks to complete
        if self._running_tasks:
            self.logger.info(f"Waiting for {len(self._running_tasks)} running tasks to complete")
            await self._wait_for_running_tasks()
    
    async def submit(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        agent_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task to the queue.
        
        Returns the task ID for tracking.
        """
        if self._shutdown:
            raise QueueError("Task queue is shutting down")
        
        # Check queue size limit
        total_queued = sum(len(queue) for queue in self._queues.values())
        if total_queued >= self.max_queue_size:
            raise QueueError(f"Queue size limit reached ({self.max_queue_size})")
        
        # Create task
        task = QueuedTask(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.default_timeout,
            max_retries=max_retries,
            agent_name=agent_name,
            operation_type=operation_type,
            context_data=context_data or {}
        )
        
        # Add to appropriate priority queue
        self._queues[priority].append(task)
        self._metrics["tasks_queued"] += 1
        
        self.logger.debug(
            f"Task {task.id[:8]} queued with priority {priority.value}",
            extra={
                "task_id": task.id,
                "priority": priority.value,
                "agent_name": agent_name,
                "operation_type": operation_type
            }
        )
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[QueuedTask]:
        """Get the status of a task."""
        # Check running tasks
        if task_id in self._running_tasks:
            return self._running_tasks[task_id]
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
        
        # Check queued tasks
        for queue in self._queues.values():
            for task in queue:
                if task.id == task_id:
                    return task
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's still pending or running."""
        # Check if task is running
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            if task_id in self._task_futures:
                future = self._task_futures[task_id]
                future.cancel()
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self._metrics["tasks_cancelled"] += 1
                return True
        
        # Check if task is queued
        for queue in self._queues.values():
            for i, task in enumerate(queue):
                if task.id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    queue.pop(i)
                    self._completed_tasks[task_id] = task
                    self._metrics["tasks_cancelled"] += 1
                    return True
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete and return its result."""
        start_time = time.time()
        
        while True:
            task = await self.get_task_status(task_id)
            
            if not task:
                raise QueueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                if task.error:
                    raise task.error
                else:
                    raise QueueError(f"Task {task_id} failed with status {task.status}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Short sleep to avoid busy waiting
            await asyncio.sleep(0.1)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics."""
        status = {
            "running_tasks": len(self._running_tasks),
            "queued_tasks": {
                priority.value: len(queue) 
                for priority, queue in self._queues.items()
            },
            "total_queued": sum(len(queue) for queue in self._queues.values()),
            "completed_tasks": len(self._completed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "max_queue_size": self.max_queue_size,
            "metrics": self._metrics.copy()
        }
        
        # Add average execution time
        if self._metrics["tasks_completed"] > 0:
            status["avg_execution_time"] = (
                self._metrics["total_execution_time"] / self._metrics["tasks_completed"]
            )
        
        return status
    
    async def _worker_loop(self) -> None:
        """Main worker loop that processes tasks from the queue."""
        self.logger.info("Task queue worker started")
        
        try:
            while not self._shutdown:
                # Get next task with highest priority
                next_task = self._get_next_task()
                
                if next_task:
                    # Try to acquire semaphore (non-blocking)
                    if self._semaphore.locked():
                        # Wait a bit if all slots are taken
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Execute task
                    asyncio.create_task(self._execute_task(next_task))
                else:
                    # No tasks available, sleep briefly
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            self.logger.info("Task queue worker cancelled")
        except Exception as e:
            self.logger.error(f"Task queue worker error: {e}", exc_info=True)
        finally:
            self.logger.info("Task queue worker stopped")
    
    def _get_next_task(self) -> Optional[QueuedTask]:
        """Get the next task to execute based on priority."""
        # Check queues in priority order
        priority_order = [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]
        
        for priority in priority_order:
            queue = self._queues[priority]
            if queue:
                task = queue.pop(0)  # FIFO within same priority
                return task
        
        return None
    
    async def _execute_task(self, task: QueuedTask) -> None:
        """Execute a single task."""
        async with self._semaphore:
            task_id = task.id
            
            try:
                # Move task to running state
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                self._running_tasks[task_id] = task
                
                self.logger.debug(
                    f"Executing task {task_id[:8]}",
                    extra={
                        "task_id": task_id,
                        "agent_name": task.agent_name,
                        "operation_type": task.operation_type,
                        "wait_time": task.wait_time
                    }
                )
                
                # Create and track the task future
                future = asyncio.create_task(
                    asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                )
                self._task_futures[task_id] = future
                
                # Execute with timeout
                with monitor_performance(f"task_execution_{task.operation_type or 'unknown'}"):
                    result = await future
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                self._metrics["tasks_completed"] += 1
                if task.execution_time:
                    self._metrics["total_execution_time"] += task.execution_time
                
                self.logger.debug(
                    f"Task {task_id[:8]} completed successfully in {task.execution_time:.3f}s",
                    extra={
                        "task_id": task_id,
                        "execution_time": task.execution_time,
                        "agent_name": task.agent_name,
                        "operation_type": task.operation_type
                    }
                )
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = asyncio.TimeoutError(f"Task timed out after {task.timeout}s")
                task.completed_at = datetime.now()
                self._metrics["tasks_failed"] += 1
                
                self.logger.warning(
                    f"Task {task_id[:8]} timed out after {task.timeout}s",
                    extra={"task_id": task_id, "timeout": task.timeout}
                )
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self._metrics["tasks_cancelled"] += 1
                
                self.logger.info(
                    f"Task {task_id[:8]} was cancelled",
                    extra={"task_id": task_id}
                )
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = e
                task.completed_at = datetime.now()
                self._metrics["tasks_failed"] += 1
                
                # Check if we should retry
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.completed_at = None
                    task.error = None
                    
                    # Re-queue the task
                    self._queues[task.priority].append(task)
                    
                    self.logger.info(
                        f"Task {task_id[:8]} failed, retrying ({task.retry_count}/{task.max_retries})",
                        extra={"task_id": task_id, "error": str(e)}
                    )
                else:
                    self.logger.error(
                        f"Task {task_id[:8]} failed permanently: {e}",
                        extra={"task_id": task_id, "error": str(e)},
                        exc_info=True
                    )
            
            finally:
                # Clean up
                if task_id in self._running_tasks:
                    del self._running_tasks[task_id]
                if task_id in self._task_futures:
                    del self._task_futures[task_id]
                
                # Move to completed tasks if not retrying
                if task.status != TaskStatus.PENDING:
                    self._completed_tasks[task_id] = task
    
    async def _cancel_all_pending_tasks(self) -> None:
        """Cancel all pending tasks in the queue."""
        cancelled_count = 0
        
        for priority, queue in self._queues.items():
            while queue:
                task = queue.pop()
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self._completed_tasks[task.id] = task
                cancelled_count += 1
        
        if cancelled_count > 0:
            self.logger.info(f"Cancelled {cancelled_count} pending tasks")
            self._metrics["tasks_cancelled"] += cancelled_count
    
    async def _wait_for_running_tasks(self, timeout: float = 30.0) -> None:
        """Wait for all running tasks to complete."""
        if not self._running_tasks:
            return
        
        try:
            # Wait for all running task futures
            futures = list(self._task_futures.values())
            if futures:
                await asyncio.wait_for(
                    asyncio.gather(*futures, return_exceptions=True),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for {len(self._running_tasks)} running tasks")
            
            # Force cancel remaining tasks
            for future in self._task_futures.values():
                if not future.done():
                    future.cancel()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed queue metrics."""
        return {
            **self._metrics,
            "queue_status": asyncio.run(self.get_queue_status()) if not self._shutdown else {},
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
