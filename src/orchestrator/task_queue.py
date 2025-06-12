"""
Task Queue module for AngelaMCP.
Advanced task queuing system with priority, scheduling, and retry capabilities.
"""

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from src.models.database import TaskStatus
from src.orchestrator.task_analyzer import TaskAnalysis, TaskComplexity


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class QueueType(Enum):
    """Queue types for different task categories."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RETRY = "retry"
    BATCH = "batch"


@dataclass
class QueuedTask:
    """Represents a task in the queue."""
    task_id: str
    priority: TaskPriority
    queue_type: QueueType
    scheduled_time: datetime
    task_data: Dict[str, Any]
    analysis: Optional[TaskAnalysis] = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    callback: Optional[Callable] = None
    timeout: Optional[int] = None
    
    def __lt__(self, other):
        """Comparison for priority queue (lower priority value = higher priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.scheduled_time < other.scheduled_time
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready for execution."""
        return (
            datetime.utcnow() >= self.scheduled_time and
            not self.dependencies and
            self.attempts < self.max_attempts
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if not self.timeout:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.timeout)


class TaskQueue:
    """Advanced task queue with multiple queue types and priority handling."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
        
        # Different queue types
        self.immediate_queue: List[QueuedTask] = []
        self.scheduled_queue: List[QueuedTask] = []
        self.retry_queue: List[QueuedTask] = []
        self.batch_queue: deque = deque()
        
        # Task tracking
        self.active_tasks: Dict[str, QueuedTask] = {}
        self.completed_tasks: deque = deque(maxlen=100)  # Keep last 100
        self.failed_tasks: deque = deque(maxlen=50)      # Keep last 50
        
        # Dependencies tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        
        # Queue statistics
        self.stats = {
            "enqueued": 0,
            "dequeued": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0
        }
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start background queue management tasks."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_tasks())
        self.logger.info("Task queue started")
    
    async def stop(self):
        """Stop background tasks."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Task queue stopped")
    
    async def enqueue(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        queue_type: QueueType = QueueType.IMMEDIATE,
        scheduled_time: Optional[datetime] = None,
        dependencies: Optional[Set[str]] = None,
        analysis: Optional[TaskAnalysis] = None,
        callback: Optional[Callable] = None,
        timeout: Optional[int] = None,
        max_attempts: int = 3
    ) -> bool:
        """Enqueue a task for execution."""
        async with self.lock:
            # Check queue capacity
            if self._get_total_queue_size() >= self.max_size:
                self.logger.warning(f"Queue at capacity ({self.max_size}), rejecting task {task_id}")
                return False
            
            # Create queued task
            queued_task = QueuedTask(
                task_id=task_id,
                priority=priority,
                queue_type=queue_type,
                scheduled_time=scheduled_time or datetime.utcnow(),
                task_data=task_data,
                analysis=analysis,
                dependencies=dependencies or set(),
                callback=callback,
                timeout=timeout,
                max_attempts=max_attempts
            )
            
            # Add to appropriate queue
            if queue_type == QueueType.IMMEDIATE:
                heapq.heappush(self.immediate_queue, queued_task)
            elif queue_type == QueueType.SCHEDULED:
                heapq.heappush(self.scheduled_queue, queued_task)
            elif queue_type == QueueType.RETRY:
                heapq.heappush(self.retry_queue, queued_task)
            elif queue_type == QueueType.BATCH:
                self.batch_queue.append(queued_task)
            
            # Update dependency tracking
            if dependencies:
                self.dependency_graph[task_id] = dependencies.copy()
                for dep in dependencies:
                    self.dependents[dep].add(task_id)
            
            self.stats["enqueued"] += 1
            self.logger.debug(f"Enqueued task {task_id} with priority {priority.name}")
            
            return True
    
    async def dequeue(self, agent_type: Optional[str] = None) -> Optional[QueuedTask]:
        """Dequeue the next ready task."""
        async with self.lock:
            # Check queues in priority order
            for queue_list in [self.immediate_queue, self.retry_queue, self.scheduled_queue]:
                if queue_list:
                    # Find the first ready task
                    ready_tasks = []
                    non_ready_tasks = []
                    
                    while queue_list:
                        task = heapq.heappop(queue_list)
                        if task.is_ready and not task.is_expired:
                            ready_tasks.append(task)
                        elif not task.is_expired:
                            non_ready_tasks.append(task)
                        else:
                            # Task expired, move to failed
                            self.failed_tasks.append(task)
                            self.stats["failed"] += 1
                    
                    # Put non-ready tasks back
                    for task in non_ready_tasks:
                        heapq.heappush(queue_list, task)
                    
                    # Return highest priority ready task
                    if ready_tasks:
                        task = min(ready_tasks)  # Lowest priority value = highest priority
                        # Put other ready tasks back
                        for other_task in ready_tasks:
                            if other_task != task:
                                heapq.heappush(queue_list, other_task)
                        
                        # Mark as active
                        self.active_tasks[task.task_id] = task
                        self.stats["dequeued"] += 1
                        
                        self.logger.debug(f"Dequeued task {task.task_id}")
                        return task
            
            # Check batch queue if no priority tasks
            if self.batch_queue:
                task = self.batch_queue.popleft()
                if task.is_ready and not task.is_expired:
                    self.active_tasks[task.task_id] = task
                    self.stats["dequeued"] += 1
                    return task
                elif not task.is_expired:
                    # Put back if not ready
                    self.batch_queue.appendleft(task)
                else:
                    # Task expired
                    self.failed_tasks.append(task)
                    self.stats["failed"] += 1
            
            return None
    
    async def mark_completed(self, task_id: str, result: Any = None):
        """Mark a task as completed."""
        async with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                task.task_data["result"] = result
                self.completed_tasks.append(task)
                self.stats["completed"] += 1
                
                # Execute callback if provided
                if task.callback:
                    try:
                        await task.callback(task_id, result)
                    except Exception as e:
                        self.logger.error(f"Callback error for task {task_id}: {e}")
                
                # Resolve dependencies
                await self._resolve_dependencies(task_id)
                
                self.logger.debug(f"Task {task_id} completed")
    
    async def mark_failed(self, task_id: str, error: Exception, retry: bool = True):
        """Mark a task as failed and optionally retry."""
        async with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                task.attempts += 1
                task.last_attempt = datetime.utcnow()
                task.task_data["last_error"] = str(error)
                
                if retry and task.attempts < task.max_attempts:
                    # Calculate retry delay with exponential backoff
                    delay = min(60, 2 ** task.attempts)  # Max 60 seconds
                    task.scheduled_time = datetime.utcnow() + timedelta(seconds=delay)
                    task.queue_type = QueueType.RETRY
                    
                    # Re-enqueue for retry
                    heapq.heappush(self.retry_queue, task)
                    self.stats["retried"] += 1
                    
                    self.logger.info(f"Task {task_id} failed, retrying in {delay}s (attempt {task.attempts}/{task.max_attempts})")
                else:
                    # Max attempts reached or no retry
                    self.failed_tasks.append(task)
                    self.stats["failed"] += 1
                    
                    # Fail dependent tasks
                    await self._fail_dependents(task_id)
                    
                    self.logger.error(f"Task {task_id} failed permanently after {task.attempts} attempts")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        async with self.lock:
            # Remove from active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                self.failed_tasks.append(task)
                await self._fail_dependents(task_id)
                self.logger.info(f"Cancelled active task {task_id}")
                return True
            
            # Remove from queues
            for queue_list in [self.immediate_queue, self.scheduled_queue, self.retry_queue]:
                updated_queue = [t for t in queue_list if t.task_id != task_id]
                if len(updated_queue) != len(queue_list):
                    queue_list.clear()
                    for task in updated_queue:
                        heapq.heappush(queue_list, task)
                    self.logger.info(f"Cancelled queued task {task_id}")
                    return True
            
            # Check batch queue
            original_len = len(self.batch_queue)
            self.batch_queue = deque(t for t in self.batch_queue if t.task_id != task_id)
            if len(self.batch_queue) != original_len:
                self.logger.info(f"Cancelled batch task {task_id}")
                return True
            
            return False
    
    async def add_dependency(self, task_id: str, dependency_id: str):
        """Add a dependency for a task."""
        async with self.lock:
            self.dependency_graph[task_id].add(dependency_id)
            self.dependents[dependency_id].add(task_id)
            
            # Update task in queues to reflect new dependency
            await self._update_task_dependencies(task_id)
    
    async def remove_dependency(self, task_id: str, dependency_id: str):
        """Remove a dependency for a task."""
        async with self.lock:
            self.dependency_graph[task_id].discard(dependency_id)
            self.dependents[dependency_id].discard(task_id)
            
            # Update task in queues
            await self._update_task_dependencies(task_id)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        async with self.lock:
            return {
                "queue_sizes": {
                    "immediate": len(self.immediate_queue),
                    "scheduled": len(self.scheduled_queue),
                    "retry": len(self.retry_queue),
                    "batch": len(self.batch_queue),
                    "active": len(self.active_tasks)
                },
                "totals": self.stats.copy(),
                "capacity": {
                    "max_size": self.max_size,
                    "current_size": self._get_total_queue_size(),
                    "utilization": self._get_total_queue_size() / self.max_size * 100
                },
                "recent_completed": len(self.completed_tasks),
                "recent_failed": len(self.failed_tasks)
            }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        async with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "status": "active",
                    "priority": task.priority.name,
                    "queue_type": task.queue_type.value,
                    "attempts": task.attempts,
                    "created_at": task.created_at.isoformat(),
                    "dependencies": list(task.dependencies)
                }
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return {
                        "status": "completed",
                        "priority": task.priority.name,
                        "attempts": task.attempts,
                        "created_at": task.created_at.isoformat(),
                        "result": task.task_data.get("result")
                    }
            
            # Check failed tasks
            for task in self.failed_tasks:
                if task.task_id == task_id:
                    return {
                        "status": "failed",
                        "priority": task.priority.name,
                        "attempts": task.attempts,
                        "created_at": task.created_at.isoformat(),
                        "last_error": task.task_data.get("last_error")
                    }
            
            # Check all queues
            for queue_name, queue_list in [
                ("immediate", self.immediate_queue),
                ("scheduled", self.scheduled_queue),
                ("retry", self.retry_queue)
            ]:
                for task in queue_list:
                    if task.task_id == task_id:
                        return {
                            "status": "queued",
                            "queue": queue_name,
                            "priority": task.priority.name,
                            "scheduled_time": task.scheduled_time.isoformat(),
                            "is_ready": task.is_ready,
                            "dependencies": list(task.dependencies)
                        }
            
            # Check batch queue
            for task in self.batch_queue:
                if task.task_id == task_id:
                    return {
                        "status": "queued",
                        "queue": "batch",
                        "priority": task.priority.name,
                        "is_ready": task.is_ready,
                        "dependencies": list(task.dependencies)
                    }
            
            return None
    
    async def _resolve_dependencies(self, completed_task_id: str):
        """Resolve dependencies when a task completes."""
        for dependent_id in self.dependents[completed_task_id]:
            self.dependency_graph[dependent_id].discard(completed_task_id)
            await self._update_task_dependencies(dependent_id)
        
        # Clean up dependency tracking
        del self.dependents[completed_task_id]
        if completed_task_id in self.dependency_graph:
            del self.dependency_graph[completed_task_id]
    
    async def _fail_dependents(self, failed_task_id: str):
        """Fail all tasks that depend on a failed task."""
        dependents_to_fail = list(self.dependents[failed_task_id])
        
        for dependent_id in dependents_to_fail:
            await self.cancel_task(dependent_id)
            self.logger.warning(f"Failed dependent task {dependent_id} due to failure of {failed_task_id}")
    
    async def _update_task_dependencies(self, task_id: str):
        """Update a task's dependencies in the queues."""
        # This is a simplified version - in a real implementation,
        # you'd need to find and update the task in the appropriate queue
        pass
    
    async def _cleanup_expired_tasks(self):
        """Background task to clean up expired tasks."""
        while self._running:
            try:
                async with self.lock:
                    current_time = datetime.utcnow()
                    
                    # Clean up all queues
                    for queue_list in [self.immediate_queue, self.scheduled_queue, self.retry_queue]:
                        valid_tasks = []
                        while queue_list:
                            task = heapq.heappop(queue_list)
                            if not task.is_expired:
                                valid_tasks.append(task)
                            else:
                                self.failed_tasks.append(task)
                                self.stats["failed"] += 1
                                self.logger.debug(f"Expired task {task.task_id}")
                        
                        # Rebuild heap
                        for task in valid_tasks:
                            heapq.heappush(queue_list, task)
                    
                    # Clean batch queue
                    valid_batch_tasks = []
                    while self.batch_queue:
                        task = self.batch_queue.popleft()
                        if not task.is_expired:
                            valid_batch_tasks.append(task)
                        else:
                            self.failed_tasks.append(task)
                            self.stats["failed"] += 1
                    
                    self.batch_queue.extend(valid_batch_tasks)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def _get_total_queue_size(self) -> int:
        """Get total size across all queues."""
        return (
            len(self.immediate_queue) +
            len(self.scheduled_queue) +
            len(self.retry_queue) +
            len(self.batch_queue) +
            len(self.active_tasks)
        )


class PriorityTaskScheduler:
    """High-level scheduler that uses TaskQueue with intelligent priority assignment."""
    
    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.logger = logging.getLogger(__name__)
    
    async def schedule_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        analysis: Optional[TaskAnalysis] = None,
        scheduled_time: Optional[datetime] = None,
        dependencies: Optional[Set[str]] = None
    ) -> bool:
        """Schedule a task with intelligent priority assignment."""
        
        # Determine priority based on analysis
        priority = self._calculate_priority(analysis, task_data)
        
        # Determine queue type
        queue_type = self._determine_queue_type(analysis, scheduled_time)
        
        # Calculate timeout
        timeout = self._calculate_timeout(analysis)
        
        # Calculate max attempts
        max_attempts = self._calculate_max_attempts(analysis)
        
        return await self.task_queue.enqueue(
            task_id=task_id,
            task_data=task_data,
            priority=priority,
            queue_type=queue_type,
            scheduled_time=scheduled_time,
            dependencies=dependencies,
            analysis=analysis,
            timeout=timeout,
            max_attempts=max_attempts
        )
    
    def _calculate_priority(self, analysis: Optional[TaskAnalysis], task_data: Dict[str, Any]) -> TaskPriority:
        """Calculate task priority based on analysis."""
        if not analysis:
            return TaskPriority.NORMAL
        
        # Critical complexity always gets high priority
        if analysis.complexity == TaskComplexity.CRITICAL:
            return TaskPriority.CRITICAL
        
        # Check for urgency indicators
        if analysis.priority >= 8:
            return TaskPriority.HIGH
        elif analysis.priority >= 6:
            return TaskPriority.NORMAL
        elif analysis.priority >= 3:
            return TaskPriority.LOW
        else:
            return TaskPriority.BACKGROUND
    
    def _determine_queue_type(self, analysis: Optional[TaskAnalysis], scheduled_time: Optional[datetime]) -> QueueType:
        """Determine appropriate queue type."""
        if scheduled_time and scheduled_time > datetime.utcnow():
            return QueueType.SCHEDULED
        
        if analysis and analysis.complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            return QueueType.BATCH
        
        return QueueType.IMMEDIATE
    
    def _calculate_timeout(self, analysis: Optional[TaskAnalysis]) -> Optional[int]:
        """Calculate task timeout based on analysis."""
        if not analysis:
            return 3600  # 1 hour default
        
        # Base timeout on estimated duration with buffer
        base_timeout = analysis.estimated_duration * 3  # 3x buffer
        
        # Minimum and maximum timeouts
        return max(300, min(7200, base_timeout))  # 5 min to 2 hours
    
    def _calculate_max_attempts(self, analysis: Optional[TaskAnalysis]) -> int:
        """Calculate maximum retry attempts."""
        if not analysis:
            return 3
        
        if analysis.complexity == TaskComplexity.CRITICAL:
            return 5  # More retries for critical tasks
        elif analysis.complexity == TaskComplexity.TRIVIAL:
            return 1  # Fewer retries for simple tasks
        else:
            return 3  # Standard retries