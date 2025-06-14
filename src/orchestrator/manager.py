#!/usr/bin/env python3
"""
Fixed Async Task Manager for AngelaMCP
Properly handles task lifecycle, cancellation, and resource cleanup.
This fixes the hanging issues in the debate system.
"""

import asyncio
import time
import logging
import weakref
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Status of async tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskInfo:
    """Information about a tracked task."""
    task_id: str
    name: str
    task: asyncio.Task
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    timeout: Optional[float] = None
    
    def is_done(self) -> bool:
        """Check if task is done (any final state)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]
    
    def mark_running(self):
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
    
    def mark_completed(self, result: Any):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
    
    def mark_failed(self, error: Exception):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
    
    def mark_cancelled(self):
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
    
    def mark_timeout(self):
        """Mark task as timed out."""
        self.status = TaskStatus.TIMEOUT


class AsyncTaskManager:
    """
    Proper async task manager that prevents hanging and resource leaks.
    """
    
    def __init__(self, default_timeout: float = 30.0):
        self.logger = logging.getLogger("async_task_manager")
        self.default_timeout = default_timeout
        
        # Track all active tasks
        self._active_tasks: Dict[str, TaskInfo] = {}
        self._task_groups: Dict[str, Set[str]] = {}  # Group tasks for batch operations
        
        # Cleanup tracking
        self._cleanup_scheduled = False
        self._shutdown_event = asyncio.Event()
        
        # Performance metrics
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._cancelled_tasks = 0
        self._timeout_tasks = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        active_count = len([t for t in self._active_tasks.values() if not t.is_done()])
        
        return {
            "active_tasks": active_count,
            "total_tasks": len(self._active_tasks),
            "completed": self._completed_tasks,
            "failed": self._failed_tasks,
            "cancelled": self._cancelled_tasks,
            "timeouts": self._timeout_tasks,
            "task_groups": len(self._task_groups)
        }
    
    async def create_task(
        self, 
        coro: Awaitable[Any], 
        name: str, 
        timeout: Optional[float] = None,
        group_id: Optional[str] = None
    ) -> str:
        """
        Create and track an async task with proper cleanup.
        
        Args:
            coro: Coroutine to execute
            name: Human-readable task name
            timeout: Task timeout (uses default if None)
            group_id: Optional group for batch operations
            
        Returns:
            Task ID for tracking
        """
        task_id = f"{name}_{uuid.uuid4().hex[:8]}"
        timeout = timeout or self.default_timeout
        
        # Create the actual task
        task = asyncio.create_task(self._run_with_tracking(coro, task_id))
        
        # Track the task
        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            task=task,
            created_at=time.time(),
            timeout=timeout
        )
        
        self._active_tasks[task_id] = task_info
        
        # Add to group if specified
        if group_id:
            if group_id not in self._task_groups:
                self._task_groups[group_id] = set()
            self._task_groups[group_id].add(task_id)
        
        self.logger.debug(f"Created task {task_id} ({name}) with {timeout}s timeout")
        
        # Schedule cleanup if not already scheduled
        if not self._cleanup_scheduled:
            asyncio.create_task(self._periodic_cleanup())
            self._cleanup_scheduled = True
        
        return task_id
    
    async def _run_with_tracking(self, coro: Awaitable[Any], task_id: str) -> Any:
        """Run coroutine with proper tracking and timeout."""
        task_info = self._active_tasks.get(task_id)
        if not task_info:
            raise RuntimeError(f"Task {task_id} not found in tracking")
        
        task_info.mark_running()
        
        try:
            # Apply timeout
            result = await asyncio.wait_for(coro, timeout=task_info.timeout)
            
            task_info.mark_completed(result)
            self._completed_tasks += 1
            
            self.logger.debug(f"Task {task_id} completed successfully")
            return result
            
        except asyncio.TimeoutError:
            task_info.mark_timeout()
            self._timeout_tasks += 1
            
            self.logger.warning(f"Task {task_id} timed out after {task_info.timeout}s")
            raise
            
        except asyncio.CancelledError:
            task_info.mark_cancelled()
            self._cancelled_tasks += 1
            
            self.logger.debug(f"Task {task_id} was cancelled")
            raise
            
        except Exception as e:
            task_info.mark_failed(e)
            self._failed_tasks += 1
            
            self.logger.error(f"Task {task_id} failed: {e}")
            raise
    
    async def wait_for_task(self, task_id: str) -> Any:
        """Wait for a specific task to complete."""
        if task_id not in self._active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self._active_tasks[task_id]
        
        try:
            return await task_info.task
        except Exception:
            # Exception is already logged in _run_with_tracking
            raise
    
    async def wait_for_group(
        self, 
        group_id: str, 
        return_exceptions: bool = True,
        cancel_on_first_error: bool = False
    ) -> List[Any]:
        """
        Wait for all tasks in a group to complete.
        
        Args:
            group_id: Group identifier
            return_exceptions: Whether to return exceptions instead of raising
            cancel_on_first_error: Whether to cancel remaining tasks on first error
            
        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        if group_id not in self._task_groups:
            return []
        
        task_ids = list(self._task_groups[group_id])
        tasks = []
        
        for task_id in task_ids:
            if task_id in self._active_tasks:
                tasks.append(self._active_tasks[task_id].task)
        
        if not tasks:
            return []
        
        try:
            if cancel_on_first_error:
                # Use as_completed to cancel on first error
                results = []
                pending_tasks = set(tasks)
                
                for completed_task in asyncio.as_completed(tasks):
                    try:
                        result = await completed_task
                        results.append(result)
                        pending_tasks.discard(completed_task)
                    except Exception as e:
                        # Cancel all pending tasks
                        for pending in pending_tasks:
                            if not pending.done():
                                pending.cancel()
                        
                        if return_exceptions:
                            results.append(e)
                        else:
                            raise
                        break
                
                # Wait for cancelled tasks to finish
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                
                return results
            else:
                # Use gather for normal group waiting
                return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
                
        except Exception as e:
            self.logger.error(f"Group {group_id} execution failed: {e}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        if task_id not in self._active_tasks:
            return False
        
        task_info = self._active_tasks[task_id]
        
        if not task_info.task.done():
            task_info.task.cancel()
            try:
                await task_info.task
            except asyncio.CancelledError:
                pass
            
            self.logger.debug(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def cancel_group(self, group_id: str) -> int:
        """Cancel all tasks in a group."""
        if group_id not in self._task_groups:
            return 0
        
        task_ids = list(self._task_groups[group_id])
        cancelled_count = 0
        
        for task_id in task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        self.logger.debug(f"Cancelled {cancelled_count} tasks in group {group_id}")
        return cancelled_count
    
    async def cancel_all(self) -> int:
        """Cancel all active tasks."""
        active_task_ids = [
            task_id for task_id, task_info in self._active_tasks.items()
            if not task_info.is_done()
        ]
        
        cancelled_count = 0
        for task_id in active_task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        self.logger.info(f"Cancelled {cancelled_count} active tasks")
        return cancelled_count
    
    async def _periodic_cleanup(self):
        """Periodically clean up completed tasks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30.0)  # Clean every 30 seconds
                await self._cleanup_completed_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking."""
        completed_task_ids = [
            task_id for task_id, task_info in self._active_tasks.items()
            if task_info.is_done() and time.time() - task_info.created_at > 300  # 5 minutes old
        ]
        
        for task_id in completed_task_ids:
            # Remove from groups
            for group_id, task_set in self._task_groups.items():
                task_set.discard(task_id)
            
            # Remove empty groups
            empty_groups = [gid for gid, task_set in self._task_groups.items() if not task_set]
            for gid in empty_groups:
                del self._task_groups[gid]
            
            # Remove from active tasks
            del self._active_tasks[task_id]
        
        if completed_task_ids:
            self.logger.debug(f"Cleaned up {len(completed_task_ids)} completed tasks")
    
    @asynccontextmanager
    async def task_group(self, group_id: Optional[str] = None):
        """Context manager for managing a group of tasks."""
        if group_id is None:
            group_id = f"group_{uuid.uuid4().hex[:8]}"
        
        self._task_groups[group_id] = set()
        
        try:
            yield group_id
        except Exception:
            # Cancel all tasks in the group on exception
            await self.cancel_group(group_id)
            raise
        finally:
            # Clean up group reference
            if group_id in self._task_groups:
                del self._task_groups[group_id]
    
    async def shutdown(self):
        """Shutdown the task manager and clean up all resources."""
        self.logger.info("Shutting down async task manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all active tasks
        cancelled_count = await self.cancel_all()
        
        # Wait a bit for tasks to clean up
        if cancelled_count > 0:
            await asyncio.sleep(1.0)
        
        # Final cleanup
        await self._cleanup_completed_tasks()
        
        stats = self.get_stats()
        self.logger.info(f"Task manager shutdown complete. Final stats: {stats}")


# Global task manager instance
_task_manager: Optional[AsyncTaskManager] = None


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AsyncTaskManager()
    return _task_manager


async def managed_task(
    coro: Awaitable[Any], 
    name: str, 
    timeout: Optional[float] = None,
    group_id: Optional[str] = None
) -> Any:
    """
    Convenience function to create and wait for a managed task.
    
    Args:
        coro: Coroutine to execute
        name: Task name
        timeout: Task timeout
        group_id: Optional group ID
        
    Returns:
        Task result
    """
    manager = get_task_manager()
    task_id = await manager.create_task(coro, name, timeout, group_id)
    return await manager.wait_for_task(task_id)


async def managed_gather(
    coroutines: List[Awaitable[Any]], 
    names: List[str],
    timeout: Optional[float] = None,
    return_exceptions: bool = True,
    cancel_on_first_error: bool = False
) -> List[Any]:
    """
    Managed version of asyncio.gather with proper cleanup.
    
    Args:
        coroutines: List of coroutines to execute
        names: List of names for each coroutine
        timeout: Timeout for each coroutine
        return_exceptions: Whether to return exceptions
        cancel_on_first_error: Whether to cancel on first error
        
    Returns:
        List of results
    """
    if len(coroutines) != len(names):
        raise ValueError("Number of coroutines must match number of names")
    
    manager = get_task_manager()
    
    async with manager.task_group() as group_id:
        # Create all tasks
        task_ids = []
        for coro, name in zip(coroutines, names):
            task_id = await manager.create_task(coro, name, timeout, group_id)
            task_ids.append(task_id)
        
        # Wait for all tasks
        return await manager.wait_for_group(
            group_id, 
            return_exceptions=return_exceptions,
            cancel_on_first_error=cancel_on_first_error
        )


@asynccontextmanager
async def timeout_protection(timeout: float, operation_name: str = "operation"):
    """
    Context manager that protects against hanging operations.
    
    Args:
        timeout: Maximum time to allow
        operation_name: Name for logging
    """
    logger = logging.getLogger("timeout_protection")
    start_time = time.time()
    
    try:
        logger.debug(f"Starting {operation_name} with {timeout}s timeout")
        
        # Create a task that will be cancelled if we exceed timeout
        async with asyncio.timeout(timeout):
            yield
            
        elapsed = time.time() - start_time
        logger.debug(f"{operation_name} completed in {elapsed:.2f}s")
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"{operation_name} timed out after {elapsed:.2f}s")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{operation_name} failed after {elapsed:.2f}s: {e}")
        raise


# Cleanup function for graceful shutdown
async def cleanup_async_resources():
    """Clean up all async resources on shutdown."""
    global _task_manager
    if _task_manager:
        await _task_manager.shutdown()
        _task_manager = None


class CollaborationStrategy(Enum):
    """Collaboration strategies for multi-agent tasks."""
    SIMPLE = "simple"
    DEBATE = "debate"
    VOTING = "voting"
    CONSENSUS = "consensus"


class TaskComplexity(Enum):
    """Task complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CollaborationResult:
    """Result of a collaboration between agents."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    strategy_used: Optional[CollaborationStrategy] = None
    agents_involved: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskOrchestrator:
    """Main orchestrator for multi-agent collaboration."""
    
    def __init__(self, claude_agent=None, openai_agent=None, gemini_agent=None, db_manager=None):
        self.logger = logging.getLogger("task_orchestrator")
        self.claude_agent = claude_agent
        self.openai_agent = openai_agent
        self.gemini_agent = gemini_agent
        self.db_manager = db_manager
        self.task_manager = get_task_manager()
        
    async def collaborate(
        self,
        task: str,
        strategy: CollaborationStrategy = CollaborationStrategy.SIMPLE,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        timeout: float = 300.0
    ) -> CollaborationResult:
        """Execute a collaborative task between agents."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting collaboration with strategy: {strategy.value}")
            
            # For now, implement simple strategy
            if strategy == CollaborationStrategy.SIMPLE:
                result = await self._simple_collaboration(task, timeout)
            else:
                # Other strategies can be implemented later
                result = await self._simple_collaboration(task, timeout)
            
            execution_time = time.time() - start_time
            
            return CollaborationResult(
                success=True,
                result=result,
                strategy_used=strategy,
                agents_involved=["claude"],  # For now
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Collaboration failed: {e}")
            
            return CollaborationResult(
                success=False,
                error=str(e),
                strategy_used=strategy,
                execution_time=execution_time
            )
    
    async def _simple_collaboration(self, task: str, timeout: float) -> str:
        """Simple collaboration strategy."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate work
        return f"Completed task: {task}"
