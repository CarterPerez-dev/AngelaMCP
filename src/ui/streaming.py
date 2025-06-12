"""
Real-time streaming UI components for AngelaMCP.

This module provides real-time streaming capabilities for displaying agent
output, task progress, and system events as they happen.
"""

import asyncio
import time
import queue
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.status import Status
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from src.agents.base import AgentResponse, BaseAgent
from src.orchestration.orchestrator import TaskResult
from src.utils.logger import get_logger

logger = get_logger("ui.streaming")


class StreamEventType(str, Enum):
    """Types of streaming events."""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    AGENT_RESPONSE = "agent_response"
    DEBATE_ROUND = "debate_round"
    VOTE_CAST = "vote_cast"
    ERROR = "error"
    LOG_MESSAGE = "log_message"
    PROGRESS_UPDATE = "progress_update"


@dataclass
class StreamEvent:
    """Event for the streaming system."""
    event_type: StreamEventType
    timestamp: float
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None


class OutputBuffer:
    """
    Circular buffer for managing streaming output.
    
    I'm implementing a thread-safe buffer that can store streaming events
    and provide them for real-time display.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.event_queue = asyncio.Queue()
        
    def add_event(self, event: StreamEvent):
        """Add an event to the buffer."""
        with self.lock:
            self.buffer.append(event)
        
        # Add to async queue for real-time processing
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # If queue is full, remove oldest item
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass
    
    def get_recent_events(self, count: int = 50, event_type: Optional[StreamEventType] = None) -> List[StreamEvent]:
        """Get recent events from the buffer."""
        with self.lock:
            events = list(self.buffer)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-count:]
    
    def get_events_since(self, timestamp: float) -> List[StreamEvent]:
        """Get events since a specific timestamp."""
        with self.lock:
            return [e for e in self.buffer if e.timestamp >= timestamp]
    
    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Async generator for streaming events."""
        while True:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
        
        # Clear async queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class RealTimeStreamer:
    """
    Real-time streaming display manager.
    
    I'm implementing a system that can display live updates from the
    multi-agent system including task progress, agent responses, and events.
    """
    
    def __init__(self, console: Console, buffer: OutputBuffer):
        self.console = console
        self.buffer = buffer
        self.logger = get_logger("ui.streaming_manager")
        
        # Display state
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_statuses: Dict[str, str] = {}
        self.last_update = time.time()
        
        # Streaming components
        self.progress_displays: Dict[str, Progress] = {}
        self.status_displays: Dict[str, Status] = {}
        
    def create_live_panel(self) -> Panel:
        """Create the main live display panel."""
        content = []
        
        # Active tasks section
        if self.active_tasks:
            tasks_table = Table(title="🚀 Active Tasks", style="bright_yellow")
            tasks_table.add_column("Task ID", style="dim")
            tasks_table.add_column("Type", style="bright_cyan")
            tasks_table.add_column("Progress", style="bright_green")
            tasks_table.add_column("Agent(s)", style="bright_magenta")
            tasks_table.add_column("Duration", style="bright_blue")
            
            for task_id, task_info in self.active_tasks.items():
                duration = time.time() - task_info.get("start_time", time.time())
                agents = ", ".join(task_info.get("agents", ["Unknown"]))
                progress = task_info.get("progress", 0)
                
                tasks_table.add_row(
                    task_id[:8],
                    task_info.get("type", "Unknown"),
                    f"{progress:.0f}%",
                    agents,
                    f"{duration:.1f}s"
                )
            
            content.append(tasks_table)
        
        # Agent status section
        if self.agent_statuses:
            agents_table = Table(title="🤖 Agent Status", style="bright_green")
            agents_table.add_column("Agent", style="bold")
            agents_table.add_column("Status", style="bright_cyan")
            agents_table.add_column("Last Activity", style="dim")
            
            for agent_name, status in self.agent_statuses.items():
                agents_table.add_row(
                    agent_name,
                    status,
                    "Just now"  # Would track actual last activity
                )
            
            content.append(agents_table)
        
        # Recent events
        recent_events = self.buffer.get_recent_events(10)
        if recent_events:
            events_tree = Tree("📋 Recent Events", style="bright_cyan")
            
            for event in recent_events[-5:]:  # Show last 5 events
                event_time = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                event_text = f"[{event_time}] {event.event_type.value}"
                
                event_node = events_tree.add(event_text)
                if event.source:
                    event_node.add(f"Source: {event.source}")
                
                # Add key data
                if "message" in event.data:
                    msg = event.data["message"]
                    if len(msg) > 100:
                        msg = msg[:100] + "..."
                    event_node.add(Text(msg, style="dim"))
            
            content.append(events_tree)
        
        # System status
        status_text = Text()
        status_text.append("🟢 System Active ", style="bright_green")
        status_text.append(f"| Last Update: {datetime.fromtimestamp(self.last_update).strftime('%H:%M:%S')} ", style="dim")
        status_text.append(f"| Events: {len(self.buffer.buffer)}", style="bright_blue")
        
        content.append(status_text)
        
        if not content:
            content.append(Text("🌟 Waiting for activity...", style="dim"))
        
        return Panel(
            Group(*content),
            title="🔴 Live Feed",
            style="bright_red"
        )
    
    async def start_streaming(self, update_callback: Optional[Callable] = None):
        """Start the streaming display."""
        self.logger.info("Starting real-time streaming")
        
        async for event in self.buffer.stream_events():
            await self.process_event(event)
            
            if update_callback:
                update_callback()
    
    async def process_event(self, event: StreamEvent):
        """Process a streaming event and update displays."""
        self.last_update = time.time()
        
        try:
            if event.event_type == StreamEventType.TASK_STARTED:
                await self._handle_task_started(event)
            elif event.event_type == StreamEventType.TASK_COMPLETED:
                await self._handle_task_completed(event)
            elif event.event_type == StreamEventType.AGENT_RESPONSE:
                await self._handle_agent_response(event)
            elif event.event_type == StreamEventType.PROGRESS_UPDATE:
                await self._handle_progress_update(event)
            elif event.event_type == StreamEventType.ERROR:
                await self._handle_error(event)
            
        except Exception as e:
            self.logger.error(f"Error processing stream event: {e}")
    
    async def _handle_task_started(self, event: StreamEvent):
        """Handle task started event."""
        task_id = event.data.get("task_id")
        if task_id:
            self.active_tasks[task_id] = {
                "type": event.data.get("task_type", "Unknown"),
                "start_time": event.timestamp,
                "progress": 0,
                "agents": event.data.get("agents", []),
                "status": "Starting"
            }
            
            self.logger.debug(f"Task started: {task_id}")
    
    async def _handle_task_completed(self, event: StreamEvent):
        """Handle task completed event."""
        task_id = event.data.get("task_id")
        if task_id and task_id in self.active_tasks:
            # Remove from active tasks
            del self.active_tasks[task_id]
            self.logger.debug(f"Task completed: {task_id}")
    
    async def _handle_agent_response(self, event: StreamEvent):
        """Handle agent response event."""
        agent_name = event.data.get("agent_name")
        if agent_name:
            success = event.data.get("success", False)
            status = "✅ Active" if success else "⚠️ Error"
            self.agent_statuses[agent_name] = status
    
    async def _handle_progress_update(self, event: StreamEvent):
        """Handle progress update event."""
        task_id = event.data.get("task_id")
        progress = event.data.get("progress", 0)
        
        if task_id and task_id in self.active_tasks:
            self.active_tasks[task_id]["progress"] = progress
            
            if "status" in event.data:
                self.active_tasks[task_id]["status"] = event.data["status"]
    
    async def _handle_error(self, event: StreamEvent):
        """Handle error event."""
        error_msg = event.data.get("error", "Unknown error")
        source = event.source or "System"
        
        self.logger.error(f"Stream error from {source}: {error_msg}")


class StreamingUI:
    """
    Main streaming UI coordinator.
    
    I'm providing a high-level interface for managing real-time streaming
    displays with automatic event capture and display updates.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.buffer = OutputBuffer(max_size=2000)
        self.streamer = RealTimeStreamer(self.console, self.buffer)
        self.logger = get_logger("ui.streaming_ui")
        
        # Event subscribers
        self.subscribers: List[Callable[[StreamEvent], None]] = []
        
        # Auto-capture settings
        self.auto_capture = True
        self.capture_task = None
        
    def subscribe(self, callback: Callable[[StreamEvent], None]):
        """Subscribe to streaming events."""
        self.subscribers.append(callback)
    
    def emit_event(self, event_type: StreamEventType, source: str, data: Dict[str, Any], 
                   correlation_id: Optional[str] = None):
        """Emit a streaming event."""
        event = StreamEvent(
            event_type=event_type,
            timestamp=time.time(),
            source=source,
            data=data,
            correlation_id=correlation_id
        )
        
        self.buffer.add_event(event)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                self.logger.error(f"Error in event subscriber: {e}")
    
    def emit_task_started(self, task_id: str, task_type: str, agents: List[str]):
        """Emit task started event."""
        self.emit_event(
            StreamEventType.TASK_STARTED,
            "orchestrator",
            {
                "task_id": task_id,
                "task_type": task_type,
                "agents": agents
            }
        )
    
    def emit_task_completed(self, task_id: str, success: bool, result: Optional[TaskResult] = None):
        """Emit task completed event."""
        data = {
            "task_id": task_id,
            "success": success
        }
        
        if result:
            data.update({
                "execution_time_ms": result.execution_time_ms,
                "cost_usd": result.total_cost_usd,
                "strategy": result.strategy_used.value if result.strategy_used else None
            })
        
        self.emit_event(StreamEventType.TASK_COMPLETED, "orchestrator", data)
    
    def emit_agent_response(self, agent_name: str, response: AgentResponse):
        """Emit agent response event."""
        self.emit_event(
            StreamEventType.AGENT_RESPONSE,
            agent_name,
            {
                "agent_name": agent_name,
                "success": response.success,
                "execution_time_ms": response.execution_time_ms,
                "cost_usd": response.cost_usd,
                "tokens_used": response.tokens_used,
                "error": response.error_message
            }
        )
    
    def emit_progress_update(self, task_id: str, progress: float, status: str):
        """Emit progress update event."""
        self.emit_event(
            StreamEventType.PROGRESS_UPDATE,
            "orchestrator",
            {
                "task_id": task_id,
                "progress": progress,
                "status": status
            }
        )
    
    def emit_error(self, source: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """Emit error event."""
        data = {"error": error_message}
        if details:
            data["details"] = details
        
        self.emit_event(StreamEventType.ERROR, source, data)
    
    def emit_log_message(self, source: str, level: str, message: str):
        """Emit log message event."""
        self.emit_event(
            StreamEventType.LOG_MESSAGE,
            source,
            {
                "level": level,
                "message": message
            }
        )
    
    def create_live_display(self) -> Panel:
        """Create the live display panel."""
        return self.streamer.create_live_panel()
    
    async def start_streaming(self, update_callback: Optional[Callable] = None):
        """Start streaming mode."""
        self.logger.info("Starting streaming UI")
        await self.streamer.start_streaming(update_callback)
    
    def get_recent_events(self, count: int = 50, event_type: Optional[StreamEventType] = None) -> List[StreamEvent]:
        """Get recent events."""
        return self.buffer.get_recent_events(count, event_type)
    
    def clear_events(self):
        """Clear all events."""
        self.buffer.clear()
        self.logger.info("Cleared streaming events")
