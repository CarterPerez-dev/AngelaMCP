"""
Claude Code agent implementation for AngelaMCP.

This module implements the Claude Code agent wrapper that provides file system access,
code execution capabilities, and session persistence. I'm implementing a production-grade
subprocess wrapper with comprehensive error handling and output parsing.
"""

import asyncio
import json
import os
import shlex
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType,
    AgentCapability, AgentError, AgentTimeoutError, AgentAuthenticationError
)
from src.utils.logger import get_logger, log_agent_interaction, AsyncPerformanceLogger
from config.settings import settings

logger = get_logger("agents.claude")


class ClaudeCodeError(AgentError):
    """Specific error for Claude Code operations."""
    pass


class ClaudeCodeAgent(BaseAgent):
    """
    Claude Code agent implementation with subprocess management.
    
    I'm implementing a comprehensive wrapper around the Claude Code CLI that provides
    file system access, code execution, session persistence, and robust error handling.
    """
    
    def __init__(self, name: str = "claude_code", claude_path: Optional[str] = None):
        super().__init__(AgentType.CLAUDE_CODE, name, settings)
        
        # Claude Code configuration
        self.claude_path = Path(claude_path or settings.claude_code_path)
        self.session_dir = Path(settings.claude_session_dir).expanduser()
        self.session_persist = settings.claude_session_persist
        self.max_turns = settings.claude_code_max_turns
        self.output_format = settings.claude_code_output_format
        
        # Active sessions tracking
        self._active_sessions: Dict[str, str] = {}  # conversation_id -> claude_session_id
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Verify Claude Code installation
        self._verify_claude_installation()
        
        # Define capabilities
        self._setup_capabilities()
        
        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Claude Code agent initialized with path: {self.claude_path}")
    
    def _verify_claude_installation(self) -> None:
        """Verify that Claude Code is installed and accessible."""
        if not self.claude_path.exists():
            raise ClaudeCodeError(
                f"Claude Code not found at {self.claude_path}",
                agent_type=self.agent_type.value,
                error_code="CLAUDE_NOT_FOUND"
            )
        
        if not os.access(self.claude_path, os.X_OK):
            raise ClaudeCodeError(
                f"Claude Code at {self.claude_path} is not executable",
                agent_type=self.agent_type.value,
                error_code="CLAUDE_NOT_EXECUTABLE"
            )
    
    def _setup_capabilities(self) -> None:
        """Define Claude Code agent capabilities."""
        self._capabilities = [
            AgentCapability(
                name="file_system_access",
                description="Read, write, and manipulate files and directories",
                supported_formats=["text", "code", "json", "yaml", "markdown"],
                limitations=["Cannot access files outside allowed directories"],
                cost_per_request=0.01
            ),
            AgentCapability(
                name="code_execution",
                description="Execute code and shell commands",
                supported_formats=["python", "bash", "shell"],
                limitations=["Sandboxed execution environment"],
                cost_per_request=0.02
            ),
            AgentCapability(
                name="code_generation",
                description="Generate, modify, and refactor code",
                supported_formats=["python", "javascript", "typescript", "java", "cpp", "go"],
                limitations=["Context length limits"],
                cost_per_request=0.015
            ),
            AgentCapability(
                name="debugging",
                description="Debug code and analyze errors",
                supported_formats=["stacktraces", "error_logs", "code"],
                limitations=["Cannot debug external services"],
                cost_per_request=0.02
            ),
            AgentCapability(
                name="session_persistence",
                description="Maintain conversation context across interactions",
                supported_formats=["conversation_state"],
                limitations=["Session storage limits"],
                cost_per_request=0.005
            )
        ]
    
    def _build_command(self, prompt: str, context: TaskContext) -> List[str]:
        """Build Claude Code command with appropriate flags."""
        cmd = [str(self.claude_path)]
        
        # Basic flags
        cmd.extend(["-p"])  # Print mode for non-interactive execution
        cmd.extend(["--output-format", self.output_format])
        cmd.extend(["--max-turns", str(self.max_turns)])
        
        # Session management
        if self.session_persist and context.conversation_id:
            session_id = self._get_or_create_session(context.conversation_id)
            if session_id:
                cmd.extend(["--resume", session_id])
        
        # Timeout handling
        if context.timeout_seconds:
            # Note: Claude Code doesn't have built-in timeout, we'll handle this at process level
            pass
        
        # Add the prompt
        cmd.append(prompt)
        
        return cmd
    
    def _get_or_create_session(self, conversation_id: str) -> Optional[str]:
        """Get existing session ID or create a new one."""
        if conversation_id in self._active_sessions:
            return self._active_sessions[conversation_id]
        
        # Generate new session ID
        session_id = str(uuid.uuid4())
        self._active_sessions[conversation_id] = session_id
        self._session_metadata[conversation_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_used": time.time(),
            "request_count": 0
        }
        
        return session_id
    
    async def _execute_command(self, cmd: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Execute Claude Code command with timeout and error handling."""
        timeout = timeout or self.timeout
        
        try:
            # Log command execution (with sanitized command for security)
            sanitized_cmd = cmd[:-1] + ["<prompt>"]  # Hide the actual prompt
            self.logger.debug(f"Executing Claude Code: {' '.join(sanitized_cmd)}")
            
            # Execute subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()  # Use current working directory
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                raise AgentTimeoutError(
                    f"Claude Code execution timed out after {timeout}s",
                    agent_type=self.agent_type.value,
                    error_code="EXECUTION_TIMEOUT"
                )
            
            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace').strip()
                self.logger.error(f"Claude Code failed with return code {process.returncode}: {error_msg}")
                
                # Handle specific error types
                if "authentication" in error_msg.lower():
                    raise AgentAuthenticationError(
                        f"Claude Code authentication failed: {error_msg}",
                        agent_type=self.agent_type.value,
                        error_code="AUTHENTICATION_FAILED"
                    )
                
                raise ClaudeCodeError(
                    f"Claude Code execution failed: {error_msg}",
                    agent_type=self.agent_type.value,
                    error_code="EXECUTION_FAILED",
                    metadata={"return_code": process.returncode, "stderr": error_msg}
                )
            
            # Parse output based on format
            output_text = stdout.decode('utf-8', errors='replace')
            
            if self.output_format == "json":
                try:
                    return json.loads(output_text)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON output: {e}")
                    # Fallback to text parsing
                    return {"type": "text", "result": output_text}
            else:
                return {"type": "text", "result": output_text}
        
        except (OSError, FileNotFoundError) as e:
            raise ClaudeCodeError(
                f"Failed to execute Claude Code: {e}",
                agent_type=self.agent_type.value,
                error_code="EXECUTION_ERROR"
            )
    
    def _parse_claude_response(self, output: Dict[str, Any], execution_time: float) -> AgentResponse:
        """Parse Claude Code output into standardized response format."""
        if output.get("type") == "result" and output.get("subtype") == "success":
            # JSON format success response
            return AgentResponse(
                success=True,
                content=output.get("result", ""),
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                cost_usd=output.get("cost_usd"),
                tokens_used=None,  # Claude Code doesn't report token usage directly
                metadata={
                    "num_turns": output.get("num_turns"),
                    "session_id": output.get("session_id"),
                    "duration_api_ms": output.get("duration_api_ms"),
                    "claude_output_type": output.get("type"),
                    "claude_subtype": output.get("subtype")
                }
            )
        elif output.get("type") == "result" and output.get("subtype") == "error_max_turns":
            # Maximum turns reached
            return AgentResponse(
                success=False,
                content=f"Maximum turns ({self.max_turns}) reached",
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                error_message="Maximum turns reached",
                metadata={
                    "num_turns": output.get("num_turns"),
                    "session_id": output.get("session_id"),
                    "claude_output_type": output.get("type"),
                    "claude_subtype": output.get("subtype")
                }
            )
        elif output.get("type") == "text":
            # Text format response
            return AgentResponse(
                success=True,
                content=output.get("result", ""),
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                metadata={"output_format": "text"}
            )
        else:
            # Unknown format - try to extract content
            content = str(output.get("result", output))
            return AgentResponse(
                success=True,
                content=content,
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                metadata={"raw_output": output}
            )
    
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate a response using Claude Code."""
        start_time = time.time()
        
        # Update session metadata
        if context.conversation_id and context.conversation_id in self._session_metadata:
            self._session_metadata[context.conversation_id]["last_used"] = start_time
            self._session_metadata[context.conversation_id]["request_count"] += 1
        
        async with AsyncPerformanceLogger(self.logger, "claude_generate", task_id=context.task_id):
            try:
                # Build and execute command
                cmd = self._build_command(prompt, context)
                output = await self.execute_with_retry(self._execute_command, cmd, context.timeout_seconds)
                
                execution_time = time.time() - start_time
                response = self._parse_claude_response(output, execution_time)
                
                # Update metrics
                self._update_metrics(response)
                
                # Log interaction
                log_agent_interaction(
                    self.logger,
                    self.name,
                    "generate",
                    input_data=prompt,
                    output_data=response.content,
                    metadata={
                        "task_id": context.task_id,
                        "session_id": context.session_id,
                        "execution_time_ms": response.execution_time_ms,
                        "cost_usd": response.cost_usd
                    }
                )
                
                return response
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Claude Code generation failed: {e}")
                
                return AgentResponse(
                    success=False,
                    content="",
                    agent_type=self.agent_type.value,
                    execution_time_ms=execution_time * 1000,
                    error_message=str(e),
                    metadata={"task_id": context.task_id}
                )
    
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide critique using Claude Code."""
        critique_prompt = f"""Please review and critique the following solution for the task: "{original_task}"

Solution to review:
{content}

Please provide:
1. Strengths of the solution
2. Potential weaknesses or issues
3. Specific suggestions for improvement
4. Overall assessment

Focus on correctness, efficiency, maintainability, and best practices."""

        # Update context for critique task
        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], 
                             context: TaskContext) -> AgentResponse:
        """Propose a solution using Claude Code."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""Please propose a solution for the following task:

Task: {task_description}

Constraints:
{constraints_text}

Please provide:
1. A clear explanation of your approach
2. Implementation details
3. Code examples if applicable
4. Potential challenges and how to address them
5. Testing strategy

Focus on creating a practical, well-structured solution."""

        # Update context for solution proposal
        solution_context = context.model_copy()
        solution_context.task_type = TaskType.CODE_GENERATION
        
        return await self.generate(solution_prompt, solution_context)
    
    async def execute_code(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Execute code using Claude Code's execution capabilities."""
        execution_prompt = f"""Please execute the following {language} code and provide the output:

```{language}
{code}
```

Please show:
1. The execution output
2. Any errors or warnings
3. Explanation of the results if needed"""

        execution_context = context.model_copy()
        execution_context.task_type = TaskType.CUSTOM
        
        return await self.generate(execution_prompt, execution_context)
    
    async def debug_code(self, code: str, error_message: str, context: TaskContext) -> AgentResponse:
        """Debug code and error using Claude Code."""
        debug_prompt = f"""Please help debug this code issue:

Code:
```
{code}
```

Error message:
{error_message}

Please provide:
1. Analysis of the error
2. Root cause identification
3. Specific fix recommendations
4. Corrected code if applicable
5. Prevention strategies for similar issues"""

        debug_context = context.model_copy()
        debug_context.task_type = TaskType.DEBUGGING
        
        return await self.generate(debug_prompt, debug_context)
    
    def get_session_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a conversation session."""
        return self._session_metadata.get(conversation_id)
    
    def cleanup_session(self, conversation_id: str) -> None:
        """Clean up session data for a conversation."""
        if conversation_id in self._active_sessions:
            del self._active_sessions[conversation_id]
        
        if conversation_id in self._session_metadata:
            del self._session_metadata[conversation_id]
        
        self.logger.debug(f"Cleaned up session for conversation {conversation_id}")
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions."""
        return self._session_metadata.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check specific to Claude Code."""
        try:
            # Test basic functionality
            test_context = TaskContext(
                task_id=str(uuid.uuid4()),
                task_type=TaskType.CUSTOM,
                timeout_seconds=30
            )
            
            start_time = time.time()
            response = await self.generate("Say 'health check successful'", test_context)
            execution_time = time.time() - start_time
            
            health_info = {
                "status": "healthy" if response.success else "unhealthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "claude_path": str(self.claude_path),
                "claude_accessible": self.claude_path.exists(),
                "session_persistence": self.session_persist,
                "active_sessions": len(self._active_sessions),
                "response_time_ms": execution_time * 1000,
                "test_response_success": response.success,
                "capabilities_count": len(self._capabilities),
                "performance_metrics": self.performance_metrics
            }
            
            if not response.success:
                health_info["error"] = response.error_message
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"Claude Code health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "error": str(e),
                "claude_path": str(self.claude_path),
                "claude_accessible": self.claude_path.exists(),
                "performance_metrics": self.performance_metrics
            }
    
    async def shutdown(self) -> None:
        """Shutdown Claude Code agent and cleanup resources."""
        self.logger.info("Shutting down Claude Code agent")
        
        # Log session information before cleanup
        if self._active_sessions:
            self.logger.info(f"Cleaning up {len(self._active_sessions)} active sessions")
            for conversation_id, session_info in self._session_metadata.items():
                self.logger.debug(f"Session {conversation_id}: {session_info['request_count']} requests")
        
        # Clear session data
        self._active_sessions.clear()
        self._session_metadata.clear()
        
        # Call parent shutdown
        await super().shutdown()
