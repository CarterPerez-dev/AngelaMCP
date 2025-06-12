"""
Claude Code Agent implementation for AngelaMCP.

This agent wraps Claude Code functionality for file operations and code execution.
I'm implementing this as the "senior developer" agent with file system access.
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.agents.base import BaseAgent, AgentType, AgentResponse, TaskContext, TaskType
from src.utils.logger import get_logger
from src.utils.exceptions import AgentError
from config.settings import settings


class ClaudeCodeAgent(BaseAgent):
    """
    Claude Code agent with file system and execution capabilities.
    
    This is the primary agent that can:
    - Create and modify files
    - Execute code
    - Manage project structure
    - Act as senior developer with final authority
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CLAUDE,
            name="Claude Code Senior Developer",
            capabilities=[
                "file_operations",
                "code_execution", 
                "project_management",
                "senior_review",
                "final_authority",
                "debugging",
                "testing"
            ]
        )
        
        self.claude_path = settings.claude_code_path
        self.timeout = settings.claude_code_timeout
        self.max_turns = settings.claude_code_max_turns
        
        # Verify Claude Code is available
        self._verify_claude_installation()
        
        logger.info(f"Initialized Claude Code agent at: {self.claude_path}")
    
    def _verify_claude_installation(self) -> None:
        """Verify Claude Code is installed and accessible."""
        try:
            result = subprocess.run(
                [str(self.claude_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise AgentError(f"Claude Code not working: {result.stderr}")
                
            self.logger.info(f"Claude Code verified: {result.stdout.strip()}")
            
        except FileNotFoundError:
            raise AgentError(f"Claude Code not found at: {self.claude_path}")
        except subprocess.TimeoutExpired:
            raise AgentError("Claude Code verification timed out")
    
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate response using Claude Code."""
        start_time = time.time()
        
        try:
            # Build Claude Code command
            cmd = self._build_claude_command(prompt, context)
            
            # Execute Claude Code
            result = await self._execute_claude_code(cmd)
            
            # Parse response
            response_content = self._parse_claude_response(result)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.9,  # High confidence for Claude Code
                execution_time_ms=execution_time * 1000,
                metadata={
                    "claude_version": self._get_claude_version(),
                    "command_used": " ".join(cmd),
                    "context": context.task_type.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Claude Code generation failed: {e}", exc_info=True)
            raise AgentError(f"Claude Code error: {e}")
    
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide senior developer critique using Claude Code."""
        critique_prompt = f"""As a senior developer, please provide a comprehensive code review of the following solution for: "{original_task}"

Solution to review:
{content}

Please provide:

**Technical Review:**
- Code quality assessment
- Architecture and design patterns
- Performance considerations
- Security implications

**Implementation Quality:**
- Best practices adherence
- Error handling
- Testing considerations
- Documentation quality

**Production Readiness:**
- Scalability concerns
- Maintenance considerations
- Deployment implications
- Monitoring and observability

**Specific Improvements:**
- Concrete code improvements
- Refactoring suggestions
- Alternative implementations
- Optimization opportunities

**Senior Developer Decision:**
- Overall assessment (Approve/Request Changes/Reject)
- Priority level of recommended changes
- Risk assessment for production deployment

Provide actionable, specific feedback that helps improve the solution."""

        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        critique_context.agent_role = "senior_reviewer"
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], context: TaskContext) -> AgentResponse:
        """Propose solution as senior developer using Claude Code."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""As a senior developer, design and implement a solution for:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Please provide a complete solution including:

**1. Architecture Design:**
- High-level system design
- Component breakdown
- Technology stack recommendations
- Design patterns to use

**2. Implementation Plan:**
- File structure and organization
- Key classes and functions
- Data models and interfaces
- Integration points

**3. Code Implementation:**
- Create actual working code files
- Include proper error handling
- Add comprehensive documentation
- Implement testing framework

**4. Deployment Considerations:**
- Environment setup
- Dependencies and requirements
- Configuration management
- Monitoring and logging

**5. Testing Strategy:**
- Unit tests
- Integration tests
- Performance tests
- Security testing

**6. Documentation:**
- README with setup instructions
- API documentation
- Developer guidelines
- User documentation

Please create actual files and provide a working implementation, not just pseudocode."""

        solution_context = context.model_copy()
        solution_context.task_type = TaskType.CODE_GENERATION
        solution_context.agent_role = "senior_developer"
        
        return await self.generate(solution_prompt, solution_context)
    
    async def execute_code(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Execute code using Claude Code capabilities."""
        execution_prompt = f"""Please execute the following {language} code and provide the results:

```{language}
{code}
```

If there are any errors, please:
1. Identify the issue
2. Provide a corrected version
3. Explain what was wrong
4. Execute the corrected version

Please show both the execution output and any debugging information."""

        execution_context = context.model_copy()
        execution_context.task_type = TaskType.CODE_GENERATION
        execution_context.metadata["execution_request"] = True
        
        return await self.generate(execution_prompt, execution_context)
    
    async def create_project(self, project_description: str, context: TaskContext) -> AgentResponse:
        """Create a complete project using Claude Code."""
        project_prompt = f"""Create a complete, production-ready project for:

{project_description}

Please:
1. Create proper project structure with folders
2. Implement all necessary files
3. Include comprehensive documentation
4. Add testing framework
5. Create deployment configuration
6. Include CI/CD setup if applicable

Make this a complete, working project that someone could clone and run immediately."""

        project_context = context.model_copy()
        project_context.task_type = TaskType.CODE_GENERATION
        project_context.metadata["project_creation"] = True
        
        return await self.generate(project_prompt, project_context)
    
    async def debug_issue(self, code: str, error: str, context: TaskContext) -> AgentResponse:
        """Debug issues using Claude Code expertise."""
        debug_prompt = f"""Please help debug the following issue:

**Code:**
```
{code}
```

**Error:**
{error}

As a senior developer, please:
1. Identify the root cause
2. Explain why this error occurred
3. Provide a corrected version
4. Suggest how to prevent similar issues
5. Add appropriate error handling
6. Include relevant tests to catch this type of error

Focus on both fixing the immediate issue and improving overall code robustness."""

        debug_context = context.model_copy()
        debug_context.task_type = TaskType.CODE_REVIEW
        debug_context.metadata["debugging"] = True
        
        return await self.generate(debug_prompt, debug_context)
    
    def _build_claude_command(self, prompt: str, context: TaskContext) -> List[str]:
        """Build Claude Code command with appropriate options."""
        cmd = [str(self.claude_path)]
        
        # Add context-specific options
        if context.task_type == TaskType.CODE_GENERATION:
            # For code generation, we want file operations
            pass
        elif context.task_type == TaskType.CODE_REVIEW:
            # For reviews, focus on analysis
            pass
        
        # Add the prompt as the final argument
        cmd.append(prompt)
        
        return cmd
    
    async def _execute_claude_code(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute Claude Code command asynchronously."""
        try:
            # Create a temporary file for complex prompts if needed
            if len(cmd[-1]) > 8000:  # Command line length limit
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(cmd[-1])
                    temp_file = f.name
                
                # Replace prompt with file input
                cmd = cmd[:-1] + ['--file', temp_file]
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise AgentError(f"Claude Code timed out after {self.timeout}s")
            
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout.decode('utf-8'),
                stderr=stderr.decode('utf-8')
            )
            
        except Exception as e:
            raise AgentError(f"Failed to execute Claude Code: {e}")
    
    def _parse_claude_response(self, result: subprocess.CompletedProcess) -> str:
        """Parse Claude Code response and extract content."""
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise AgentError(f"Claude Code failed: {error_msg}")
        
        response = result.stdout.strip()
        
        if not response:
            raise AgentError("Claude Code returned empty response")
        
        return response
    
    def _get_claude_version(self) -> str:
        """Get Claude Code version information."""
        try:
            result = subprocess.run(
                [str(self.claude_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Claude Code health and connectivity."""
        try:
            start_time = time.time()
            
            # Test basic functionality
            test_prompt = "Please respond with 'Claude Code is working' and nothing else."
            cmd = [str(self.claude_path), test_prompt]
            
            result = await self._execute_claude_code(cmd)
            response_time = time.time() - start_time
            
            # Check if response is reasonable
            if result.returncode == 0 and "working" in result.stdout.lower():
                status = "healthy"
            else:
                status = "degraded"
            
            return {
                "status": status,
                "claude_path": str(self.claude_path),
                "response_time": response_time,
                "version": self._get_claude_version(),
                "capabilities": self.capabilities,
                "last_check": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Claude Code health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "claude_path": str(self.claude_path),
                "last_check": time.time()
            }
