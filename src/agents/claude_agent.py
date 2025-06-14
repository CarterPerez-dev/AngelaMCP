"""
Claude Code Agent for AngelaMCP.

This agent integrates with Claude Code for file operations, code execution,
and project development. I'm implementing a production-grade agent that
leverages Claude's full capabilities as the senior developer.
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.agents.base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType, 
    AgentCapabilities, TokenUsage, track_performance
)
from src.utils import get_logger
from src.utils import AgentError
from config import settings


class ClaudeCodeAgent(BaseAgent):
    """
    Claude Code agent for file operations and code execution.
    
    This agent acts as the senior developer with full file system access
    and code execution capabilities through Claude Code.
    """
    
    def __init__(self):
        # Define Claude Code capabilities
        capabilities = AgentCapabilities(
            can_execute_code=True,
            can_read_files=True,
            can_write_files=True,
            can_browse_web=False,  # Limited in current Claude Code version
            can_use_tools=True,
            supported_languages=[
                "python", "javascript", "typescript", "java", "cpp", "c", 
                "go", "rust", "ruby", "php", "swift", "kotlin", "scala",
                "html", "css", "sql", "bash", "powershell", "yaml", "json"
            ],
            supported_formats=[
                "text", "markdown", "code", "json", "yaml", "xml", "csv"
            ],
            max_context_length=200000,  # Claude's context window
            supports_streaming=True,
            supports_function_calling=True
        )
        
        super().__init__(AgentType.CLAUDE, "claude", capabilities)
        
        # Claude Code configuration
        self.claude_code_path = settings.claude_code_path
        self.timeout = settings.claude_code_timeout
        self.max_turns = settings.claude_code_max_turns
        self.session_dir = settings.claude_session_dir
        
        # Verify Claude Code installation
        self._verify_claude_installation()
        
        # Session management
        self.current_session: Optional[str] = None
        self.session_persistent = settings.claude_session_persist
    
    def _verify_claude_installation(self) -> None:
        """Verify Claude Code is installed and accessible."""
        try:
            if not self.claude_code_path.exists():
                raise AgentError(f"Claude Code not found at {self.claude_code_path}")
            
            # Test basic Claude Code functionality
            result = subprocess.run(
                [str(self.claude_code_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise AgentError(f"Claude Code version check failed: {result.stderr}")
            
            self.logger.info(f"✅ Claude Code verified: {result.stdout.strip()}")
            
        except Exception as e:
            self.logger.error(f"❌ Claude Code verification failed: {e}")
            raise AgentError(f"Claude Code setup failed: {e}")
    
    @track_performance
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate response using Claude Code.
        
        This method handles the core interaction with Claude Code,
        managing sessions and parsing responses.
        """
        start_time = time.time()
        
        try:
            self.agent_logger.log_request(f"Generating response for {context.task_type.value} task")
            
            # Build Claude Code command - simplified without unsupported --format
            cmd = await self._build_claude_command(prompt, context)
            
            # Execute Claude Code
            result = await self._execute_claude_code(cmd)
            
            # Parse response - expect plain text
            response_content = result.stdout.strip() if result.stdout else ""
            
            if not response_content:
                response_content = "I'm ready to help with your request. What would you like me to do?"
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create response
            response = AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content=response_content,
                success=True,
                confidence=0.9,  # Claude is highly reliable
                execution_time_ms=execution_time,
                token_usage=self._estimate_token_usage(prompt, response_content),
                metadata={
                    "claude_session": self.current_session,
                    "task_type": context.task_type.value,
                    "agent_role": context.agent_role.value if context.agent_role else None,
                    "claude_code_version": "latest"
                }
            )
            
            self.agent_logger.log_response(f"Generated {len(response_content)} characters")
            return response
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Claude Code generation failed: {e}"
            
            self.agent_logger.log_error(error_msg, e)
            
            return AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content="",
                success=False,
                execution_time_ms=execution_time,
                error=error_msg
            )
    
    async def _build_claude_command(self, prompt: str, context: TaskContext) -> List[str]:
        """Build Claude Code command with appropriate options - simplified."""
        
        cmd = [str(self.claude_code_path)]
        
        # Add session management if available
        if self.session_persistent and context.session_id:
            session_file = self.session_dir / f"session_{context.session_id}.json"
            if session_file.parent.exists():
                cmd.extend(["--session", str(session_file)])
                self.current_session = context.session_id
        
        # Enhanced prompt with context
        enhanced_prompt = await self._enhance_prompt_with_context(prompt, context)
        
        # Add the prompt directly as argument
        cmd.append(enhanced_prompt)
        
        return cmd
    
    async def _enhance_prompt_with_context(self, prompt: str, context: TaskContext) -> str:
        """Enhance prompt with context information."""
        
        enhanced_parts = []
        
        # Add role context
        if context.agent_role:
            role_prompts = {
                "senior_developer": "You are an expert senior developer with deep technical knowledge.",
                "code_reviewer": "You are performing a thorough code review with focus on quality and security.",
                "project_architect": "You are designing system architecture with scalability in mind.",
                "debug_specialist": "You are debugging code with systematic problem-solving approach."
            }
            
            role_key = context.agent_role.value if hasattr(context.agent_role, 'value') else str(context.agent_role)
            if role_key in role_prompts:
                enhanced_parts.append(role_prompts[role_key])
        
        # Add task type context
        if context.task_type == TaskType.CODE_GENERATION:
            enhanced_parts.append("Focus on writing clean, well-documented, production-ready code.")
        elif context.task_type == TaskType.CODE_REVIEW:
            enhanced_parts.append("Provide detailed code review with specific suggestions for improvement.")
        elif context.task_type == TaskType.DEBATE:
            enhanced_parts.append("Present your perspective clearly and constructively in this collaborative discussion.")
        
        # Add constraints
        if context.constraints:
            constraints_text = "\n".join(f"- {constraint}" for constraint in context.constraints)
            enhanced_parts.append(f"**Constraints to follow:**\n{constraints_text}")
        
        # Add user preferences
        if context.user_preferences:
            prefs_text = "\n".join(f"- {k}: {v}" for k, v in context.user_preferences.items())
            enhanced_parts.append(f"**User preferences:**\n{prefs_text}")
        
        # Combine with original prompt
        if enhanced_parts:
            enhanced_prompt = "\n\n".join(enhanced_parts) + "\n\n" + prompt
        else:
            enhanced_prompt = prompt
        
        return enhanced_prompt
    
    async def _execute_claude_code(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute Claude Code command asynchronously."""
        
        try:
            self.logger.debug(f"Executing Claude Code: {' '.join(cmd[:3])}...")
            
            # Execute asynchronously with simplified approach
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=process.returncode,
                    stdout=stdout.decode('utf-8'),
                    stderr=stderr.decode('utf-8')
                )
                
                if result.returncode != 0:
                    # Log but don't fail - Claude might still provide useful output
                    self.logger.warning(f"Claude Code exit code {result.returncode}: {result.stderr}")
                
                return result
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise AgentError(f"Claude Code timed out after {self.timeout}s")
                        
        except Exception as e:
            raise AgentError(f"Claude Code execution failed: {e}")
    
    def _estimate_token_usage(self, prompt: str, response: str) -> TokenUsage:
        """Estimate token usage for Claude interaction."""
        
        # Rough token estimation (Claude uses different tokenization than GPT)
        # This is approximate - actual usage may vary
        prompt_tokens = len(prompt.split()) * 1.3  # Claude tokens are slightly different
        response_tokens = len(response.split()) * 1.3
        
        return TokenUsage(
            input_tokens=int(prompt_tokens),
            output_tokens=int(response_tokens),
            total_tokens=int(prompt_tokens + response_tokens)
        )
    
    # Enhanced methods for specific Claude Code capabilities
    
    async def execute_code(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Execute code using Claude Code capabilities."""
        
        execution_prompt = f"""Please execute the following {language} code and provide the results:

```{language}
{code}
```

If there are any errors:
1. Identify the issue clearly
2. Provide a corrected version
3. Explain what was wrong
4. Execute the corrected version

Show both the execution output and any debugging information.
Be thorough in your analysis and provide working solutions."""

        execution_context = context.model_copy()
        execution_context.task_type = TaskType.CODE_EXECUTION
        execution_context.metadata["execution_request"] = True
        execution_context.metadata["language"] = language
        
        return await self.generate(execution_prompt, execution_context)
    
    async def create_project(self, project_description: str, context: TaskContext) -> AgentResponse:
        """Create a complete project using Claude Code."""
        
        project_prompt = f"""Create a complete, production-ready project for:

{project_description}

Please:
1. **Create proper project structure** with organized folders
2. **Write all necessary files** with complete implementations
3. **Include configuration files** (requirements.txt, package.json, etc.)
4. **Add comprehensive documentation** (README, API docs, comments)
5. **Implement proper error handling** and logging
6. **Include tests** with good coverage
7. **Add deployment configuration** (Docker, CI/CD if relevant)
8. **Follow best practices** for the chosen technology stack

Provide a working, deployable solution that follows industry standards.
Focus on code quality, maintainability, and scalability."""

        project_context = context.model_copy()
        project_context.task_type = TaskType.CODE_GENERATION
        project_context.agent_role = "project_architect"
        project_context.metadata["project_creation"] = True
        
        return await self.generate(project_prompt, project_context)
    
    async def review_code(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Perform comprehensive code review."""
        
        review_prompt = f"""Please perform a thorough code review of this {language} code:

```{language}
{code}
```

Provide a comprehensive review covering:

**1. Code Quality:**
- Readability and maintainability
- Naming conventions and structure
- Code organization and modularity

**2. Best Practices:**
- Language-specific best practices
- Design patterns usage
- Error handling approach

**3. Security Analysis:**
- Potential security vulnerabilities
- Input validation issues
- Authentication/authorization concerns

**4. Performance Considerations:**
- Efficiency and optimization opportunities
- Resource usage
- Scalability implications

**5. Testing & Documentation:**
- Test coverage assessment
- Documentation quality
- Example usage clarity

**6. Specific Recommendations:**
- Prioritized list of improvements
- Code snippets showing better approaches
- Rationale for each suggestion

Be constructive and specific in your feedback. Provide working examples for improvements."""

        review_context = context.model_copy()
        review_context.task_type = TaskType.CODE_REVIEW
        review_context.agent_role = "code_reviewer"
        review_context.metadata["review_request"] = True
        review_context.metadata["language"] = language
        
        return await self.generate(review_prompt, review_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], context: TaskContext) -> AgentResponse:
        """Propose solution as senior developer using Claude Code."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""As a senior developer, design and implement a complete solution for:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Please provide a comprehensive solution including:

**1. Architecture Design:**
- High-level system design and component breakdown
- Technology stack recommendations with rationale
- Design patterns and architectural principles to apply
- Data flow and integration points

**2. Implementation Plan:**
- Detailed file structure and organization
- Key classes, functions, and interfaces
- Data models and schemas
- API design (if applicable)

**3. Complete Code Implementation:**
- Working, production-ready code files
- Proper error handling and logging
- Input validation and security measures
- Performance optimization considerations

**4. Testing Strategy:**
- Unit tests with good coverage
- Integration tests for key workflows
- Performance and load testing approach
- Security testing considerations

**5. Deployment & Operations:**
- Environment setup and configuration
- Dependencies and requirements
- Deployment scripts and procedures
- Monitoring and logging setup

**6. Documentation:**
- Comprehensive README with setup instructions
- API documentation (if applicable)
- Developer guidelines and contribution guide
- User documentation and examples

Create a professional, enterprise-grade solution that follows industry best practices.
Provide actual, working code rather than pseudocode or placeholders."""

        solution_context = context.model_copy()
        solution_context.task_type = TaskType.CODE_GENERATION
        solution_context.agent_role = "senior_developer"
        solution_context.metadata["solution_request"] = True
        
        return await self.generate(solution_prompt, solution_context)
    
    async def debug_issue(self, code: str, error_description: str, context: TaskContext) -> AgentResponse:
        """Debug code issues using Claude Code."""
        
        debug_prompt = f"""Help debug this issue:

**Error Description:**
{error_description}

**Code:**
```
{code}
```

Please:
1. **Identify the Problem:** Analyze the code and error description
2. **Root Cause Analysis:** Explain why this issue occurred
3. **Provide Fixed Code:** Show the corrected version
4. **Testing:** Provide test cases to verify the fix
5. **Prevention:** Suggest how to avoid similar issues

Be systematic in your debugging approach and provide working solutions."""

        debug_context = context.model_copy()
        debug_context.task_type = TaskType.CODE_REVIEW
        debug_context.agent_role = "debug_specialist"
        debug_context.metadata["debug_request"] = True
        
        return await self.generate(debug_prompt, debug_context)
    
    async def shutdown(self) -> None:
        """Shutdown Claude Code agent and cleanup sessions."""
        self.logger.info("Shutting down Claude Code agent...")
        
        # Cleanup session files if needed
        if self.session_persistent and self.current_session:
            try:
                session_file = self.session_dir / f"session_{self.current_session}.json"
                if session_file.exists():
                    self.logger.info(f"Preserving session file: {session_file}")
            except Exception as e:
                self.logger.error(f"Error handling session cleanup: {e}")
        
        await super().shutdown()
