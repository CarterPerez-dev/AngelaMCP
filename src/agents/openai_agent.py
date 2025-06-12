"""
OpenAI agent implementation for AngelaMCP.

This module implements the OpenAI agent wrapper using the o3-mini model with
comprehensive error handling, retry logic, and cost tracking. I'm implementing
a production-grade client that supports both review and research roles.
"""

import asyncio
import time
import tiktoken
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai._exceptions import (
    OpenAIError, RateLimitError, APITimeoutError, 
    APIConnectionError, AuthenticationError
)

from .base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType,
    AgentCapability, AgentError, AgentRateLimitError, 
    AgentTimeoutError, AgentAuthenticationError
)
from src.utils.logger import get_logger, log_agent_interaction, AsyncPerformanceLogger
from config.settings import settings

logger = get_logger("agents.openai")


class OpenAIError(AgentError):
    """Specific error for OpenAI operations."""
    pass


class OpenAIAgent(BaseAgent):
    """
    OpenAI agent implementation using o3-mini model.
    
    I'm implementing a comprehensive OpenAI client that provides review and research
    capabilities with proper cost tracking, retry logic, and error handling.
    """
    
    def __init__(self, name: str = "openai", api_key: Optional[str] = None):
        super().__init__(AgentType.OPENAI, name, settings)
        
        # OpenAI configuration
        self.api_key = api_key or settings.openai_api_key.get_secret_value()
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.top_p = settings.openai_top_p
        self.frequency_penalty = settings.openai_frequency_penalty
        self.presence_penalty = settings.openai_presence_penalty
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        # Token encoding for cost calculation
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost tracking (per 1K tokens)
        self.input_cost_per_1k = settings.openai_input_cost
        self.output_cost_per_1k = settings.openai_output_cost
        
        # Define capabilities
        self._setup_capabilities()
        
        self.logger.info(f"OpenAI agent initialized with model: {self.model}")
    
    def _setup_capabilities(self) -> None:
        """Define OpenAI agent capabilities."""
        self._capabilities = [
            AgentCapability(
                name="text_generation",
                description="Generate high-quality text responses",
                supported_formats=["text", "markdown", "json"],
                limitations=["Context length limits", "No real-time information"],
                cost_per_request=0.006  # Estimated average cost
            ),
            AgentCapability(
                name="code_review",
                description="Review and critique code for quality and best practices",
                supported_formats=["python", "javascript", "typescript", "java", "cpp"],
                limitations=["Cannot execute code", "Limited to static analysis"],
                cost_per_request=0.008
            ),
            AgentCapability(
                name="research_analysis",
                description="Analyze and synthesize information on technical topics",
                supported_formats=["text", "structured_data"],
                limitations=["No access to real-time data", "Knowledge cutoff limitations"],
                cost_per_request=0.012
            ),
            AgentCapability(
                name="problem_solving",
                description="Break down complex problems and propose solutions",
                supported_formats=["text", "structured_analysis"],
                limitations=["Cannot verify solutions practically"],
                cost_per_request=0.010
            ),
            AgentCapability(
                name="structured_output",
                description="Generate structured JSON responses",
                supported_formats=["json", "yaml"],
                limitations=["Schema complexity limits"],
                cost_per_request=0.007
            )
        ]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # Fallback to rough estimation
            self.logger.warning(f"Token counting failed: {e}")
            return len(text.split()) * 1.3  # Rough approximation
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def _build_messages(self, prompt: str, context: TaskContext, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Build messages array for OpenAI chat completion."""
        messages = []
        
        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif context.agent_role.value == "reviewer":
            messages.append({
                "role": "system", 
                "content": "You are an expert code reviewer and technical analyst. Provide thorough, constructive feedback focusing on code quality, best practices, potential issues, and improvement suggestions. Be specific and actionable in your recommendations."
            })
        elif context.agent_role.value == "researcher":
            messages.append({
                "role": "system",
                "content": "You are a technical researcher and analyst. Provide comprehensive analysis, gather relevant information, and synthesize findings into clear, actionable insights. Focus on accuracy, thoroughness, and practical applicability."
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant specializing in software development and technical problem-solving. Provide clear, accurate, and practical responses."
            })
        
        # User message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def _make_completion_request(self, messages: List[Dict[str, str]], context: TaskContext) -> ChatCompletion:
        """Make a chat completion request to OpenAI."""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=context.max_tokens or self.max_tokens,
                temperature=context.temperature or self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                timeout=context.timeout_seconds or self.timeout
            )
            
            return completion
            
        except RateLimitError as e:
            raise AgentRateLimitError(
                f"OpenAI rate limit exceeded: {e}",
                agent_type=self.agent_type.value,
                error_code="RATE_LIMIT_EXCEEDED"
            ) from e
        
        except AuthenticationError as e:
            raise AgentAuthenticationError(
                f"OpenAI authentication failed: {e}",
                agent_type=self.agent_type.value,
                error_code="AUTHENTICATION_FAILED"
            ) from e
        
        except APITimeoutError as e:
            raise AgentTimeoutError(
                f"OpenAI request timed out: {e}",
                agent_type=self.agent_type.value,
                error_code="REQUEST_TIMEOUT"
            ) from e
        
        except (APIConnectionError, OpenAIError) as e:
            raise OpenAIError(
                f"OpenAI API error: {e}",
                agent_type=self.agent_type.value,
                error_code="API_ERROR"
            ) from e
    
    def _parse_completion_response(self, completion: ChatCompletion, input_tokens: int, execution_time: float) -> AgentResponse:
        """Parse OpenAI completion response into standardized format."""
        choice = completion.choices[0]
        content = choice.message.content or ""
        
        # Calculate tokens and cost
        output_tokens = completion.usage.completion_tokens if completion.usage else self._count_tokens(content)
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)
        
        return AgentResponse(
            success=True,
            content=content,
            agent_type=self.agent_type.value,
            execution_time_ms=execution_time * 1000,
            tokens_used=total_tokens,
            cost_usd=cost,
            metadata={
                "model": completion.model,
                "finish_reason": choice.finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "openai_id": completion.id,
                "system_fingerprint": completion.system_fingerprint,
                "usage": completion.usage.model_dump() if completion.usage else None
            }
        )
    
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate a response using OpenAI."""
        start_time = time.time()
        
        async with AsyncPerformanceLogger(self.logger, "openai_generate", task_id=context.task_id):
            try:
                # Build messages and count input tokens
                messages = self._build_messages(prompt, context)
                input_text = "\n".join(msg["content"] for msg in messages)
                input_tokens = self._count_tokens(input_text)
                
                # Make API request with retry logic
                completion = await self.execute_with_retry(
                    self._make_completion_request, messages, context
                )
                
                execution_time = time.time() - start_time
                response = self._parse_completion_response(completion, input_tokens, execution_time)
                
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
                        "model": self.model,
                        "tokens_used": response.tokens_used,
                        "cost_usd": response.cost_usd,
                        "execution_time_ms": response.execution_time_ms
                    }
                )
                
                return response
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"OpenAI generation failed: {e}")
                
                return AgentResponse(
                    success=False,
                    content="",
                    agent_type=self.agent_type.value,
                    execution_time_ms=execution_time * 1000,
                    error_message=str(e),
                    metadata={"task_id": context.task_id, "model": self.model}
                )
    
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide detailed critique using OpenAI."""
        critique_prompt = f"""Please provide a thorough critique of the following solution for the task: "{original_task}"

Solution to review:
{content}

Provide a detailed analysis including:

**Strengths:**
- What aspects of the solution work well
- Good practices demonstrated
- Effective approaches used

**Weaknesses and Issues:**
- Technical problems or bugs
- Logic errors or edge cases missed
- Performance concerns
- Security vulnerabilities
- Code quality issues

**Specific Improvements:**
- Concrete suggestions for fixing issues
- Alternative approaches to consider
- Best practices that should be applied
- Code refactoring recommendations

**Overall Assessment:**
- Summary of solution quality
- Readiness for production use
- Priority of recommended changes

Focus on being constructive, specific, and actionable in your feedback."""

        # Update context for critique task
        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        critique_context.agent_role = critique_context.agent_role or "reviewer"
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], 
                             context: TaskContext) -> AgentResponse:
        """Propose a solution using OpenAI's analytical capabilities."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""Analyze the following task and propose a comprehensive solution:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Please provide a structured solution including:

**1. Problem Analysis:**
- Break down the key requirements
- Identify potential challenges
- Consider edge cases and corner scenarios

**2. Proposed Approach:**
- High-level solution strategy
- Technology choices and rationale
- Architecture considerations

**3. Implementation Plan:**
- Step-by-step implementation approach
- Key components and their responsibilities
- Integration points and dependencies

**4. Risk Assessment:**
- Potential risks and mitigation strategies
- Alternative approaches if the main solution fails
- Scalability and maintenance considerations

**5. Success Criteria:**
- How to measure if the solution works
- Testing approach and validation methods
- Performance benchmarks

Focus on creating a practical, well-reasoned solution that addresses all requirements while considering real-world constraints."""

        # Update context for solution proposal
        solution_context = context.model_copy()
        solution_context.task_type = TaskType.ANALYSIS
        
        return await self.generate(solution_prompt, solution_context)
    
    async def research_topic(self, topic: str, focus_areas: List[str], context: TaskContext) -> AgentResponse:
        """Research a technical topic using OpenAI's knowledge base."""
        focus_text = "\n".join(f"- {area}" for area in focus_areas) if focus_areas else "General overview"
        
        research_prompt = f"""Conduct comprehensive research on the following topic:

**Topic:** {topic}

**Focus Areas:**
{focus_text}

Please provide a thorough research analysis including:

**1. Overview:**
- Definition and key concepts
- Current state and relevance
- Important context and background

**2. Technical Details:**
- Core technologies and methodologies
- Implementation approaches
- Standards and best practices

**3. Comparative Analysis:**
- Alternative solutions or approaches
- Pros and cons of different methods
- Use case scenarios for each approach

**4. Current Trends:**
- Recent developments and innovations
- Industry adoption patterns
- Future outlook and predictions

**5. Practical Applications:**
- Real-world use cases
- Implementation examples
- Common pitfalls and how to avoid them

**6. Resources and References:**
- Key tools and frameworks
- Recommended learning resources
- Important standards or specifications

Provide accurate, up-to-date information with practical insights for implementation."""

        # Update context for research task
        research_context = context.model_copy()
        research_context.task_type = TaskType.RESEARCH
        research_context.agent_role = "researcher"
        
        return await self.generate(research_prompt, research_context)
    
    async def analyze_code_quality(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Analyze code quality using OpenAI's expertise."""
        analysis_prompt = f"""Perform a comprehensive code quality analysis of the following {language} code:

```{language}
{code}
```

Provide detailed analysis covering:

**1. Code Quality Metrics:**
- Readability and maintainability
- Complexity assessment
- Adherence to coding standards
- Documentation quality

**2. Best Practices Review:**
- Language-specific best practices
- Design patterns usage
- Error handling approaches
- Performance considerations

**3. Security Analysis:**
- Potential security vulnerabilities
- Input validation issues
- Data handling concerns
- Access control considerations

**4. Optimization Opportunities:**
- Performance improvements
- Memory usage optimization
- Algorithm efficiency
- Resource management

**5. Refactoring Suggestions:**
- Code structure improvements
- Function/class organization
- Naming conventions
- Code duplication elimination

**6. Testing Recommendations:**
- Test coverage assessment
- Missing test scenarios
- Testing strategy suggestions
- Mock and stub opportunities

Provide specific, actionable recommendations with code examples where helpful."""

        # Update context for code analysis
        analysis_context = context.model_copy()
        analysis_context.task_type = TaskType.CODE_REVIEW
        analysis_context.agent_role = "reviewer"
        
        return await self.generate(analysis_prompt, analysis_context)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check specific to OpenAI."""
        try:
            # Test basic functionality
            test_context = TaskContext(
                task_id="health_check",
                task_type=TaskType.CUSTOM,
                timeout_seconds=30
            )
            
            start_time = time.time()
            response = await self.generate("Respond with 'OpenAI health check successful'", test_context)
            execution_time = time.time() - start_time
            
            health_info = {
                "status": "healthy" if response.success else "unhealthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "model": self.model,
                "api_key_configured": bool(self.api_key),
                "response_time_ms": execution_time * 1000,
                "test_response_success": response.success,
                "test_cost_usd": response.cost_usd,
                "test_tokens_used": response.tokens_used,
                "capabilities_count": len(self._capabilities),
                "performance_metrics": self.performance_metrics
            }
            
            if not response.success:
                health_info["error"] = response.error_message
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "error": str(e),
                "model": self.model,
                "api_key_configured": bool(self.api_key),
                "performance_metrics": self.performance_metrics
            }
    
    async def shutdown(self) -> None:
        """Shutdown OpenAI agent and cleanup resources."""
        self.logger.info("Shutting down OpenAI agent")
        
        # Close the HTTP client
        if hasattr(self.client, 'close'):
            await self.client.close()
        
        # Call parent shutdown
        await super().shutdown()
