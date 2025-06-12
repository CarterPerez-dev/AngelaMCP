"""
OpenAI Agent for AngelaMCP.

This agent integrates with OpenAI's API for code review, analysis,
and optimization. I'm implementing a production-grade agent with
proper rate limiting, error handling, and cost tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.agents.base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType, 
    AgentCapabilities, TokenUsage, track_performance
)
from src.utils import get_logger
from src.utils import AgentError
from config import settings


class OpenAIAgent(BaseAgent):
    """
    OpenAI agent for code review and analysis.
    
    This agent acts as a code reviewer and analyst, providing detailed
    feedback on code quality, security, and optimization opportunities.
    """
    
    def __init__(self):
        # Define OpenAI capabilities
        capabilities = AgentCapabilities(
            can_execute_code=False,
            can_read_files=False,
            can_write_files=False,
            can_browse_web=False,
            can_use_tools=True,
            supported_languages=[
                "python", "javascript", "typescript", "java", "cpp", "c", 
                "go", "rust", "ruby", "php", "swift", "kotlin", "scala",
                "html", "css", "sql", "bash", "r", "matlab"
            ],
            supported_formats=[
                "text", "markdown", "code", "json", "yaml"
            ],
            max_context_length=128000,  # GPT-4 context window
            supports_streaming=True,
            supports_function_calling=True
        )
        
        super().__init__(AgentType.OPENAI, "openai", capabilities)
        
        # OpenAI configuration
        self.api_key = settings.openai_api_key.get_secret_value()
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.top_p = settings.openai_top_p
        self.frequency_penalty = settings.openai_frequency_penalty
        self.presence_penalty = settings.openai_presence_penalty
        self.timeout = settings.openai_timeout
        
        # Rate limiting
        self.rate_limit = settings.openai_rate_limit
        self.max_retries = settings.openai_max_retries
        self.retry_delay = settings.openai_retry_delay
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # Verify API key
        self._verify_api_key()
    
    def _verify_api_key(self) -> None:
        """Verify OpenAI API key is valid."""
        try:
            if not self.api_key or not self.api_key.startswith('sk-'):
                raise AgentError("Invalid OpenAI API key format")
            
            self.logger.info("✅ OpenAI API key verified")
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI API key verification failed: {e}")
            raise AgentError(f"OpenAI setup failed: {e}")
    
    @track_performance
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate response using OpenAI API.
        
        This method handles the core interaction with OpenAI's API,
        including rate limiting and error handling.
        """
        start_time = time.time()
        
        try:
            self.agent_logger.log_request(f"Generating response for {context.task_type.value} task")
            
            # Check rate limit
            await self._wait_for_rate_limit(self.rate_limit)
            
            # Build messages for chat completion
            messages = await self._build_messages(prompt, context)
            
            # Make API call with retries
            completion = await self._make_api_call(messages, context)
            
            # Extract response content
            response_content = completion.choices[0].message.content or ""
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create token usage
            token_usage = TokenUsage(
                input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                output_tokens=completion.usage.completion_tokens if completion.usage else 0,
                total_tokens=completion.usage.total_tokens if completion.usage else 0
            )
            
            # Create response
            response = AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content=response_content,
                success=True,
                confidence=0.85,  # OpenAI is reliable but not as high as Claude for code
                execution_time_ms=execution_time,
                token_usage=token_usage,
                metadata={
                    "model": self.model,
                    "task_type": context.task_type.value,
                    "agent_role": context.agent_role.value if context.agent_role else None,
                    "finish_reason": completion.choices[0].finish_reason,
                    "temperature": self.temperature
                }
            )
            
            self.agent_logger.log_response(f"Generated {len(response_content)} characters")
            return response
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"OpenAI generation failed: {e}"
            
            self.agent_logger.log_error(error_msg, e)
            
            return AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content="",
                success=False,
                execution_time_ms=execution_time,
                error=error_msg
            )
    
    async def _build_messages(self, prompt: str, context: TaskContext) -> List[Dict[str, str]]:
        """Build messages for OpenAI chat completion."""
        
        messages = []
        
        # System message based on context
        system_message = await self._get_system_message(context)
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # User message with enhanced prompt
        enhanced_prompt = await self._enhance_prompt_for_openai(prompt, context)
        messages.append({"role": "user", "content": enhanced_prompt})
        
        return messages
    
    async def _get_system_message(self, context: TaskContext) -> str:
        """Get system message based on task context."""
        
        base_message = "You are an expert AI assistant specializing in code review, analysis, and optimization."
        
        role_messages = {
            "reviewer": "You excel at thorough code reviews, identifying bugs, security issues, and optimization opportunities.",
            "analyst": "You provide deep technical analysis and insights on code architecture and design patterns.",
            "critic": "You provide constructive criticism to improve code quality and adherence to best practices.",
            "specialist": "You have deep expertise in multiple programming languages and frameworks."
        }
        
        task_messages = {
            TaskType.CODE_REVIEW: "Focus on code quality, security, performance, and maintainability in your review.",
            TaskType.ANALYSIS: "Provide comprehensive technical analysis with specific recommendations.",
            TaskType.CRITIQUE: "Give constructive feedback aimed at improving the solution.",
            TaskType.DEBATE: "Present well-reasoned arguments backed by technical expertise.",
            TaskType.RESEARCH: "Conduct thorough research and provide evidence-based recommendations."
        }
        
        message_parts = [base_message]
        
        # Add role-specific guidance
        if context.agent_role:
            role_key = context.agent_role.value if hasattr(context.agent_role, 'value') else str(context.agent_role)
            if role_key in role_messages:
                message_parts.append(role_messages[role_key])
        
        # Add task-specific guidance
        if context.task_type in task_messages:
            message_parts.append(task_messages[context.task_type])
        
        # Add constraints and preferences
        if context.constraints:
            constraints_text = "\n".join(f"- {constraint}" for constraint in context.constraints)
            message_parts.append(f"Important constraints to follow:\n{constraints_text}")
        
        return " ".join(message_parts)
    
    async def _enhance_prompt_for_openai(self, prompt: str, context: TaskContext) -> str:
        """Enhance prompt specifically for OpenAI's strengths."""
        
        enhanced_parts = []
        
        # Add context for better OpenAI performance
        if context.task_type == TaskType.CODE_REVIEW:
            enhanced_parts.append("""Please provide a comprehensive code review with the following structure:

1. **Overall Assessment**: High-level evaluation of the code quality
2. **Security Analysis**: Identify potential security vulnerabilities
3. **Performance Review**: Assess efficiency and optimization opportunities
4. **Best Practices**: Check adherence to coding standards and best practices
5. **Specific Issues**: List concrete problems with line references if possible
6. **Recommendations**: Prioritized suggestions for improvement

Be thorough, specific, and constructive in your feedback.""")
        
        elif context.task_type == TaskType.ANALYSIS:
            enhanced_parts.append("""Provide a detailed technical analysis including:

1. **Architecture Assessment**: Evaluate the overall design and structure
2. **Technology Choices**: Assess the appropriateness of technologies used
3. **Scalability Considerations**: Identify potential bottlenecks and scaling issues
4. **Maintainability Review**: Evaluate code organization and documentation
5. **Risk Assessment**: Identify technical and business risks
6. **Strategic Recommendations**: Suggest improvements and future directions

Support your analysis with specific examples and reasoning.""")
        
        elif context.task_type == TaskType.DEBATE:
            enhanced_parts.append("""Present your position in this collaborative debate:

1. **Clear Position**: State your stance clearly and concisely
2. **Supporting Evidence**: Provide technical reasoning and examples
3. **Consideration of Alternatives**: Acknowledge other viewpoints
4. **Specific Benefits**: Explain why your approach is superior
5. **Potential Drawbacks**: Honestly assess limitations
6. **Implementation Details**: Provide concrete next steps

Be persuasive but fair, and focus on technical merit.""")
        
        # Combine with original prompt
        if enhanced_parts:
            enhanced_prompt = "\n\n".join(enhanced_parts) + "\n\n" + prompt
        else:
            enhanced_prompt = prompt
        
        return enhanced_prompt
    
    async def _make_api_call(self, messages: List[Dict[str, str]], context: TaskContext) -> ChatCompletion:
        """Make OpenAI API call with proper error handling."""
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                timeout=self.timeout
            )
            
            return completion
            
        except openai.RateLimitError as e:
            self.logger.warning(f"OpenAI rate limit hit: {e}")
            await asyncio.sleep(self.retry_delay)
            raise AgentError(f"Rate limit exceeded: {e}")
            
        except openai.AuthenticationError as e:
            raise AgentError(f"Authentication failed: {e}")
            
        except openai.BadRequestError as e:
            raise AgentError(f"Bad request: {e}")
            
        except openai.APIError as e:
            raise AgentError(f"API error: {e}")
            
        except Exception as e:
            raise AgentError(f"Unexpected error: {e}")
    
    # Enhanced methods for OpenAI's strengths
    
    async def review_code(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Perform detailed code review using OpenAI's analytical capabilities."""
        
        review_prompt = f"""Please conduct a thorough code review of this {language} code:

```{language}
{code}
```

As an expert code reviewer, analyze this code comprehensively and provide detailed feedback."""
        
        review_context = context.model_copy()
        review_context.task_type = TaskType.CODE_REVIEW
        review_context.agent_role = "reviewer"
        review_context.metadata["language"] = language
        
        return await self.generate(review_prompt, review_context)
    
    async def analyze_architecture(self, description: str, context: TaskContext) -> AgentResponse:
        """Analyze system architecture and provide recommendations."""
        
        analysis_prompt = f"""Analyze the following system architecture description:

{description}

Provide a comprehensive technical analysis focusing on:
- Architectural patterns and design quality
- Scalability and performance implications
- Security considerations
- Technology stack evaluation
- Potential risks and mitigation strategies
- Recommendations for improvement"""
        
        analysis_context = context.model_copy()
        analysis_context.task_type = TaskType.ANALYSIS
        analysis_context.agent_role = "analyst"
        
        return await self.generate(analysis_prompt, analysis_context)
    
    async def critique_solution(self, solution: str, context: TaskContext) -> AgentResponse:
        """Provide constructive critique of a proposed solution."""
        
        critique_prompt = f"""Please provide a constructive critique of this solution:

{solution}

Focus on:
1. **Strengths**: What works well in this approach
2. **Weaknesses**: Areas that could be improved
3. **Alternatives**: Other approaches to consider
4. **Risks**: Potential issues or concerns
5. **Specific Suggestions**: Concrete ways to enhance the solution

Be balanced and constructive in your feedback."""
        
        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CRITIQUE
        critique_context.agent_role = "critic"
        
        return await self.generate(critique_prompt, critique_context)
    
    async def optimize_performance(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Analyze code for performance optimization opportunities."""
        
        optimization_prompt = f"""Analyze this {language} code for performance optimization:

```{language}
{code}
```

Provide:
1. **Performance Bottlenecks**: Identify slow or inefficient parts
2. **Optimization Opportunities**: Specific improvements to make
3. **Algorithmic Improvements**: Better algorithms or data structures
4. **Best Practices**: Language-specific optimizations
5. **Benchmarking**: How to measure improvements
6. **Optimized Code**: Provide improved versions of critical sections

Focus on measurable performance improvements."""
        
        optimization_context = context.model_copy()
        optimization_context.task_type = TaskType.ANALYSIS
        optimization_context.agent_role = "specialist"
        optimization_context.metadata["optimization_focus"] = True
        optimization_context.metadata["language"] = language
        
        return await self.generate(optimization_prompt, optimization_context)
    
    async def security_review(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Perform security-focused code review."""
        
        security_prompt = f"""Conduct a security review of this {language} code:

```{language}
{code}
```

Focus on:
1. **Vulnerability Assessment**: Identify potential security flaws
2. **Input Validation**: Check for proper input sanitization
3. **Authentication/Authorization**: Review access controls
4. **Data Protection**: Assess sensitive data handling
5. **Common Attacks**: Check for SQL injection, XSS, CSRF, etc.
6. **Security Best Practices**: Recommend security improvements
7. **Compliance**: Note any regulatory concerns

Provide specific recommendations for each issue found."""
        
        security_context = context.model_copy()
        security_context.task_type = TaskType.CODE_REVIEW
        security_context.agent_role = "specialist"
        security_context.metadata["security_focus"] = True
        security_context.metadata["language"] = language
        
        return await self.generate(security_prompt, security_context)
    
    async def shutdown(self) -> None:
        """Shutdown OpenAI agent and cleanup resources."""
        self.logger.info("Shutting down OpenAI agent...")
        
        try:
            await self.client.close()
        except Exception as e:
            self.logger.error(f"Error closing OpenAI client: {e}")
        
        await super().shutdown()
