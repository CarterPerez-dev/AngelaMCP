"""
OpenAI Agent implementation for AngelaMCP.

This agent specializes in code review, analysis, and quality assessment.
I'm implementing this as the "code reviewer" with focus on best practices and security.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.agents.base import BaseAgent, AgentType, AgentResponse, TaskContext, TaskType
from src.utils.logger import get_logger
from src.utils.exceptions import AgentError
from config.settings import settings


class OpenAIAgent(BaseAgent):
    """
    OpenAI agent specializing in code review and analysis.
    
    Capabilities:
    - Code quality assessment
    - Security analysis  
    - Performance optimization
    - Best practices review
    - Detailed technical analysis
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.OPENAI,
            name="OpenAI Code Reviewer",
            capabilities=[
                "code_review",
                "security_analysis",
                "performance_optimization", 
                "best_practices",
                "technical_analysis",
                "quality_assessment"
            ]
        )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.max_retries = settings.openai_max_retries
        self.retry_delay = settings.openai_retry_delay
        
        self.logger.info(f"Initialized OpenAI agent with model: {self.model}")
    
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate response using OpenAI with retry logic."""
        start_time = time.time()
        
        try:
            # Build messages based on context
            messages = self._build_messages(prompt, context)
            
            # Generate with retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self._get_temperature(context),
                        top_p=settings.openai_top_p,
                        frequency_penalty=settings.openai_frequency_penalty,
                        presence_penalty=settings.openai_presence_penalty,
                        timeout=settings.openai_timeout
                    )
                    
                    # Extract response content
                    content = response.choices[0].message.content
                    if not content:
                        raise AgentError("Empty response from OpenAI")
                    
                    execution_time = time.time() - start_time
                    
                    return AgentResponse(
                        agent_type=self.agent_type,
                        content=content,
                        confidence=self._calculate_confidence(response),
                        execution_time_ms=execution_time * 1000,
                        token_usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        },
                        metadata={
                            "model": self.model,
                            "finish_reason": response.choices[0].finish_reason,
                            "attempt": attempt + 1,
                            "context": context.task_type.value
                        }
                    )
                    
                except Exception as e:
                    if attempt < self.max_retries:
                        self.logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise AgentError(f"OpenAI API failed after {self.max_retries + 1} attempts: {e}")
        
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}", exc_info=True)
            raise AgentError(f"OpenAI generation error: {e}")
    
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide detailed code review and critique."""
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

**Security Analysis:**
- Potential security vulnerabilities
- Input validation concerns
- Authentication and authorization issues
- Data handling security

**Performance Assessment:**
- Performance bottlenecks
- Optimization opportunities
- Scalability considerations
- Resource usage analysis

**Overall Assessment:**
- Summary of solution quality
- Readiness for production use
- Priority of recommended changes
- Risk assessment

Focus on being constructive, specific, and actionable in your feedback."""

        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        critique_context.agent_role = "reviewer"
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], context: TaskContext) -> AgentResponse:
        """Propose a solution with focus on best practices and quality."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""Analyze the following task and propose a comprehensive solution with emphasis on code quality, security, and best practices:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Please provide a structured solution including:

**1. Problem Analysis:**
- Break down the key requirements
- Identify potential challenges
- Consider edge cases and corner scenarios
- Security implications

**2. Proposed Approach:**
- High-level solution strategy
- Technology choices and rationale
- Architecture considerations
- Security-first design principles

**3. Implementation Plan:**
- Step-by-step implementation approach
- Key components and their responsibilities
- Error handling strategy
- Testing approach

**4. Code Quality Considerations:**
- Design patterns to apply
- Code organization principles
- Documentation standards
- Maintainability factors

**5. Security Assessment:**
- Security requirements and considerations
- Potential vulnerabilities to address
- Authentication and authorization needs
- Data protection measures

**6. Performance Optimization:**
- Performance requirements
- Optimization strategies
- Scalability considerations
- Resource management

**7. Risk Assessment:**
- Technical risks and mitigation strategies
- Alternative approaches if main solution fails
- Monitoring and alerting needs

**8. Testing Strategy:**
- Unit testing approach
- Integration testing plan
- Security testing requirements
- Performance testing needs

Focus on creating a practical, well-reasoned solution that prioritizes security, performance, and maintainability."""

        solution_context = context.model_copy()
        solution_context.task_type = TaskType.ANALYSIS
        
        return await self.generate(solution_prompt, solution_context)
    
    async def research_topic(self, topic: str, focus_areas: List[str], context: TaskContext) -> AgentResponse:
        """Research a technical topic with focus on best practices."""
        focus_text = "\n".join(f"- {area}" for area in focus_areas) if focus_areas else "General overview"
        
        research_prompt = f"""Conduct comprehensive research on the following topic with emphasis on practical implementation and best practices:

**Topic:** {topic}

**Focus Areas:**
{focus_text}

Please provide a thorough research analysis including:

**1. Overview:**
- Definition and key concepts
- Current state and relevance
- Important context and background
- Industry standards

**2. Technical Details:**
- Core technologies and methodologies
- Implementation approaches
- Standards and best practices
- Common pitfalls and how to avoid them

**3. Comparative Analysis:**
- Alternative solutions or approaches
- Pros and cons of different methods
- Use case scenarios for each approach
- Performance and security implications

**4. Best Practices:**
- Industry-standard approaches
- Security considerations
- Performance optimization
- Maintainability guidelines

**5. Implementation Guidance:**
- Step-by-step implementation approaches
- Common integration patterns
- Testing strategies
- Monitoring and debugging

**6. Security Considerations:**
- Security best practices
- Common vulnerabilities
- Protection strategies
- Compliance requirements

**7. Performance Optimization:**
- Performance best practices
- Optimization techniques
- Scalability considerations
- Resource management

**8. Resources and Tools:**
- Recommended tools and frameworks
- Learning resources
- Standards and specifications
- Community resources

Provide accurate, up-to-date information with practical insights for secure and efficient implementation."""

        research_context = context.model_copy()
        research_context.task_type = TaskType.RESEARCH
        research_context.agent_role = "researcher"
        
        return await self.generate(research_prompt, research_context)
    
    async def analyze_security(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Perform detailed security analysis."""
        security_prompt = f"""Perform a comprehensive security analysis of the following {language} code:

```{language}
{code}
```

Please provide detailed security assessment covering:

**1. Vulnerability Assessment:**
- Identify potential security vulnerabilities
- OWASP Top 10 considerations
- Language-specific security issues
- Configuration security problems

**2. Input Validation:**
- Input validation weaknesses
- Injection attack vectors (SQL, XSS, etc.)
- Parameter tampering risks
- Data sanitization issues

**3. Authentication & Authorization:**
- Authentication mechanism review
- Authorization logic assessment
- Session management security
- Access control implementation

**4. Data Protection:**
- Data encryption at rest and in transit
- Sensitive data handling
- Privacy considerations
- Data leakage risks

**5. Error Handling:**
- Information disclosure through errors
- Exception handling security
- Logging security considerations
- Debug information exposure

**6. Infrastructure Security:**
- Deployment security considerations
- Configuration management security
- Dependency security assessment
- Environment security

**7. Remediation Recommendations:**
- Specific fixes for identified issues
- Security implementation best practices
- Secure coding guidelines
- Testing recommendations

**8. Risk Assessment:**
- Risk level classification (Critical/High/Medium/Low)
- Impact analysis
- Exploitability assessment
- Mitigation priority

Focus on providing actionable security improvements and specific remediation steps."""

        security_context = context.model_copy()
        security_context.task_type = TaskType.CODE_REVIEW
        security_context.metadata["security_analysis"] = True
        
        return await self.generate(security_prompt, security_context)
    
    async def optimize_performance(self, code: str, language: str, context: TaskContext) -> AgentResponse:
        """Analyze and suggest performance optimizations."""
        optimization_prompt = f"""Analyze the following {language} code for performance optimization opportunities:

```{language}
{code}
```

Please provide comprehensive performance analysis including:

**1. Performance Analysis:**
- Identify performance bottlenecks
- Algorithmic complexity assessment
- Resource usage analysis
- Memory consumption patterns

**2. Optimization Opportunities:**
- Algorithm optimization suggestions
- Data structure improvements
- Caching strategies
- Database query optimization

**3. Scalability Assessment:**
- Horizontal scaling considerations
- Vertical scaling opportunities
- Load handling capabilities
- Resource contention points

**4. Memory Optimization:**
- Memory leak detection
- Memory usage optimization
- Garbage collection considerations
- Resource cleanup improvements

**5. I/O Optimization:**
- File I/O improvements
- Network I/O optimization
- Database access optimization
- Caching implementation

**6. Concurrency Improvements:**
- Parallel processing opportunities
- Async/await optimization
- Thread safety considerations
- Lock contention reduction

**7. Implementation Recommendations:**
- Specific code improvements
- Performance monitoring suggestions
- Benchmarking strategies
- Testing approaches

**8. Trade-off Analysis:**
- Performance vs readability
- Performance vs maintainability
- Memory vs CPU trade-offs
- Optimization priority recommendations

Provide specific, actionable performance improvements with measurable impact estimates."""

        optimization_context = context.model_copy()
        optimization_context.task_type = TaskType.ANALYSIS
        optimization_context.metadata["performance_analysis"] = True
        
        return await self.generate(optimization_prompt, optimization_context)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI agent health and connectivity."""
        try:
            start_time = time.time()
            
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Respond with 'OK' for health check."}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model,
                "response_time": response_time,
                "token_usage": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "capabilities": self.capabilities,
                "last_check": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model,
                "last_check": time.time()
            }
    
    def _build_messages(self, prompt: str, context: TaskContext) -> List[Dict[str, str]]:
        """Build message list based on context."""
        system_message = self._get_system_message(context)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _get_system_message(self, context: TaskContext) -> Optional[str]:
        """Get system message based on task context."""
        system_messages = {
            TaskType.CODE_REVIEW: "You are an expert code reviewer specializing in security, performance, and best practices. Provide thorough, constructive feedback with specific, actionable recommendations.",
            TaskType.ANALYSIS: "You are a senior technical analyst. Provide comprehensive analysis with focus on security, performance, and architectural best practices.",
            TaskType.RESEARCH: "You are a technical researcher specializing in best practices, security, and performance optimization. Provide thorough, practical research with actionable insights.",
            TaskType.GENERAL: "You are a technical expert focused on code quality, security, and best practices. Provide detailed, practical guidance."
        }
        
        return system_messages.get(context.task_type)
    
    def _get_temperature(self, context: TaskContext) -> float:
        """Get temperature based on task context."""
        # Lower temperature for code review and analysis
        if context.task_type in [TaskType.CODE_REVIEW, TaskType.ANALYSIS]:
            return 0.3
        # Higher temperature for creative tasks
        elif context.task_type == TaskType.CREATIVE:
            return 0.8
        # Default temperature
        else:
            return self.temperature
    
    def _calculate_confidence(self, response: ChatCompletion) -> float:
        """Calculate confidence score based on response quality."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on finish reason
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "stop":
            confidence += 0.1
        elif finish_reason == "length":
            confidence -= 0.1
        
        # Adjust based on response length (longer usually indicates more thorough analysis)
        content_length = len(response.choices[0].message.content or "")
        if content_length > 1000:
            confidence += 0.1
        elif content_length < 100:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
