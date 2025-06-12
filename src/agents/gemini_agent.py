"""
Gemini agent implementation for AngelaMCP.

This module implements the Gemini agent wrapper using the 2.5-pro model with
comprehensive safety settings, error handling, and cost tracking. I'm implementing
a production-grade client that specializes in research and creative problem-solving.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types
from google.genai._exceptions import (
    ClientError, ServerError, ValidationError,
    QuotaExhaustedError, PermissionDeniedError
)

from .base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType,
    AgentCapability, AgentError, AgentRateLimitError,
    AgentTimeoutError, AgentAuthenticationError
)
from src.utils.logger import get_logger, log_agent_interaction, AsyncPerformanceLogger
from config.settings import settings

logger = get_logger("agents.gemini")


class GeminiError(AgentError):
    """Specific error for Gemini operations."""
    pass


class GeminiAgent(BaseAgent):
    """
    Gemini agent implementation using 2.5-pro model.
    
    I'm implementing a comprehensive Gemini client that provides research and creative
    capabilities with proper safety settings, cost tracking, and error handling.
    """
    
    def __init__(self, name: str = "gemini", api_key: Optional[str] = None):
        super().__init__(AgentType.GEMINI, name, settings)
        
        # Gemini configuration
        self.api_key = api_key or settings.google_api_key.get_secret_value()
        self.model = settings.gemini_model
        self.max_output_tokens = settings.gemini_max_output_tokens
        self.temperature = settings.gemini_temperature
        self.top_p = settings.gemini_top_p
        self.top_k = settings.gemini_top_k
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Cost tracking (estimated per 1K tokens)
        self.input_cost_per_1k = settings.gemini_input_cost
        self.output_cost_per_1k = settings.gemini_output_cost
        
        # Safety settings
        self.safety_settings = self._setup_safety_settings()
        
        # Define capabilities
        self._setup_capabilities()
        
        self.logger.info(f"Gemini agent initialized with model: {self.model}")
    
    def _setup_safety_settings(self) -> List[types.SafetySetting]:
        """Configure safety settings for Gemini."""
        return [
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_MEDIUM_AND_ABOVE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_MEDIUM_AND_ABOVE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_MEDIUM_AND_ABOVE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_MEDIUM_AND_ABOVE',
            ),
        ]
    
    def _setup_capabilities(self) -> None:
        """Define Gemini agent capabilities."""
        self._capabilities = [
            AgentCapability(
                name="creative_thinking",
                description="Generate creative and innovative solutions",
                supported_formats=["text", "markdown", "structured"],
                limitations=["Safety filtering may block some content"],
                cost_per_request=0.005
            ),
            AgentCapability(
                name="research_synthesis",
                description="Synthesize information from multiple perspectives",
                supported_formats=["text", "markdown", "json"],
                limitations=["Knowledge cutoff limitations"],
                cost_per_request=0.008
            ),
            AgentCapability(
                name="technical_analysis",
                description="Analyze complex technical problems",
                supported_formats=["text", "code", "diagrams"],
                limitations=["Cannot execute or test code"],
                cost_per_request=0.010
            ),
            AgentCapability(
                name="alternative_approaches",
                description="Propose alternative solutions and approaches",
                supported_formats=["text", "structured_analysis"],
                limitations=["May require validation for practicality"],
                cost_per_request=0.007
            ),
            AgentCapability(
                name="long_form_content",
                description="Generate detailed, comprehensive content",
                supported_formats=["text", "markdown", "documentation"],
                limitations=["Output length limits"],
                cost_per_request=0.012
            )
        ]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for cost calculation."""
        # Rough estimation for Gemini tokens (similar to GPT)
        return int(len(text.split()) * 1.3)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost based on token usage."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def _build_generation_config(self, context: TaskContext) -> types.GenerateContentConfig:
        """Build generation configuration for Gemini."""
        return types.GenerateContentConfig(
            temperature=context.temperature or self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=context.max_tokens or self.max_output_tokens,
            safety_settings=self.safety_settings
        )
    
    def _build_system_prompt(self, context: TaskContext) -> str:
        """Build appropriate system prompt based on agent role."""
        if context.agent_role.value == "researcher":
            return """You are a comprehensive technical researcher and analyst. Your role is to:
- Conduct thorough research on technical topics
- Synthesize information from multiple perspectives
- Provide balanced, well-reasoned analysis
- Identify trends, patterns, and implications
- Suggest innovative approaches and solutions
- Present findings in a clear, structured format

Focus on accuracy, depth, and practical insights."""

        elif context.agent_role.value == "specialist":
            return """You are a technical specialist with deep expertise. Your role is to:
- Provide expert analysis on complex technical problems
- Offer specialized insights and advanced solutions
- Identify nuanced issues that others might miss
- Propose cutting-edge approaches and methodologies
- Evaluate trade-offs and implications thoroughly
- Share best practices from specialized domains

Focus on expertise, innovation, and technical excellence."""

        else:
            return """You are a creative and analytical AI assistant specializing in technical problem-solving. Your role is to:
- Think creatively about complex problems
- Propose innovative solutions and approaches
- Provide comprehensive analysis and insights
- Consider multiple perspectives and alternatives
- Generate detailed, well-structured responses
- Focus on practical applicability and value

Be thorough, creative, and solution-oriented in your responses."""
    
    async def _make_generation_request(self, prompt: str, context: TaskContext) -> Any:
        """Make a content generation request to Gemini."""
        try:
            # Build full prompt with system context
            system_prompt = self._build_system_prompt(context)
            full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}"
            
            # Build configuration
            config = self._build_generation_config(context)
            
            # Make API request
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config
            )
            
            return response
            
        except QuotaExhaustedError as e:
            raise AgentRateLimitError(
                f"Gemini quota exhausted: {e}",
                agent_type=self.agent_type.value,
                error_code="QUOTA_EXHAUSTED"
            ) from e
        
        except PermissionDeniedError as e:
            raise AgentAuthenticationError(
                f"Gemini permission denied: {e}",
                agent_type=self.agent_type.value,
                error_code="PERMISSION_DENIED"
            ) from e
        
        except ValidationError as e:
            raise GeminiError(
                f"Gemini validation error: {e}",
                agent_type=self.agent_type.value,
                error_code="VALIDATION_ERROR"
            ) from e
        
        except (ClientError, ServerError) as e:
            raise GeminiError(
                f"Gemini API error: {e}",
                agent_type=self.agent_type.value,
                error_code="API_ERROR"
            ) from e
        
        except asyncio.TimeoutError as e:
            raise AgentTimeoutError(
                f"Gemini request timed out: {e}",
                agent_type=self.agent_type.value,
                error_code="REQUEST_TIMEOUT"
            ) from e
    
    def _parse_generation_response(self, response: Any, input_tokens: int, execution_time: float) -> AgentResponse:
        """Parse Gemini generation response into standardized format."""
        try:
            # Extract content
            if hasattr(response, 'text') and response.text:
                content = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Handle multiple candidates - take the first one
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content.parts[0].text if candidate.content.parts else ""
                else:
                    content = str(candidate)
            else:
                content = str(response)
            
            # Estimate output tokens and calculate cost
            output_tokens = self._estimate_tokens(content)
            total_tokens = input_tokens + output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Check for safety blocks
            safety_blocked = False
            safety_ratings = []
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    safety_ratings = [
                        {
                            "category": rating.category,
                            "probability": rating.probability,
                            "blocked": getattr(rating, 'blocked', False)
                        }
                        for rating in candidate.safety_ratings
                    ]
                    safety_blocked = any(rating.get('blocked', False) for rating in safety_ratings)
                
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason in ['SAFETY', 'BLOCKED']:
                        safety_blocked = True
            
            if safety_blocked:
                return AgentResponse(
                    success=False,
                    content="Content blocked by safety filters",
                    agent_type=self.agent_type.value,
                    execution_time_ms=execution_time * 1000,
                    error_message="Content blocked by safety filters",
                    metadata={
                        "safety_blocked": True,
                        "safety_ratings": safety_ratings,
                        "model": self.model
                    }
                )
            
            return AgentResponse(
                success=True,
                content=content,
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                tokens_used=total_tokens,
                cost_usd=cost,
                metadata={
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "safety_ratings": safety_ratings,
                    "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if hasattr(response, 'candidates') and response.candidates else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            return AgentResponse(
                success=False,
                content="",
                agent_type=self.agent_type.value,
                execution_time_ms=execution_time * 1000,
                error_message=f"Response parsing error: {e}",
                metadata={"model": self.model, "raw_response": str(response)}
            )
    
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate a response using Gemini."""
        start_time = time.time()
        
        async with AsyncPerformanceLogger(self.logger, "gemini_generate", task_id=context.task_id):
            try:
                # Estimate input tokens
                input_tokens = self._estimate_tokens(prompt)
                
                # Make API request with retry logic
                response = await self.execute_with_retry(
                    self._make_generation_request, prompt, context
                )
                
                execution_time = time.time() - start_time
                parsed_response = self._parse_generation_response(response, input_tokens, execution_time)
                
                # Update metrics
                self._update_metrics(parsed_response)
                
                # Log interaction
                log_agent_interaction(
                    self.logger,
                    self.name,
                    "generate",
                    input_data=prompt,
                    output_data=parsed_response.content,
                    metadata={
                        "task_id": context.task_id,
                        "model": self.model,
                        "tokens_used": parsed_response.tokens_used,
                        "cost_usd": parsed_response.cost_usd,
                        "execution_time_ms": parsed_response.execution_time_ms,
                        "safety_blocked": parsed_response.metadata.get("safety_blocked", False)
                    }
                )
                
                return parsed_response
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Gemini generation failed: {e}")
                
                return AgentResponse(
                    success=False,
                    content="",
                    agent_type=self.agent_type.value,
                    execution_time_ms=execution_time * 1000,
                    error_message=str(e),
                    metadata={"task_id": context.task_id, "model": self.model}
                )
    
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide comprehensive critique using Gemini's analytical capabilities."""
        critique_prompt = f"""Please provide a comprehensive critique and analysis of the following solution for the task: "{original_task}"

Solution to analyze:
{content}

Provide a thorough analysis including:

**Creative Assessment:**
- Innovation and originality of the approach
- Creative problem-solving elements
- Unique insights or perspectives demonstrated

**Technical Evaluation:**
- Technical soundness and correctness
- Efficiency and performance considerations
- Scalability and maintainability aspects
- Security and reliability factors

**Alternative Perspectives:**
- Different approaches that could be considered
- Trade-offs and decision rationale
- Potential improvements or optimizations
- Edge cases and corner scenarios

**Implementation Analysis:**
- Practical feasibility of the solution
- Resource requirements and constraints
- Integration considerations
- Deployment and operational aspects

**Holistic Evaluation:**
- Overall quality and completeness
- Alignment with best practices
- Long-term viability and sustainability
- Risk assessment and mitigation strategies

Focus on providing balanced, constructive feedback that considers multiple dimensions of the solution."""

        # Update context for critique task
        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        critique_context.agent_role = "specialist"
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], 
                             context: TaskContext) -> AgentResponse:
        """Propose innovative solutions using Gemini's creative capabilities."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        solution_prompt = f"""Analyze the following challenge and propose innovative, comprehensive solutions:

**Challenge:** {task_description}

**Constraints:**
{constraints_text}

Please provide creative and thorough solutions including:

**1. Creative Problem Analysis:**
- Multiple perspectives on the problem
- Hidden assumptions and underlying factors
- Root cause analysis and systemic considerations
- Innovative framing of the challenge

**2. Solution Exploration:**
- Multiple creative approaches and alternatives
- Unconventional methods and techniques
- Cross-disciplinary insights and applications
- Emerging technologies and methodologies

**3. Detailed Solution Design:**
- Step-by-step implementation strategy
- Technical architecture and design patterns
- Resource allocation and timeline considerations
- Integration points and dependencies

**4. Innovation Opportunities:**
- Novel applications of existing technologies
- Creative combinations of different approaches
- Potential for breakthrough solutions
- Future-oriented considerations

**5. Risk-Benefit Analysis:**
- Comprehensive risk assessment
- Mitigation strategies and contingency plans
- Expected benefits and success metrics
- Long-term implications and sustainability

**6. Implementation Roadmap:**
- Phased approach with milestones
- Critical success factors
- Monitoring and evaluation criteria
- Adaptation and evolution strategies

Think outside the box while maintaining practical applicability."""

        # Update context for solution proposal
        solution_context = context.model_copy()
        solution_context.task_type = TaskType.ANALYSIS
        solution_context.agent_role = "specialist"
        
        return await self.generate(solution_prompt, solution_context)
    
    async def research_topic(self, topic: str, focus_areas: List[str], context: TaskContext) -> AgentResponse:
        """Conduct comprehensive research using Gemini's analytical capabilities."""
        focus_text = "\n".join(f"- {area}" for area in focus_areas) if focus_areas else "Comprehensive overview"
        
        research_prompt = f"""Conduct an in-depth research analysis on the following topic:

**Research Topic:** {topic}

**Focus Areas:**
{focus_text}

Please provide comprehensive research covering:

**1. Foundational Understanding:**
- Core concepts and definitions
- Historical context and evolution
- Theoretical foundations and principles
- Key stakeholders and ecosystem

**2. Current State Analysis:**
- Present landscape and major players
- Current trends and developments
- Market dynamics and adoption patterns
- Regulatory and policy considerations

**3. Technical Deep Dive:**
- Underlying technologies and methodologies
- Implementation approaches and frameworks
- Standards, protocols, and best practices
- Tools, platforms, and infrastructure

**4. Comparative Analysis:**
- Alternative approaches and solutions
- Competitive landscape analysis
- Strengths, weaknesses, and trade-offs
- Use case scenarios and applicability

**5. Future Outlook:**
- Emerging trends and innovations
- Predicted developments and evolution
- Opportunities and challenges ahead
- Strategic implications and recommendations

**6. Practical Applications:**
- Real-world implementations and case studies
- Success stories and failure lessons
- Common pitfalls and how to avoid them
- Best practices and guidelines

**7. Research Resources:**
- Key publications and authorities
- Important tools and frameworks
- Learning resources and communities
- Standards organizations and bodies

Provide thorough, well-researched insights with multiple perspectives and practical implications."""

        # Update context for research task
        research_context = context.model_copy()
        research_context.task_type = TaskType.RESEARCH
        research_context.agent_role = "researcher"
        
        return await self.generate(research_prompt, research_context)
    
    async def brainstorm_alternatives(self, problem: str, current_approach: str, context: TaskContext) -> AgentResponse:
        """Brainstorm alternative approaches using Gemini's creative thinking."""
        brainstorm_prompt = f"""Help brainstorm creative alternative approaches to solve this problem:

**Problem:** {problem}

**Current Approach:** {current_approach}

Please provide creative brainstorming including:

**1. Alternative Paradigms:**
- Completely different approaches to the problem
- Paradigm shifts and reframing techniques
- Cross-industry inspirations and analogies
- Unconventional methodologies and perspectives

**2. Technology Alternatives:**
- Different technology stacks and platforms
- Emerging technologies that could be applied
- Hybrid approaches combining multiple technologies
- Low-tech and high-tech alternative solutions

**3. Process Innovations:**
- Alternative workflows and methodologies
- Innovative process designs and optimizations
- Automation vs. manual approach trade-offs
- Collaborative and distributed approaches

**4. Creative Solutions:**
- Out-of-the-box thinking and innovations
- Artistic and design-thinking approaches
- Gamification and engagement strategies
- Social and community-driven solutions

**5. Resource Optimization:**
- Cost-effective alternative approaches
- Resource-constrained solutions
- Sustainability and environmental considerations
- Scalability and efficiency improvements

**6. Implementation Variations:**
- Phased vs. big-bang approaches
- Incremental vs. revolutionary changes
- Top-down vs. bottom-up strategies
- Centralized vs. distributed implementations

Think creatively and propose diverse alternatives that challenge conventional approaches."""

        # Update context for brainstorming
        brainstorm_context = context.model_copy()
        brainstorm_context.task_type = TaskType.RESEARCH
        brainstorm_context.agent_role = "researcher"
        
        return await self.generate(brainstorm_prompt, brainstorm_context)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check specific to Gemini."""
        try:
            # Test basic functionality
            test_context = TaskContext(
                task_id="health_check",
                task_type=TaskType.CUSTOM,
                timeout_seconds=30
            )
            
            start_time = time.time()
            response = await self.generate("Respond with 'Gemini health check successful'", test_context)
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
                "safety_settings_count": len(self.safety_settings),
                "capabilities_count": len(self._capabilities),
                "performance_metrics": self.performance_metrics
            }
            
            if not response.success:
                health_info["error"] = response.error_message
                health_info["safety_blocked"] = response.metadata.get("safety_blocked", False)
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"Gemini health check failed: {e}")
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
        """Shutdown Gemini agent and cleanup resources."""
        self.logger.info("Shutting down Gemini agent")
        
        # Close the client if it has cleanup methods
        if hasattr(self.client, 'close'):
            try:
                await self.client.close()
            except Exception as e:
                self.logger.warning(f"Error closing Gemini client: {e}")
        
        # Call parent shutdown
        await super().shutdown()
