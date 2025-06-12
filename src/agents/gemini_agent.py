"""
Gemini Agent implementation for AngelaMCP using the new Google Gen AI SDK.

This agent specializes in research, documentation, and creative problem-solving.
I'm using the latest google-genai SDK for better performance and features.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional

# New Gemini SDK imports
from google import genai
from google.genai import types

from src.agents.base import BaseAgent, AgentType, AgentResponse, TaskContext, TaskType
from src.utils.logger import get_logger
from src.utils.exceptions import AgentError
from config.settings import settings

logger = get_logger("agents.gemini")


class GeminiAgent(BaseAgent):
    """
    Gemini agent using the new Google Gen AI SDK.

    Specialized in:
    - Research and analysis
    - Documentation generation
    - Creative problem-solving
    - Best practices research
    """

    def __init__(self):
        super().__init__(
            agent_type=AgentType.GEMINI,
            name="Gemini Research Specialist",
            capabilities=[
                "research_analysis",
                "documentation",
                "creative_solutions",
                "best_practices",
                "comprehensive_analysis"
            ]
        )

        # Initialize the new Gemini client
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = settings.gemini_model
        self.max_retries = settings.gemini_max_retries
        self.retry_delay = settings.gemini_retry_delay

        # Default generation config
        self.default_config = types.GenerateContentConfig(
            max_output_tokens=settings.gemini_max_output_tokens,
            temperature=settings.gemini_temperature,
            top_p=settings.gemini_top_p,
            top_k=settings.gemini_top_k,
            stop_sequences=[],
        )

        logger.info(f"Initialized Gemini agent with model: {self.model}")

    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """Generate response using Gemini with the new SDK."""
        start_time = time.time()

        try:
            # Build generation config
            config = self._build_config(context)

            # Add system instruction based on task type
            system_instruction = self._get_system_instruction(context)
            if system_instruction:
                config.system_instruction = system_instruction

            logger.debug(f"Generating response for prompt: {prompt[:100]}...")

            # Generate content with retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config
                    )

                    # Extract response text
                    if response.candidates and response.candidates[0].content.parts:
                        content = response.candidates[0].content.parts[0].text

                        # Calculate metrics
                        execution_time = time.time() - start_time
                        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

                        return AgentResponse(
                            agent_type=self.agent_type,
                            content=content,
                            confidence=self._calculate_confidence(response),
                            execution_time_ms=execution_time * 1000,
                            token_usage={
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens
                            },
                            metadata={
                                "model": self.model,
                                "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                                "safety_ratings": self._extract_safety_ratings(response),
                                "attempt": attempt + 1
                            }
                        )
                    else:
                        raise AgentError("No valid response content from Gemini")

                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise AgentError(f"Gemini API failed after {self.max_retries + 1} attempts: {e}")

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}", exc_info=True)
            raise AgentError(f"Gemini generation error: {e}")

    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """Provide comprehensive critique using Gemini's analytical capabilities."""
        critique_prompt = f"""Please provide a thorough critique of the following solution for the task: "{original_task}"

Solution to review:
{content}

Provide a comprehensive analysis including:

**Research-Based Assessment:**
- How well does this align with current best practices?
- What recent developments or trends are relevant?
- Are there established patterns or frameworks being followed?

**Creative Alternatives:**
- What innovative approaches could be considered?
- Are there unconventional but effective solutions?
- How could this be enhanced with creative thinking?

**Documentation Quality:**
- Is the solution well-documented and explained?
- What additional documentation would be helpful?
- How clear is the approach for future maintenance?

**Comprehensive Analysis:**
- Long-term implications and sustainability
- Scalability considerations
- Integration with broader ecosystem
- Knowledge gaps that should be addressed

**Research Recommendations:**
- What additional research would strengthen this solution?
- Are there relevant case studies or examples?
- What resources should be consulted for improvement?

Focus on providing deep, research-backed insights that go beyond surface-level review."""

        # Update context for critique task
        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CODE_REVIEW
        critique_context.agent_role = "research_analyst"

        return await self.generate(critique_prompt, critique_context)

    async def propose_solution(self, task_description: str, constraints: List[str],
                             context: TaskContext) -> AgentResponse:
        """Propose innovative solutions using Gemini's research capabilities."""
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"

        solution_prompt = f"""Conduct comprehensive research and propose an innovative solution for:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Please provide a research-driven solution including:

**1. Background Research:**
- Current state of the art in this domain
- Historical context and evolution
- Key players and established solutions
- Recent innovations and emerging trends

**2. Comprehensive Solution Design:**
- Step-by-step implementation strategy
- Technical architecture and design patterns
- Resource allocation and timeline considerations
- Integration points and dependencies

**3. Innovation Opportunities:**
- Novel applications of existing technologies
- Creative combinations of different approaches
- Potential for breakthrough solutions
- Future-oriented considerations

**4. Risk-Benefit Analysis:**
- Comprehensive risk assessment
- Mitigation strategies and contingency plans
- Expected benefits and success metrics
- Long-term implications and sustainability

**5. Implementation Roadmap:**
- Phased approach with milestones
- Critical success factors
- Monitoring and evaluation criteria
- Adaptation and evolution strategies

**6. Knowledge Resources:**
- Key references and documentation
- Communities and expert networks
- Tools and frameworks to leverage
- Continuous learning recommendations

Think outside the box while maintaining practical applicability."""

        # Update context for solution proposal
        solution_context = context.model_copy()
        solution_context.task_type = TaskType.ANALYSIS
        solution_context.agent_role = "research_specialist"

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

**7. Resource Compilation:**
- Essential tools and frameworks
- Key documentation and references
- Learning resources and communities
- Standards and specifications

Provide authoritative, well-researched information with practical insights."""

        # Update context for research task
        research_context = context.model_copy()
        research_context.task_type = TaskType.RESEARCH
        research_context.agent_role = "researcher"

        return await self.generate(research_prompt, research_context)

    async def generate_documentation(self, topic: str, audience: str, context: TaskContext) -> AgentResponse:
        """Generate comprehensive documentation using Gemini."""
        doc_prompt = f"""Create comprehensive documentation for:

**Topic:** {topic}
**Target Audience:** {audience}

Please generate documentation that includes:

**1. Executive Summary:**
- High-level overview
- Key benefits and value proposition
- Quick start guide

**2. Detailed Content:**
- Step-by-step instructions
- Technical specifications
- Configuration examples
- Best practices

**3. Visual Elements:**
- Diagrams and flowcharts (described in text)
- Code examples and snippets
- Configuration files
- Screenshots descriptions

**4. Troubleshooting:**
- Common issues and solutions
- Debugging approaches
- FAQ section
- Support resources

**5. Advanced Topics:**
- Customization options
- Integration guides
- Performance optimization
- Security considerations

**6. References:**
- Additional resources
- Related documentation
- Standards and specifications
- Community resources

Make it comprehensive, well-structured, and appropriate for the target audience."""

        doc_context = context.model_copy()
        doc_context.task_type = TaskType.DOCUMENTATION
        doc_context.agent_role = "technical_writer"

        return await self.generate(doc_prompt, doc_context)

    async def health_check(self) -> Dict[str, Any]:
        """Check Gemini agent health and connectivity."""
        try:
            start_time = time.time()

            # Simple test request
            response = self.client.models.generate_content(
                model=self.model,
                contents="Hello, this is a health check. Please respond with 'OK'.",
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.0
                )
            )

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "model": self.model,
                "response_time": response_time,
                "last_check": time.time(),
                "capabilities": self.capabilities
            }

        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": time.time()
            }

    def _build_config(self, context: TaskContext) -> types.GenerateContentConfig:
        """Build generation config based on context."""
        config = types.GenerateContentConfig(
            max_output_tokens=self.default_config.max_output_tokens,
            temperature=self.default_config.temperature,
            top_p=self.default_config.top_p,
            top_k=self.default_config.top_k,
        )

        # Adjust config based on task type
        if context.task_type == TaskType.CODE_GENERATION:
            config.temperature = 0.3  # More deterministic for code
        elif context.task_type == TaskType.CREATIVE:
            config.temperature = 0.9  # More creative
        elif context.task_type == TaskType.RESEARCH:
            config.temperature = 0.7  # Balanced for research

        return config

    def _get_system_instruction(self, context: TaskContext) -> Optional[str]:
        """Get system instruction based on context."""
        instructions = {
            TaskType.RESEARCH: "You are a comprehensive research specialist. Provide thorough, well-researched analysis with citations and practical insights.",
            TaskType.DOCUMENTATION: "You are a technical documentation expert. Create clear, comprehensive documentation that is well-structured and user-friendly.",
            TaskType.CREATIVE: "You are a creative problem-solver. Think outside the box and propose innovative solutions while maintaining practical applicability.",
            TaskType.ANALYSIS: "You are an analytical expert. Provide deep, systematic analysis with multiple perspectives and actionable insights.",
            TaskType.CODE_REVIEW: "You are a research-oriented code reviewer. Focus on best practices, innovative approaches, and comprehensive improvement suggestions."
        }

        return instructions.get(context.task_type)

    def _calculate_confidence(self, response) -> float:
        """Calculate confidence score based on response quality."""
        try:
            # Base confidence
            confidence = 0.8

            # Adjust based on safety ratings
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == "STOP":
                        confidence += 0.1
                    elif candidate.finish_reason in ["MAX_TOKENS", "SAFETY"]:
                        confidence -= 0.2

                # Check content length (longer usually indicates more thoughtful response)
                if hasattr(candidate, 'content') and candidate.content.parts:
                    content_length = len(candidate.content.parts[0].text)
                    if content_length > 1000:
                        confidence += 0.1

            return min(1.0, max(0.0, confidence))

        except Exception:
            return 0.7  # Default confidence

    def _extract_safety_ratings(self, response) -> List[Dict[str, Any]]:
        """Extract safety ratings from response."""
        try:
            safety_ratings = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        safety_ratings.append({
                            "category": rating.category,
                            "probability": rating.probability
                        })
            return safety_ratings
        except Exception:
            return []
