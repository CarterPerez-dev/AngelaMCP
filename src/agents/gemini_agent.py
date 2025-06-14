"""
Gemini Agent for AngelaMCP - FIXED VERSION.

Fixed issues:
- Corrected Part.from_text() API call to use keyword argument
- Improved error handling for API changes
- Added fallback approaches for content creation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any

import google.genai as genai
from google.genai import types

from src.agents.base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType, 
    AgentCapabilities, TokenUsage, track_performance
)
from src.utils import get_logger
from src.utils import AgentError
from config import settings


class GeminiAgent(BaseAgent):
    """
    Gemini agent for research and documentation.
    
    This agent acts as a research specialist and documentation expert,
    providing comprehensive analysis and best practices guidance.
    """
    
    def __init__(self):
        # Define Gemini capabilities
        capabilities = AgentCapabilities(
            can_execute_code=False,
            can_read_files=False,
            can_write_files=False,
            can_browse_web=False,
            can_use_tools=True,
            supported_languages=[
                "python", "javascript", "typescript", "java", "cpp", "c", 
                "go", "rust", "ruby", "php", "swift", "kotlin", "scala",
                "html", "css", "sql", "bash", "r", "matlab", "dart"
            ],
            supported_formats=[
                "text", "markdown", "code", "json", "yaml", "xml"
            ],
            max_context_length=1000000,
            supports_streaming=True,
            supports_function_calling=True
        )
        
        super().__init__(AgentType.GEMINI, "gemini", capabilities)
        
        # Gemini configuration
        self.api_key = settings.google_api_key.get_secret_value()
        self.model = settings.gemini_model
        self.max_output_tokens = settings.gemini_max_output_tokens
        self.temperature = settings.gemini_temperature
        self.top_p = settings.gemini_top_p
        self.top_k = settings.gemini_top_k
        self.timeout = settings.gemini_timeout
        
        # Rate limiting
        self.rate_limit = settings.gemini_rate_limit
        self.max_retries = settings.gemini_max_retries
        self.retry_delay = settings.gemini_retry_delay
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Verify API key
        self._verify_api_key()
    
    def _verify_api_key(self) -> None:
        """Verify Gemini API key is valid."""
        try:
            if not self.api_key or len(self.api_key) < 20:
                raise AgentError("Invalid Gemini API key format")
            
            # Test API connection by listing models
            try:
                models = list(self.client.models.list(config={'page_size': 1}))
                if not models:
                    raise AgentError("No models available")
            except Exception as e:
                raise AgentError(f"API connection test failed: {e}")
            
            self.logger.info("✅ Gemini API key verified")
            
        except Exception as e:
            self.logger.error(f"❌ Gemini API key verification failed: {e}")
            raise AgentError(f"Gemini setup failed: {e}")
    
    @track_performance
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate response using Gemini API.
        
        This method handles the core interaction with Gemini's API,
        including rate limiting and error handling.
        """
        start_time = time.time()
        
        try:
            self.agent_logger.log_request(f"Generating response for {context.task_type.value} task")
            
            # Check rate limit
            await self._wait_for_rate_limit(self.rate_limit)
            
            # Build enhanced prompt for Gemini
            enhanced_prompt = await self._enhance_prompt_for_gemini(prompt, context)
            
            # Make API call with retries
            response = await self._make_api_call(enhanced_prompt, context)
            
            # Extract response content
            response_content = response.text or ""
            
            execution_time = (time.time() - start_time) * 1000
            
            # Estimate token usage (Gemini doesn't always provide exact counts)
            token_usage = self._estimate_token_usage(enhanced_prompt, response_content)
            
            # Create response
            response_obj = AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content=response_content,
                success=True,
                confidence=0.82,  # Gemini is reliable for research and documentation
                execution_time_ms=execution_time,
                token_usage=token_usage,
                metadata={
                    "model": self.model,
                    "task_type": context.task_type.value,
                    "agent_role": context.agent_role.value if context.agent_role else None,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                }
            )
            
            self.agent_logger.log_response(f"Generated {len(response_content)} characters")
            return response_obj
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Gemini generation failed: {e}"
            
            self.agent_logger.log_error(error_msg, e)
            
            return AgentResponse(
                agent_type=self.agent_type,
                agent_name=self.name,
                content="",
                success=False,
                execution_time_ms=execution_time,
                error=error_msg
            )
    
    async def _enhance_prompt_for_gemini(self, prompt: str, context: TaskContext) -> str:
        """Enhance prompt specifically for Gemini's strengths."""
        
        enhanced_parts = []
        
        # Add role context for Gemini
        role_prompts = {
            "researcher": "You are an expert researcher with access to comprehensive knowledge across multiple domains.",
            "documentation_specialist": "You excel at creating clear, comprehensive documentation and tutorials.",
            "best_practices_expert": "You have deep knowledge of industry best practices and standards.",
            "technical_writer": "You specialize in explaining complex technical concepts clearly and accurately."
        }
        
        if context.agent_role:
            role_key = context.agent_role.value if hasattr(context.agent_role, 'value') else str(context.agent_role)
            if role_key in role_prompts:
                enhanced_parts.append(role_prompts[role_key])
        
        # Add task-specific context for Gemini's strengths
        if context.task_type == TaskType.RESEARCH:
            enhanced_parts.append("""Conduct thorough research and provide comprehensive information including:

1. **Current State**: Latest developments and trends
2. **Best Practices**: Industry-standard approaches and methodologies
3. **Comparative Analysis**: Different options with pros and cons
4. **Expert Insights**: Professional recommendations and insights
5. **Implementation Guidelines**: Practical steps and considerations
6. **Resources**: Relevant tools, frameworks, and references

Provide well-structured, evidence-based information.""")
        
        elif context.task_type == TaskType.DOCUMENTATION:
            enhanced_parts.append("""Create comprehensive documentation that includes:

1. **Clear Overview**: Purpose and scope explanation
2. **Detailed Instructions**: Step-by-step procedures
3. **Code Examples**: Working examples with explanations
4. **Best Practices**: Recommended approaches and patterns
5. **Troubleshooting**: Common issues and solutions
6. **References**: Links to additional resources

Ensure clarity, completeness, and professional presentation.""")
        
        elif context.task_type == TaskType.ANALYSIS:
            enhanced_parts.append("""Provide in-depth analysis covering:

1. **Comprehensive Assessment**: Thorough evaluation of all aspects
2. **Multiple Perspectives**: Different viewpoints and considerations
3. **Evidence-Based Insights**: Data and research-backed conclusions
4. **Practical Implications**: Real-world impact and applications
5. **Future Considerations**: Long-term trends and implications
6. **Actionable Recommendations**: Specific next steps and guidance

Be thorough, objective, and insightful in your analysis.""")
        
        elif context.task_type == TaskType.DEBATE:
            enhanced_parts.append("""Present your position in this collaborative discussion:

1. **Well-Researched Position**: Evidence-based stance with supporting data
2. **Comprehensive Context**: Historical background and current landscape
3. **Multiple Considerations**: Various factors and trade-offs
4. **Industry Standards**: Alignment with established best practices
5. **Future-Oriented Thinking**: Long-term implications and trends
6. **Balanced Perspective**: Acknowledgment of limitations and alternatives

Provide thoughtful, research-backed contributions to the discussion.""")
        
        # Combine with original prompt
        if enhanced_parts:
            enhanced_prompt = "\n\n".join(enhanced_parts) + "\n\n" + prompt
        else:
            enhanced_prompt = prompt
        
        return enhanced_prompt
    
    async def _make_api_call(self, prompt: str, context: TaskContext) -> Any:
        """Make Gemini API call with proper error handling - FIXED."""
        
        try:
            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                candidate_count=1
            )
            
            # Create content with proper API call - FIXED
            try:
                # Try the new API format first
                content = types.Part.from_text(text=prompt)
            except TypeError:
                # Fallback to older format if new one fails
                try:
                    content = types.Part(text=prompt)
                except:
                    # Final fallback - use string directly
                    content = prompt
            
            # Make the API call
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=content,
                config=config
            )
            
            if not response or not response.text:
                raise AgentError("Empty response from Gemini API")
            
            return response
            
        except Exception as e:
            # Handle different types of Gemini API errors
            error_msg = str(e).lower()
            
            if "quota" in error_msg or "rate limit" in error_msg:
                self.logger.warning(f"Gemini rate limit hit: {e}")
                await asyncio.sleep(self.retry_delay)
                raise AgentError(f"Rate limit exceeded: {e}")
            elif "authentication" in error_msg or "api key" in error_msg:
                raise AgentError(f"Authentication failed: {e}")
            elif "invalid" in error_msg:
                raise AgentError(f"Invalid request: {e}")
            else:
                raise AgentError(f"Gemini API error: {e}")
    
    def _estimate_token_usage(self, prompt: str, response: str) -> TokenUsage:
        """Estimate token usage for Gemini interaction."""
        
        # Rough token estimation for Gemini
        prompt_tokens = len(prompt.split()) * 1.2
        response_tokens = len(response.split()) * 1.2
        
        return TokenUsage(
            input_tokens=int(prompt_tokens),
            output_tokens=int(response_tokens),
            total_tokens=int(prompt_tokens + response_tokens)
        )
    
    async def research_topic(self, topic: str, context: TaskContext) -> AgentResponse:
        """Conduct comprehensive research on a topic."""
        
        research_prompt = f"""Conduct comprehensive research on the following topic:

**Topic:** {topic}

Please provide a thorough research report covering:

1. **Overview and Background**
   - Current state and context
   - Historical development
   - Key concepts and terminology

2. **Current Trends and Developments**
   - Latest advancements and innovations
   - Emerging patterns and directions
   - Industry adoption and usage

3. **Best Practices and Standards**
   - Industry-standard approaches
   - Recommended methodologies
   - Professional guidelines and frameworks

4. **Comparative Analysis**
   - Different approaches and solutions
   - Pros and cons of each option
   - Use case recommendations

5. **Expert Insights and Recommendations**
   - Professional opinions and advice
   - Critical success factors
   - Common pitfalls to avoid

6. **Implementation Considerations**
   - Practical steps and requirements
   - Resource needs and constraints
   - Timeline and milestone planning

7. **Future Outlook**
   - Predicted trends and developments
   - Potential challenges and opportunities
   - Long-term implications

8. **Resources and References**
   - Key tools and frameworks
   - Educational materials
   - Community and support resources

Provide comprehensive, well-structured information that would be valuable to both beginners and experts."""
        
        research_context = context.model_copy()
        research_context.task_type = TaskType.RESEARCH
        research_context.agent_role = "researcher"
        research_context.metadata["research_topic"] = topic
        
        return await self.generate(research_prompt, research_context)
    
    async def shutdown(self) -> None:
        """Shutdown Gemini agent and cleanup resources."""
        self.logger.info("Shutting down Gemini agent...")
        
        # Gemini client doesn't require explicit cleanup in current SDK
        # but we can add any necessary cleanup here
        
        await super().shutdown()
