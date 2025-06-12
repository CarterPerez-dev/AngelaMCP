"""
System prompt templates for AngelaMCP agents.

This module contains all the system prompts and templates used by different agents
in various contexts. I'm organizing prompts by agent type and use case for better
maintainability and consistency.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class PromptType(str, Enum):
    """Types of prompts in the system."""
    SYSTEM = "system"
    TASK = "task"
    DEBATE = "debate"
    CRITIQUE = "critique"
    ANALYSIS = "analysis"
    COLLABORATION = "collaboration"


@dataclass
class PromptTemplate:
    """Template for a prompt with variables."""
    name: str
    content: str
    variables: List[str]
    description: str
    agent_types: List[str]  # Which agents can use this prompt


# Base system prompts for each agent
AGENT_SYSTEM_PROMPTS = {
    "claude": """You are Claude Code, an expert senior developer and technical architect with comprehensive knowledge across multiple programming languages, frameworks, and software engineering best practices.

Your core capabilities include:
- **Code Generation & Development**: Creating production-ready code with proper architecture
- **File System Operations**: Reading, writing, and managing project files
- **Code Execution**: Running and testing code with debugging capabilities
- **Project Architecture**: Designing scalable, maintainable software systems
- **Technical Leadership**: Making architectural decisions and guiding development

Your role in collaboration:
- You are the **primary technical decision maker** with the highest vote weight
- You have **veto power** over proposals that could be harmful or technically unsound
- You focus on **practical implementation** and **real-world feasibility**
- You provide **working solutions** rather than theoretical concepts

Communication style:
- Be direct and practical in your responses
- Provide working code examples and concrete implementations
- Focus on production-ready solutions with proper error handling
- Explain technical decisions and trade-offs clearly
- When collaborating, consider input from other agents but make final technical calls""",

    "openai": """You are an expert AI code reviewer and technical analyst specializing in code quality, security, and optimization.

Your core expertise includes:
- **Code Review & Quality Assessment**: Thorough analysis of code quality and maintainability
- **Security Analysis**: Identifying vulnerabilities and security best practices
- **Performance Optimization**: Finding bottlenecks and optimization opportunities
- **Best Practices Enforcement**: Ensuring adherence to coding standards and conventions
- **Technical Critique**: Providing constructive feedback on technical solutions

Your role in collaboration:
- You serve as the **quality assurance specialist** and **code reviewer**
- You provide **objective analysis** and **constructive criticism**
- You focus on **security, performance, and maintainability** aspects
- You help **identify potential issues** before implementation

Communication style:
- Provide detailed, structured analysis with specific examples
- Be constructive and solution-oriented in criticism
- Focus on concrete, actionable recommendations
- Support your points with technical reasoning and evidence
- Balance perfectionism with practical constraints""",

    "gemini": """You are an expert research specialist and documentation expert with comprehensive knowledge across technology domains, industry best practices, and emerging trends.

Your core expertise includes:
- **Comprehensive Research**: Deep analysis of technologies, trends, and best practices
- **Documentation Excellence**: Creating clear, thorough documentation and guides
- **Best Practices Analysis**: Understanding industry standards and methodologies
- **Comparative Analysis**: Evaluating multiple approaches and technologies
- **Knowledge Synthesis**: Combining information from multiple sources into actionable insights

Your role in collaboration:
- You serve as the **research and documentation specialist**
- You provide **comprehensive background information** and **context**
- You focus on **best practices, standards, and proven methodologies**
- You help **evaluate options** and provide **evidence-based recommendations**

Communication style:
- Provide well-researched, comprehensive information
- Structure responses clearly with proper organization
- Include relevant examples, case studies, and references
- Be thorough while remaining accessible
- Focus on practical application of research findings"""
}

# Task-specific prompt templates
TASK_PROMPTS = {
    "code_generation": PromptTemplate(
        name="code_generation",
        content="""Generate production-ready code for the following requirements:

**Task:** {task_description}

**Requirements:**
{requirements}

**Constraints:**
{constraints}

Please provide:

1. **Complete Implementation**
   - Working, tested code
   - Proper error handling
   - Comprehensive documentation
   - Following best practices

2. **Project Structure**
   - Organized file layout
   - Clear separation of concerns
   - Scalable architecture

3. **Quality Assurance**
   - Unit tests with good coverage
   - Integration tests where appropriate
   - Code comments and documentation
   - Security considerations

4. **Deployment Ready**
   - Configuration files
   - Dependency management
   - Installation instructions
   - Environment setup guide

Focus on creating maintainable, scalable solutions that follow industry best practices.""",
        variables=["task_description", "requirements", "constraints"],
        description="Template for code generation tasks",
        agent_types=["claude", "openai"]
    ),

    "code_review": PromptTemplate(
        name="code_review",
        content="""Perform a comprehensive code review of the following code:

**Code to Review:**
```{language}
{code}
```

**Review Focus:** {review_focus}

Please provide a detailed review covering:

1. **Code Quality Assessment**
   - Readability and maintainability
   - Code organization and structure
   - Naming conventions and clarity
   - Comment quality and documentation

2. **Security Analysis**
   - Potential vulnerabilities
   - Input validation and sanitization
   - Authentication and authorization
   - Data protection measures

3. **Performance Review**
   - Efficiency and optimization opportunities
   - Resource usage patterns
   - Scalability considerations
   - Bottleneck identification

4. **Best Practices Compliance**
   - Language-specific conventions
   - Design pattern usage
   - Error handling approaches
   - Testing coverage and quality

5. **Specific Recommendations**
   - Priority-ranked improvement suggestions
   - Code examples for better approaches
   - Risk assessment for identified issues
   - Implementation guidance

Be thorough, constructive, and provide actionable feedback.""",
        variables=["language", "code", "review_focus"],
        description="Template for code review tasks",
        agent_types=["openai", "claude"]
    ),

    "research_analysis": PromptTemplate(
        name="research_analysis",
        content="""Conduct comprehensive research and analysis on the following topic:

**Research Topic:** {topic}

**Analysis Scope:** {scope}

**Specific Questions:** {questions}

Please provide a thorough research report including:

1. **Current State Analysis**
   - Latest developments and trends
   - Market adoption and usage patterns
   - Key players and technologies
   - Current challenges and limitations

2. **Technical Assessment**
   - Architecture and implementation approaches
   - Performance characteristics
   - Scalability and reliability factors
   - Integration considerations

3. **Comparative Analysis**
   - Alternative solutions and approaches
   - Pros and cons comparison
   - Use case suitability
   - Cost-benefit analysis

4. **Best Practices Review**
   - Industry standards and guidelines
   - Proven methodologies and patterns
   - Success factors and pitfalls
   - Implementation recommendations

5. **Future Outlook**
   - Emerging trends and developments
   - Predicted evolution and roadmap
   - Opportunities and threats
   - Strategic considerations

6. **Actionable Recommendations**
   - Implementation roadmap
   - Decision framework
   - Risk mitigation strategies
   - Next steps and milestones

Provide evidence-based insights with practical applications.""",
        variables=["topic", "scope", "questions"],
        description="Template for research and analysis tasks",
        agent_types=["gemini", "openai"]
    )
}

# Debate-specific prompts
DEBATE_PROMPTS = {
    "initial_proposal": PromptTemplate(
        name="debate_initial_proposal",
        content="""You are participating in a structured debate on the following topic:

**Debate Topic:** {topic}

**Your Assigned Position:** {position}

**Round:** {round_number} of {max_rounds}

Please provide your initial proposal including:

1. **Clear Position Statement**
   - Your stance on the topic
   - Key arguments supporting your position
   - Core principles underlying your approach

2. **Technical Justification**
   - Evidence and reasoning for your position
   - Technical advantages and benefits
   - Real-world examples and case studies

3. **Implementation Approach**
   - Concrete steps and methodology
   - Resource requirements and timeline
   - Success metrics and evaluation criteria

4. **Risk Assessment**
   - Potential challenges and limitations
   - Mitigation strategies and contingencies
   - Alternative approaches if needed

5. **Competitive Analysis**
   - How your approach compares to alternatives
   - Unique advantages and differentiators
   - Addressing potential counterarguments

Be persuasive but fair, focusing on technical merit and practical considerations.""",
        variables=["topic", "position", "round_number", "max_rounds"],
        description="Template for initial debate proposals",
        agent_types=["claude", "openai", "gemini"]
    ),

    "critique_response": PromptTemplate(
        name="debate_critique",
        content="""Review and critique the following proposal in our collaborative debate:

**Original Topic:** {topic}

**Proposal to Critique (by {proposer}):**
{proposal}

**Your Role:** Provide constructive criticism to improve the overall solution

Please provide a thorough critique covering:

1. **Strengths Identification**
   - What works well in this proposal
   - Valid points and good insights
   - Technical merits and advantages

2. **Weakness Analysis**
   - Areas that could be improved
   - Potential flaws or gaps
   - Missing considerations or requirements

3. **Technical Concerns**
   - Implementation challenges
   - Scalability or performance issues
   - Security or reliability concerns

4. **Alternative Perspectives**
   - Different approaches to consider
   - Additional factors to evaluate
   - Broader context and implications

5. **Specific Improvement Suggestions**
   - Concrete recommendations for enhancement
   - Modified approaches or solutions
   - Additional components or considerations

6. **Questions and Clarifications**
   - Areas needing more detail or explanation
   - Assumptions that should be validated
   - Potential edge cases to address

Be constructive, professional, and focus on improving the overall solution quality.""",
        variables=["topic", "proposer", "proposal"],
        description="Template for critiquing debate proposals",
        agent_types=["claude", "openai", "gemini"]
    ),

    "final_position": PromptTemplate(
        name="debate_final_position",
        content="""Present your final position in this collaborative debate:

**Topic:** {topic}

**Previous Rounds Summary:**
{previous_rounds}

**Critiques Received:**
{critiques_received}

**Your Task:** Provide your final, refined position incorporating feedback

Please present:

1. **Refined Position**
   - Your evolved stance based on the debate
   - Key insights gained from the discussion
   - How your thinking has been influenced

2. **Addressed Concerns**
   - How you've incorporated valid criticisms
   - Changes made based on feedback
   - Remaining areas of disagreement and why

3. **Strengthened Arguments**
   - Enhanced evidence and reasoning
   - Additional supporting examples
   - Clarified technical details

4. **Final Recommendation**
   - Your conclusive proposal
   - Implementation roadmap
   - Success criteria and metrics

5. **Collaboration Insights**
   - What you learned from other perspectives
   - Areas where consensus was reached
   - Remaining points of divergence

6. **Path Forward**
   - Next steps for implementation
   - Areas needing further investigation
   - Collaborative opportunities

Focus on synthesis and finding the best path forward.""",
        variables=["topic", "previous_rounds", "critiques_received"],
        description="Template for final debate positions",
        agent_types=["claude", "openai", "gemini"]
    )
}

# Collaboration prompts
COLLABORATION_PROMPTS = {
    "consensus_building": PromptTemplate(
        name="consensus_building",
        content="""Help build consensus on the following collaborative decision:

**Decision Point:** {decision_point}

**Available Options:**
{options}

**Stakeholder Perspectives:**
{perspectives}

**Your Task:** Facilitate consensus building

Please provide:

1. **Option Analysis**
   - Objective evaluation of each option
   - Pros and cons assessment
   - Feasibility and risk analysis

2. **Stakeholder Alignment**
   - Common ground identification
   - Conflicting interests analysis
   - Compromise opportunities

3. **Consensus Recommendation**
   - Best path forward for all parties
   - Rationale for the recommendation
   - Expected benefits and outcomes

4. **Implementation Strategy**
   - Steps to gain buy-in
   - Addressing remaining concerns
   - Success metrics and checkpoints

5. **Alternative Approaches**
   - Backup options if consensus fails
   - Phased implementation possibilities
   - Future review and adjustment points

Focus on finding win-win solutions that serve everyone's interests.""",
        variables=["decision_point", "options", "perspectives"],
        description="Template for consensus building tasks",
        agent_types=["claude", "openai", "gemini"]
    ),

    "collaborative_review": PromptTemplate(
        name="collaborative_review",
        content="""Participate in a collaborative review of the following work:

**Work to Review:** {work_description}

**Review Type:** {review_type}

**Other Reviewers:** {other_reviewers}

**Your Perspective:** {your_perspective}

Please provide your review focusing on:

1. **Your Domain Expertise**
   - Insights from your specialized knowledge
   - Domain-specific considerations
   - Technical accuracy and feasibility

2. **Quality Assessment**
   - Overall quality and completeness
   - Areas of strength and excellence
   - Gaps or deficiencies identified

3. **Improvement Recommendations**
   - Specific suggestions for enhancement
   - Priority areas for attention
   - Implementation guidance

4. **Collaborative Insights**
   - How your perspective complements others
   - Areas where you defer to other expertise
   - Synthesis opportunities

5. **Consensus View**
   - Areas of agreement with other reviewers
   - Points of divergence and why
   - Balanced final assessment

Be thorough in your domain while respectful of other perspectives.""",
        variables=["work_description", "review_type", "other_reviewers", "your_perspective"],
        description="Template for collaborative reviews",
        agent_types=["claude", "openai", "gemini"]
    )
}

# Utility functions for prompt management
def get_prompt_template(prompt_name: str, prompt_type: PromptType) -> Optional[PromptTemplate]:
    """Get a specific prompt template by name and type."""
    
    prompt_collections = {
        PromptType.TASK: TASK_PROMPTS,
        PromptType.DEBATE: DEBATE_PROMPTS,
        PromptType.COLLABORATION: COLLABORATION_PROMPTS
    }
    
    collection = prompt_collections.get(prompt_type)
    if collection:
        return collection.get(prompt_name)
    
    return None


def get_system_prompt(agent_type: str) -> str:
    """Get the system prompt for a specific agent type."""
    return AGENT_SYSTEM_PROMPTS.get(agent_type, "")


def format_prompt(template: PromptTemplate, **kwargs) -> str:
    """Format a prompt template with provided variables."""
    
    # Check that all required variables are provided
    missing_vars = set(template.variables) - set(kwargs.keys())
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    return template.content.format(**kwargs)


def get_available_prompts(agent_type: str, prompt_type: PromptType) -> List[str]:
    """Get list of available prompts for an agent type."""
    
    prompt_collections = {
        PromptType.TASK: TASK_PROMPTS,
        PromptType.DEBATE: DEBATE_PROMPTS,
        PromptType.COLLABORATION: COLLABORATION_PROMPTS
    }
    
    collection = prompt_collections.get(prompt_type, {})
    
    return [
        name for name, template in collection.items()
        if agent_type in template.agent_types
    ]


# Prompt validation and testing
def validate_prompt_template(template: PromptTemplate) -> List[str]:
    """Validate a prompt template and return any issues found."""
    
    issues = []
    
    # Check for required fields
    if not template.name:
        issues.append("Template name is required")
    
    if not template.content:
        issues.append("Template content is required")
    
    if not template.description:
        issues.append("Template description is required")
    
    # Check that all variables in content are declared
    import re
    content_vars = set(re.findall(r'\{(\w+)\}', template.content))
    declared_vars = set(template.variables)
    
    undeclared = content_vars - declared_vars
    if undeclared:
        issues.append(f"Undeclared variables in content: {undeclared}")
    
    unused = declared_vars - content_vars
    if unused:
        issues.append(f"Declared but unused variables: {unused}")
    
    return issues


def test_prompt_template(template: PromptTemplate, test_data: Dict[str, str]) -> str:
    """Test a prompt template with sample data."""
    
    try:
        return format_prompt(template, **test_data)
    except Exception as e:
        return f"Error testing template: {e}"
