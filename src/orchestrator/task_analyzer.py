"""
Task Analyzer module for AngelaMCP.
Analyzes incoming tasks and determines the best execution strategy.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Set
import ast
import keyword

from src.models.database import AgentType


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class TaskCategory(Enum):
    """Task categories."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    GENERAL = "general"


@dataclass
class TaskAnalysis:
    """Result of task analysis."""
    category: TaskCategory
    complexity: TaskComplexity
    estimated_duration: int  # seconds
    requires_collaboration: bool
    primary_agent: AgentType
    supporting_agents: List[AgentType]
    priority: int  # 1-10, higher is more urgent
    metadata: Dict[str, Any]
    
    @property
    def all_agents(self) -> List[AgentType]:
        """Get all agents involved in the task."""
        agents = [self.primary_agent]
        agents.extend(self.supporting_agents)
        return list(set(agents))  # Remove duplicates


class TaskAnalyzer:
    """Analyzes tasks to determine optimal execution strategy."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Keywords for different task categories
        self.category_keywords = {
            TaskCategory.CODE_GENERATION: [
                "create", "write", "generate", "implement", "build", "develop",
                "code", "function", "class", "script", "program", "app", "api"
            ],
            TaskCategory.CODE_REVIEW: [
                "review", "check", "audit", "inspect", "analyze code", "examine",
                "validate", "verify code", "assess", "evaluate code"
            ],
            TaskCategory.CODE_DEBUG: [
                "debug", "fix", "error", "bug", "issue", "problem", "troubleshoot",
                "resolve", "repair", "correct", "exception", "crash"
            ],
            TaskCategory.DOCUMENTATION: [
                "document", "docs", "readme", "explain", "describe", "manual",
                "guide", "tutorial", "help", "instructions", "comments"
            ],
            TaskCategory.RESEARCH: [
                "research", "investigate", "study", "explore", "find", "search",
                "learn", "discover", "analyze", "compare", "evaluate"
            ],
            TaskCategory.ANALYSIS: [
                "analyze", "assess", "evaluate", "examine", "study", "review",
                "investigate", "measure", "benchmark", "profile"
            ],
            TaskCategory.PLANNING: [
                "plan", "design", "architecture", "strategy", "approach",
                "roadmap", "outline", "structure", "organize"
            ],
            TaskCategory.TESTING: [
                "test", "testing", "unittest", "integration test", "verify",
                "validate", "check", "qa", "quality assurance"
            ],
            TaskCategory.DEPLOYMENT: [
                "deploy", "deployment", "release", "publish", "ship",
                "production", "staging", "ci/cd", "pipeline"
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            TaskComplexity.TRIVIAL: [
                "hello world", "print", "simple calculation", "basic",
                "trivial", "quick", "one line"
            ],
            TaskComplexity.SIMPLE: [
                "single function", "small script", "basic", "simple",
                "straightforward", "easy"
            ],
            TaskComplexity.MODERATE: [
                "multiple functions", "class", "module", "moderate",
                "several", "medium", "standard"
            ],
            TaskComplexity.COMPLEX: [
                "application", "system", "framework", "complex",
                "advanced", "sophisticated", "multiple files",
                "integration", "database"
            ],
            TaskComplexity.CRITICAL: [
                "critical", "urgent", "production", "security",
                "performance", "scalability", "enterprise",
                "large scale", "mission critical"
            ]
        }
        
        # Programming language detection
        self.programming_languages = {
            "python", "javascript", "typescript", "java", "c++", "c#",
            "go", "rust", "ruby", "php", "kotlin", "swift", "scala",
            "r", "matlab", "sql", "html", "css", "bash", "shell"
        }
        
        # Agent specializations
        self.agent_specializations = {
            AgentType.CLAUDE_CODE: [
                "code generation", "file operations", "debugging",
                "refactoring", "testing", "deployment"
            ],
            AgentType.OPENAI: [
                "code review", "analysis", "optimization", "security",
                "best practices", "documentation"
            ],
            AgentType.GEMINI: [
                "research", "documentation", "planning", "explanation",
                "learning", "comparison", "tutorial"
            ]
        }
    
    def analyze_task(self, task_description: str, context: Dict[str, Any] = None) -> TaskAnalysis:
        """Analyze a task and return execution strategy."""
        context = context or {}
        
        # Normalize the task description
        normalized_task = task_description.lower().strip()
        
        # Analyze different aspects
        category = self._analyze_category(normalized_task)
        complexity = self._analyze_complexity(normalized_task, context)
        duration = self._estimate_duration(category, complexity)
        priority = self._analyze_priority(normalized_task, context)
        
        # Determine agent assignment
        primary_agent, supporting_agents = self._determine_agents(
            category, complexity, normalized_task
        )
        
        # Check if collaboration is needed
        requires_collaboration = self._requires_collaboration(
            category, complexity, len(supporting_agents)
        )
        
        # Extract metadata
        metadata = self._extract_metadata(normalized_task, context)
        
        analysis = TaskAnalysis(
            category=category,
            complexity=complexity,
            estimated_duration=duration,
            requires_collaboration=requires_collaboration,
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            priority=priority,
            metadata=metadata
        )
        
        self.logger.debug(f"Task analysis complete: {analysis}")
        return analysis
    
    def _analyze_category(self, task: str) -> TaskCategory:
        """Determine the task category."""
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in task:
                    score += 1
                    
            if score > 0:
                scores[category] = score
        
        if not scores:
            return TaskCategory.GENERAL
            
        # Return category with highest score
        return max(scores, key=scores.get)
    
    def _analyze_complexity(self, task: str, context: Dict[str, Any]) -> TaskComplexity:
        """Determine task complexity."""
        # Check for explicit complexity indicators
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in task:
                    return complexity
        
        # Analyze task characteristics
        complexity_score = 0
        
        # Check length and detail
        if len(task) > 500:
            complexity_score += 2
        elif len(task) > 200:
            complexity_score += 1
            
        # Check for multiple requirements
        if any(word in task for word in ["and", "also", "additionally", "furthermore"]):
            complexity_score += 1
            
        # Check for technical terms
        technical_terms = [
            "database", "api", "authentication", "security", "performance",
            "scalability", "integration", "architecture", "framework"
        ]
        complexity_score += sum(1 for term in technical_terms if term in task)
        
        # Check for code complexity indicators
        if self._contains_code(task):
            complexity_score += 1
            
        # Convert score to complexity level
        if complexity_score >= 5:
            return TaskComplexity.CRITICAL
        elif complexity_score >= 3:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            return TaskComplexity.MODERATE
        elif complexity_score >= 1:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL
    
    def _estimate_duration(self, category: TaskCategory, complexity: TaskComplexity) -> int:
        """Estimate task duration in seconds."""
        base_durations = {
            TaskComplexity.TRIVIAL: 30,
            TaskComplexity.SIMPLE: 120,
            TaskComplexity.MODERATE: 300,
            TaskComplexity.COMPLEX: 900,
            TaskComplexity.CRITICAL: 1800
        }
        
        category_multipliers = {
            TaskCategory.CODE_GENERATION: 1.5,
            TaskCategory.CODE_REVIEW: 1.2,
            TaskCategory.CODE_DEBUG: 2.0,
            TaskCategory.DOCUMENTATION: 1.0,
            TaskCategory.RESEARCH: 1.3,
            TaskCategory.ANALYSIS: 1.4,
            TaskCategory.PLANNING: 1.1,
            TaskCategory.TESTING: 1.6,
            TaskCategory.DEPLOYMENT: 1.8,
            TaskCategory.GENERAL: 1.0
        }
        
        base_duration = base_durations[complexity]
        multiplier = category_multipliers.get(category, 1.0)
        
        return int(base_duration * multiplier)
    
    def _analyze_priority(self, task: str, context: Dict[str, Any]) -> int:
        """Analyze task priority (1-10)."""
        priority = 5  # Default medium priority
        
        # High priority indicators
        high_priority_words = [
            "urgent", "critical", "emergency", "asap", "immediately",
            "production", "bug", "error", "broken", "down"
        ]
        
        # Low priority indicators  
        low_priority_words = [
            "eventually", "sometime", "nice to have", "optional",
            "future", "later", "enhancement", "improvement"
        ]
        
        for word in high_priority_words:
            if word in task:
                priority += 2
                
        for word in low_priority_words:
            if word in task:
                priority -= 2
                
        # Check context for priority hints
        if context.get("deadline"):
            priority += 1
        if context.get("user_priority"):
            priority = max(priority, context["user_priority"])
            
        return max(1, min(10, priority))
    
    def _determine_agents(self, category: TaskCategory, complexity: TaskComplexity, task: str) -> tuple[AgentType, List[AgentType]]:
        """Determine primary and supporting agents."""
        # Default assignment based on category
        agent_preferences = {
            TaskCategory.CODE_GENERATION: AgentType.CLAUDE_CODE,
            TaskCategory.CODE_DEBUG: AgentType.CLAUDE_CODE,
            TaskCategory.CODE_REVIEW: AgentType.OPENAI,
            TaskCategory.DOCUMENTATION: AgentType.GEMINI,
            TaskCategory.RESEARCH: AgentType.GEMINI,
            TaskCategory.ANALYSIS: AgentType.OPENAI,
            TaskCategory.PLANNING: AgentType.GEMINI,
            TaskCategory.TESTING: AgentType.CLAUDE_CODE,
            TaskCategory.DEPLOYMENT: AgentType.CLAUDE_CODE,
            TaskCategory.GENERAL: AgentType.CLAUDE_CODE
        }
        
        primary_agent = agent_preferences.get(category, AgentType.CLAUDE_CODE)
        supporting_agents = []
        
        # Add supporting agents based on complexity and category
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            all_agents = list(AgentType)
            supporting_agents = [agent for agent in all_agents if agent != primary_agent]
        elif complexity == TaskComplexity.MODERATE:
            if category in [TaskCategory.CODE_GENERATION, TaskCategory.CODE_DEBUG]:
                supporting_agents = [AgentType.OPENAI]  # For code review
            elif category == TaskCategory.RESEARCH:
                supporting_agents = [AgentType.OPENAI]  # For analysis
        
        return primary_agent, supporting_agents
    
    def _requires_collaboration(self, category: TaskCategory, complexity: TaskComplexity, num_supporting: int) -> bool:
        """Determine if task requires collaboration."""
        # Always collaborate for critical tasks
        if complexity == TaskComplexity.CRITICAL:
            return True
            
        # Collaborate for complex tasks with multiple aspects
        if complexity == TaskComplexity.COMPLEX and num_supporting > 0:
            return True
            
        # Specific categories that benefit from collaboration
        collaborative_categories = [
            TaskCategory.CODE_REVIEW,
            TaskCategory.ANALYSIS,
            TaskCategory.PLANNING
        ]
        
        if category in collaborative_categories and complexity != TaskComplexity.TRIVIAL:
            return True
            
        return False
    
    def _extract_metadata(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from task description."""
        metadata = {}
        
        # Detect programming languages
        detected_languages = [lang for lang in self.programming_languages if lang in task]
        if detected_languages:
            metadata["languages"] = detected_languages
            
        # Detect file operations
        file_operations = ["create file", "edit file", "delete file", "read file"]
        detected_operations = [op for op in file_operations if op in task]
        if detected_operations:
            metadata["file_operations"] = detected_operations
            
        # Extract code snippets
        if self._contains_code(task):
            metadata["contains_code"] = True
            
        # Add context metadata
        metadata.update(context)
        
        return metadata
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code snippets."""
        code_indicators = [
            "```", "def ", "class ", "function", "import ", "from ",
            "if __name__", "return ", "print(", "console.log"
        ]
        
        return any(indicator in text for indicator in code_indicators)
    
    def get_task_insights(self, task_description: str) -> Dict[str, Any]:
        """Get detailed insights about a task."""
        analysis = self.analyze_task(task_description)
        
        return {
            "analysis": analysis,
            "recommendations": self._get_recommendations(analysis),
            "risks": self._identify_risks(analysis),
            "success_factors": self._identify_success_factors(analysis)
        }
    
    def _get_recommendations(self, analysis: TaskAnalysis) -> List[str]:
        """Get recommendations for task execution."""
        recommendations = []
        
        if analysis.complexity == TaskComplexity.CRITICAL:
            recommendations.append("Consider breaking down into smaller subtasks")
            recommendations.append("Implement thorough testing and validation")
            
        if analysis.requires_collaboration:
            recommendations.append("Schedule collaboration session between agents")
            recommendations.append("Define clear roles and responsibilities")
            
        if analysis.category == TaskCategory.CODE_GENERATION:
            recommendations.append("Review code quality and security")
            recommendations.append("Add comprehensive documentation")
            
        return recommendations
    
    def _identify_risks(self, analysis: TaskAnalysis) -> List[str]:
        """Identify potential risks in task execution."""
        risks = []
        
        if analysis.complexity == TaskComplexity.CRITICAL:
            risks.append("High complexity may lead to longer execution times")
            risks.append("Increased chance of errors or incomplete implementation")
            
        if not analysis.requires_collaboration and analysis.complexity == TaskComplexity.COMPLEX:
            risks.append("Single agent may miss important considerations")
            
        if analysis.estimated_duration > 1200:  # 20 minutes
            risks.append("Long execution time may impact user experience")
            
        return risks
    
    def _identify_success_factors(self, analysis: TaskAnalysis) -> List[str]:
        """Identify factors that contribute to task success."""
        factors = []
        
        if analysis.requires_collaboration:
            factors.append("Multi-agent collaboration provides diverse perspectives")
            
        if analysis.primary_agent == AgentType.CLAUDE_CODE:
            factors.append("Claude Code provides excellent implementation capabilities")
            
        if analysis.category in [TaskCategory.DOCUMENTATION, TaskCategory.RESEARCH]:
            factors.append("Task aligns well with AI agent strengths")
            
        return factors