"""
Prompt templates and configuration for AngelaMCP agents.
Centralizes all prompt templates used across the multi-agent system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import settings

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent

def load_prompt_template(template_name: str, category: str = "debate") -> Optional[str]:
    """Load a prompt template from YAML files."""
    template_file = PROMPTS_DIR / f"{category}.yaml"
    
    if not template_file.exists():
        return None
        
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = yaml.safe_load(f)
        
    return templates.get(category, {}).get(template_name)

def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided variables."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")

class PromptTemplates:
    """Central repository for all prompt templates."""
    
    @staticmethod
    def get_debate_initial_proposal(specialty: str, task_description: str, constraints: str = "") -> str:
        """Get the initial proposal prompt for debate mode."""
        template = load_prompt_template("initial_proposal", "debate")
        if not template:
            return f"As an AI agent specializing in {specialty}, analyze and provide your solution for: {task_description}"
        
        return format_prompt(template, 
                           specialty=specialty,
                           task_description=task_description,
                           constraints=constraints)
    
    @staticmethod
    def get_debate_critique(agent_name: str, task_description: str, solution: str) -> str:
        """Get the critique prompt for debate mode."""
        template = load_prompt_template("critique", "debate")
        if not template:
            return f"Review and critique this solution from {agent_name}: {solution}"
        
        return format_prompt(template,
                           agent_name=agent_name,
                           task_description=task_description,
                           solution=solution)
    
    @staticmethod
    def get_debate_rebuttal(original_solution: str, critiques: str) -> str:
        """Get the rebuttal prompt for debate mode."""
        template = load_prompt_template("rebuttal", "debate")
        if not template:
            return f"Address these critiques of your solution:\nOriginal: {original_solution}\nCritiques: {critiques}"
        
        return format_prompt(template,
                           original_solution=original_solution,
                           critiques=critiques)
    
    @staticmethod
    def get_debate_final_proposal(task_description: str, debate_summary: str) -> str:
        """Get the final proposal prompt for debate mode."""
        template = load_prompt_template("final_proposal", "debate")
        if not template:
            return f"Provide your final solution for: {task_description}\nConsidering: {debate_summary}"
        
        return format_prompt(template,
                           task_description=task_description,
                           debate_summary=debate_summary)

# Convenience instance
prompts = PromptTemplates()