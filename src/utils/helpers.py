"""
Utility helper functions for AngelaMCP.

Common utility functions used throughout the application.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from agent, handling various formats."""
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON-like content
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, response)
    
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def clean_agent_response(response: str) -> str:
    """Clean agent response by removing markdown and extra whitespace."""
    # Remove markdown code blocks
    response = re.sub(r'```[^`]*```', '', response, flags=re.DOTALL)
    
    # Remove excessive whitespace
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = response.strip()
    
    return response


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from text."""
    code_blocks = []
    pattern = r'```(\w+)?\n(.*?)\n```'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        language = match.group(1) or "text"
        code = match.group(2)
        
        code_blocks.append({
            "language": language,
            "code": code.strip()
        })
    
    return code_blocks


def validate_file_path(path: Union[str, Path]) -> bool:
    """Validate if file path is safe and accessible."""
    try:
        path = Path(path).resolve()
        
        # Check if path is within allowed directories
        # This is a basic security check
        cwd = Path.cwd().resolve()
        
        try:
            path.relative_to(cwd)
            return True
        except ValueError:
            return False
            
    except Exception:
        return False


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity between two strings."""
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "Free"
    elif cost < 0.001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def safe_filename(filename: str) -> str:
    """Convert string to safe filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '', filename)
    safe = re.sub(r'[-\s]+', '-', safe)
    return safe.strip('-')


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
