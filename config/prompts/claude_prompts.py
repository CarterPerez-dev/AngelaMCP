"""
Specialized prompts for Claude Code agent operations.

These prompts are tailored for Claude's file system and execution capabilities.
"""

CLAUDE_CODE_GENERATION_PROMPT = """As the senior developer in this collaboration, please generate a complete, working implementation for the following task:

**Task**: {task_description}

**Requirements**:
{requirements}

**Constraints**:
{constraints}

Please provide:

1. **Architecture Overview**
   - High-level design decisions
   - Technology stack choices
   - Key components and their relationships

2. **File Structure**
   - Complete directory layout
   - Purpose of each file/directory
   - Dependencies and configurations

3. **Implementation**
   - Working code for all components
   - Proper error handling
   - Comprehensive comments
   - Type hints where appropriate

4. **Testing Strategy**
   - Unit tests for key functions
   - Integration tests for workflows
   - Example usage and edge cases

5. **Documentation**
   - README with setup instructions
   - API documentation
   - Deployment guide

6. **Quality Assurance**
   - Code follows best practices
   - Security considerations addressed
   - Performance optimization included
   - Maintainability features

Create actual, executable files that solve the problem completely. Focus on production-ready code that can be deployed immediately."""

CLAUDE_REVIEW_PROMPT = """As the senior developer, please review the following solution and provide implementation feedback:

**Original Task**: {task_description}

**Proposed Solution**: {solution}

**Review Focus Areas**:
1. **Implementation Feasibility**
   - Can this actually be built as described?
   - Are there any technical impossibilities?
   - What are the implementation challenges?

2. **Architecture Quality**
   - Is the architecture sound?
   - Are there better structural approaches?
   - How scalable is this design?

3. **Code Quality** (if code provided)
   - Syntax and logic correctness
   - Error handling adequacy
   - Performance implications
   - Security considerations

4. **Completeness**
   - Does it fully address the requirements?
   - What's missing?
   - Are edge cases handled?

5. **Implementation Plan**
   - What would you do differently?
   - What are the next steps?
   - How long would this take to implement?

Provide specific, actionable feedback that will improve the final implementation."""

CLAUDE_PROJECT_PROMPT = """As the senior developer, please create a complete project for the following requirements:

**Project**: {project_description}

**Scope**: {scope}

**Technical Requirements**: {tech_requirements}

Create a production-ready project including:

1. **Project Setup**
   ```
   Create proper project structure
   Set up dependency management
   Configure development environment
   Add necessary configuration files
   ```

2. **Core Implementation**
   ```
   Implement all required features
   Add proper error handling
   Include logging and monitoring
   Ensure security best practices
   ```

3. **Quality Assurance**
   ```
   Write comprehensive tests
   Add code documentation
   Include type hints
   Set up linting and formatting
   ```

4. **Deployment Preparation**
   ```
   Create Docker configuration
   Add environment management
   Include deployment scripts
   Set up CI/CD pipeline basics
   ```

5. **Documentation**
   ```
   Write detailed README
   Add API documentation
   Include architecture diagrams
   Create user guides
   ```

This should be a complete, professional-grade project that can be used in production immediately."""
