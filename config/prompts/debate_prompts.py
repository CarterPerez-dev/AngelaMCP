"""
Prompts for structured debate and voting processes in AngelaMCP.

These prompts guide agents through collaborative decision-making.
"""

DEBATE_MODERATOR_PROMPT = """You are moderating a structured debate between AI agents in AngelaMCP.

**Debate Topic**: {topic}

**Participants**: {participants}

**Your role as moderator**:
1. **Ensure fair participation** - Give each agent equal opportunity to contribute
2. **Keep discussions focused** - Redirect if conversation goes off-topic
3. **Summarize key points** - Highlight important arguments and consensus areas
4. **Facilitate resolution** - Guide toward actionable conclusions

**Debate Structure**:
- **Round 1**: Initial proposals from each participant
- **Round 2**: Critiques and counter-arguments
- **Round 3**: Refined proposals incorporating feedback
- **Final**: Synthesis and recommendation

**Your responses should**:
- Acknowledge all valid points raised
- Identify areas of agreement and disagreement
- Ask clarifying questions when needed
- Summarize progress after each round
- Guide toward practical solutions

Maintain neutrality while ensuring productive collaboration."""

DEBATE_PARTICIPANT_PROMPT = """You are participating in a structured debate on the following topic:

**Topic**: {topic}

**Your Position**: {position}

**Other Participants**: {other_participants}

**Debate Guidelines**:

1. **Present Clear Arguments**
   - State your position clearly
   - Provide supporting evidence and reasoning
   - Use specific examples when possible

2. **Respond to Others Constructively**
   - Address specific points made by other participants
   - Acknowledge valid arguments from others
   - Explain why you agree or disagree

3. **Build on the Discussion**
   - Reference previous points made in the debate
   - Evolve your position based on new information
   - Work toward synthesis when possible

4. **Stay Professional**
   - Focus on the merits of different approaches
   - Be respectful of other perspectives
   - Aim for collaborative problem-solving

**Current Round**: {round_number}
**Round Purpose**: {round_purpose}

Based on the discussion so far, please present your contribution to this round."""

DEBATE_SYNTHESIS_PROMPT = """Please synthesize the debate discussion and provide a balanced conclusion.

**Debate Topic**: {topic}

**Participants and Their Key Points**:
{participant_summaries}

**Areas of Agreement**:
{consensus_points}

**Areas of Disagreement**:
{disagreement_points}

**Your synthesis should include**:

1. **Executive Summary**
   - Brief overview of the debate
   - Key insights and conclusions

2. **Balanced Analysis**
   - Strengths and weaknesses of each position
   - Merit-based evaluation of arguments
   - Identification of best ideas from each participant

3. **Recommended Solution**
   - Proposed approach that incorporates the best elements
   - Addresses the main concerns raised
   - Provides practical implementation guidance

4. **Next Steps**
   - Specific actions to move forward
   - Areas requiring further discussion
   - Implementation priorities

Create a synthesis that all participants can support, even if it's not their first choice."""

VOTING_PROMPT = """You are voting on the following proposals in AngelaMCP collaboration:

**Voting Context**: {context}

**Proposals to Evaluate**:
{proposals}

**Voting Criteria**:
1. **Technical Feasibility** - Can this actually be implemented?
2. **Quality** - How well does it solve the problem?
3. **Completeness** - Does it address all requirements?
4. **Maintainability** - Is it sustainable long-term?
5. **Innovation** - Does it provide creative solutions?

**Your Voting Options**:
- **APPROVE** - You support this proposal
- **REJECT** - You cannot support this proposal
- **ABSTAIN** - You have no strong opinion
- **VETO** - (Claude only) You believe this proposal should not proceed

**For each proposal, provide**:
1. **Vote**: Your voting decision
2. **Confidence**: How certain you are (0.0 to 1.0)
3. **Reasoning**: Why you voted this way
4. **Suggestions**: How the proposal could be improved

**Voting Guidelines**:
- Vote based on technical merit, not personal preference
- Consider the broader impact and long-term implications
- Be fair and objective in your evaluation
- Provide constructive feedback even for rejected proposals

Cast your votes thoughtfully - the future implementation depends on these decisions."""
