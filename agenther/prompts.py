"""Prompt templates for each AgentHER pipeline stage.

Centralized prompt management using Jinja2 templates. Each prompt is designed
to be robust against diverse agent trajectory formats and failure modes.
"""

from __future__ import annotations

from jinja2 import Template

# ---------------------------------------------------------------------------
# Failure Detector
# ---------------------------------------------------------------------------

FAILURE_DETECTION_SYSTEM = """\
You are an expert evaluator of LLM agent trajectories. Your task is to determine \
whether an agent trajectory represents a FAILURE relative to the user's original intent.

A trajectory is a FAILURE if the agent:
- Did not satisfy key constraints in the user prompt (price, quantity, format, etc.)
- Returned incorrect or fabricated information
- Failed to complete the task (got stuck, gave up, or went off-topic)
- Encountered irrecoverable tool errors

A trajectory is NOT a failure if:
- It completed the task approximately correctly with minor deviations
- The user prompt was ambiguous and the agent made a reasonable interpretation

Classify the failure type as one of:
  constraint_violation, wrong_result, incomplete, tool_error, hallucination, off_topic

Also assess whether the trajectory is RECOVERABLE — meaning hindsight relabeling \
could turn it into valid training data. A trajectory with some useful observations \
is recoverable; a trajectory that just crashed with no output is not."""

FAILURE_DETECTION_USER = Template("""\
## Original User Prompt
{{ original_prompt }}

## Agent Trajectory ({{ num_steps }} steps)
{% for step in steps %}
### Step {{ loop.index }}
- **Thought:** {{ step.thought }}
- **Action:** {{ step.action_name }}({{ step.action_input }})
- **Observation:** {{ step.observation }}
{% endfor %}

## Agent's Final Answer
{{ final_answer }}

## Stated Failure Reason
{{ failure_reason }}

Analyze this trajectory and respond with the structured JSON.""")

# ---------------------------------------------------------------------------
# Outcome Extractor
# ---------------------------------------------------------------------------

OUTCOME_EXTRACTION_SYSTEM = """\
You are an expert at analyzing LLM agent execution traces. Your job is to extract \
a factual summary of what the agent ACTUALLY achieved, regardless of what it was \
asked to do.

Focus on:
1. Concrete facts, data points, or results the agent discovered
2. Tools that were successfully called and their outputs
3. Information gathered, even if incomplete or not matching the original goal
4. Any partial progress toward any objective

Be STRICTLY factual. Only list things that are evidenced by the observations. \
Do NOT infer or extrapolate beyond what the trajectory shows."""

OUTCOME_EXTRACTION_USER = Template("""\
## Original User Prompt
{{ original_prompt }}

## Full Trajectory
{% for step in steps %}
### Step {{ loop.index }}
- **Action:** {{ step.action_name }}({{ step.action_input }})
- **Observation:** {{ step.observation }}
{% endfor %}

## Final Answer (if any)
{{ final_answer }}

Extract the actual achievements and key observations from this trajectory.""")

# ---------------------------------------------------------------------------
# Prompt Relabeler
# ---------------------------------------------------------------------------

PROMPT_RELABEL_SYSTEM = """\
You are a creative prompt engineer. Given a summary of what an LLM agent actually \
achieved during a task, you must write a NEW user prompt that would make the agent's \
trajectory look like a PERFECT, SUCCESSFUL execution.

Requirements for the hindsight prompt:
1. It must be natural — something a real user might actually ask
2. It must be specific enough that the trajectory clearly satisfies it
3. It must NOT reference the original failed prompt or mention "hindsight"
4. It should match the complexity/style of the original prompt
5. All constraints in the new prompt must be satisfied by the trajectory's outcomes

Also assess your confidence (0.0–1.0) that this relabeling produces a genuinely \
useful training sample. Low confidence if the trajectory is too noisy or the \
achievements are trivial."""

PROMPT_RELABEL_USER = Template("""\
## What the Agent Actually Achieved
{% for a in achievements %}
- {{ a }}
{% endfor %}

## Key Observations from the Trajectory
{% for o in observations %}
- {{ o }}
{% endfor %}

## Original Prompt (for style/complexity reference only — do NOT reuse it)
{{ original_prompt }}

## Number of Steps in Trajectory
{{ num_steps }}

Write a new user prompt and provide your rationale.""")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM = """\
You are a quality assurance expert for LLM training data. Given a hindsight prompt \
and the corresponding agent trajectory, verify that the trajectory genuinely \
satisfies the hindsight prompt.

Check for:
1. Does the trajectory actually achieve what the hindsight prompt asks?
2. Are there any contradictions between the prompt and the observations?
3. Is the hindsight prompt realistic and not overly trivial?

Respond with a pass/fail judgment and a quality score from 0.0 to 1.0."""

VALIDATION_USER = Template("""\
## Hindsight Prompt
{{ hindsight_prompt }}

## Agent Trajectory
{% for step in steps %}
### Step {{ loop.index }}
- **Action:** {{ step.action_name }}({{ step.action_input }})
- **Observation:** {{ step.observation }}
{% endfor %}

## Final Answer
{{ final_answer }}

Validate whether this trajectory satisfies the hindsight prompt.""")
