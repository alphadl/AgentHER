# AgentHER: Hindsight Experience Replay for LLM Agents

<p align="center">
  <img src="assets/logo.jpg" width="220" alt="AgentHER Logo" />
</p>

<p align="center">
  <em>Turning failed agent trajectories into high-quality training data</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#usage">Usage</a> •
  <a href="#citation">Citation</a>
</p>

---

## Motivation

In LLM Agent training, **failed tool-use trajectories are routinely discarded**. This is wasteful — a trajectory that fails Goal A may perfectly succeed for Goal B.

**AgentHER** borrows the core insight from [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495) in reinforcement learning: instead of discarding failures, we **relabel the goal** to match what was actually achieved, creating valid training data from every trajectory.

### Example

| | Original (Failed) | Hindsight (Success) |
|---|---|---|
| **Prompt** | "Find copper wire **under $5/kg**" | "Find copper wire suppliers and **compare pricing**" |
| **Trajectory** | Searched 7 suppliers, best found at $5.30/kg | *(same trajectory)* |
| **Label** | ❌ Failure | ✅ Success |

The agent's work was thorough and correct — it just didn't meet an arbitrary price constraint. AgentHER recovers this data.

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────────┐
│  1. Failure      │────▸│  2. Outcome      │────▸│  3. Prompt      │────▸│  4. Data       │
│     Detector     │     │     Extractor    │     │     Relabeler   │     │     Augmenter  │
│                  │     │                  │     │                 │     │                │
│  Is this really  │     │  What did the    │     │  Reverse-       │     │  Package as    │
│  a failure?      │     │  agent achieve?  │     │  engineer a new │     │  SFT / DPO /   │
│  Recoverable?    │     │                  │     │  matching prompt│     │  ShareGPT      │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └────────────────┘
```

**Stage 1 — Failure Detector:** Validates whether the trajectory truly fails, classifies the failure type (constraint violation, wrong result, tool error, etc.), and assesses recoverability. Supports rule-based (free) or LLM-judge modes.

**Stage 2 — Outcome Extractor:** Analyzes observations to build a factual summary of what the agent *actually* accomplished, ignoring the original goal entirely.

**Stage 3 — Prompt Relabeler:** Uses an LLM to craft a natural, human-like prompt that the trajectory *perfectly satisfies*. Includes confidence scoring and retry logic.

**Stage 4 — Data Augmenter:** Packages the new (prompt, trajectory) pair into standard training formats: SFT, DPO (with chosen/rejected pairs), or ShareGPT multi-turn.

## Architecture

```
agenther/
├── models.py             # Pydantic data models (AgentStep, FailedTrajectory, etc.)
├── constants.py          # Shared thresholds (min observation length, truncation, etc.)
├── llm_client.py         # OpenAI-compatible LLM client with structured output
├── prompts.py            # Jinja2 prompt templates + steps_for_prompt()
├── failure_detector.py   # Stage 1: rule-based + LLM failure classification
├── outcome_extractor.py  # Stage 2: extract actual achievements
├── prompt_relabeler.py   # Stage 3: reverse-engineer hindsight prompts
├── data_augmenter.py     # Stage 4: SFT/DPO/ShareGPT formatting
├── pipeline.py           # End-to-end pipeline orchestrator
└── cli.py                # Command-line interface
```

## Quickstart

### Installation

```bash
# Recommended: use a virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

pip install -e .
# Optional, for running tests: pip install -e ".[dev]"
```

### Rule-Based Demo (No LLM Needed)

```bash
python examples/run_example.py --rule-based
```

### Full Pipeline

```bash
export OPENAI_API_KEY="your-key"

# Process failed trajectories → SFT data
agenther run examples/example_trajectories.json -f sft -o outputs/sft_data.jsonl

# Generate DPO pairs
agenther run examples/example_trajectories.json -f dpo -o outputs/dpo_data.jsonl

# Validate input format
agenther validate examples/example_trajectories.json
```

### Use a Custom Model / API

```bash
# With vLLM / Ollama / any OpenAI-compatible endpoint
agenther run data.json --model "llama3" --base-url "http://localhost:8000/v1"
```

## Usage

### Python API

```python
from agenther import AgentHERPipeline, PipelineConfig
from agenther.models import FailedTrajectory, AgentStep, OutputFormat

# Define a failed trajectory
trajectory = FailedTrajectory(
    original_prompt="Find flights to Tokyo under $500",
    steps=[
        AgentStep(
            thought="Searching for flights",
            action_name="flight_search",
            action_input={"destination": "Tokyo", "max_price": 500},
            observation="Found: ANA $680, JAL $720, United $590",
        ),
    ],
    final_answer="No flights under $500 found.",
    failure_reason="All flights exceed $500 budget",
)

# Run the pipeline
config = PipelineConfig(model="gpt-4o", output_format=OutputFormat.SFT)
pipeline = AgentHERPipeline(config)
result = pipeline.process(trajectory)

if result.success:
    print(f"Hindsight prompt: {result.relabeled.hindsight_prompt}")
    # e.g., "Search for flights to Tokyo and compare prices across airlines"
```

### Input Data Format

Provide failed trajectories as JSON or JSONL. `steps` must contain at least one step.

```json
{
  "trajectory_id": "optional_id",
  "original_prompt": "The user's original request",
  "steps": [
    {
      "thought": "Agent's reasoning",
      "action_name": "tool_name",
      "action_input": {"key": "value"},
      "observation": "Tool output"
    }
  ],
  "final_answer": "Agent's final response",
  "failure_reason": "Why this is considered a failure"
}
```

## Configuration

CLI options override defaults; there is no config file loading. For reference, [`configs/default.yaml`](configs/default.yaml) documents the same options (use it as a template; pass values via CLI or `PipelineConfig` in code):

```yaml
llm:
  model: "gpt-4o"
  temperature: 0.3

pipeline:
  use_llm_detector: false    # Rule-based is faster and free
  use_llm_extractor: true    # LLM gives better outcome extraction
  output_format: "sft"       # sft | dpo | sharegpt
  min_confidence: 0.5        # Quality threshold for relabeling
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Limitations

- **Batch processing is sequential** — no parallelism; large batches may be slow.
- **No config file** — options are passed via CLI or `PipelineConfig` in code.
- **Rule-based stages are heuristics** — for best quality, use LLM for detector/extractor when cost allows.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/alphadl/AgentHER).

## Related projects

- **[AdaRubrics](https://github.com/alphadl/AdaRubrics)** — Adaptive dynamic rubric evaluator for agent trajectories: generates task-specific dimensions and scores runs for filtering/RLHF. Use AdaRubric to score or filter relabeled data from AgentHER.
- **[AgentSynth](https://github.com/alphadl/AgentSynth)** — Synthetic agent data pipeline (forward + back-translation, execution-based reject sampling). AgentHER can relabel failed or low-quality synthetic runs into valid SFT/DPO data.
- **[trajectory_tokenization](https://github.com/alphadl/trajectory_tokenization)** — ReAct with trajectory tokenization: compresses long (Thought, Action, Observation) history so long-horizon runs fit in context. Addresses context length; AgentHER addresses *reuse* of failed trajectories.

## Citation

```bibtex
@software{agenther2025,
  title   = {AgentHER: Hindsight Experience Replay for LLM Agents},
  author  = {Ding, Liang},
  year    = {2025},
  url     = {https://github.com/alphadl/AgentHER},
}
```

## License

Apache 2.0
