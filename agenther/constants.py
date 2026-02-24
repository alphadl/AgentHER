"""Shared constants for AgentHER pipeline stages.

Centralizes magic numbers so thresholds are documented and easy to tune.
"""

# Failure detector: minimum observation length (chars) to consider trajectory recoverable
MIN_OBS_LEN_RECOVERABLE = 20

# Outcome extractor: skip observations shorter than this (chars)
MIN_OBS_LEN_EXTRACT = 15
# Max length for observation summary in achievement text
TRUNCATE_LEN = 200
# Max numeric values to include in key_observations per step
MAX_NUMERIC_VALUES_PER_STEP = 5

# Data augmenter: max observation chars in assistant response preview
OBS_PREVIEW_LEN = 300
