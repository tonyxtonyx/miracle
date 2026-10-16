"""Diagnosis configuration. Every value is recorded in the run manifest; none is tuned per task."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiagnosisConfig:
    # global budgets
    max_iterations: int = 8                # select -> investigate -> review cycles
    max_hypotheses: int = 12
    max_depth: int = 3                     # refinement depth
    max_graph_size: int = 80               # hypotheses + evidence + probes: forces synthesis
    min_initial_hypotheses: int = 2
    max_initial_hypotheses: int = 5        # planning-phase cap: the graph must not balloon before any evidence exists
    # per-call step budgets
    hypothesize_max_steps: int = 12
    review_max_steps: int = 10
    synthesis_max_steps: int = 10
    branch_max_steps: int = 24
    implementation_max_steps: int = 50
    # synthesis triggers
    stall_iterations: int = 2
    repeated_action_limit: int = 3
    support_threshold: str = "low"         # below this ordinal level a hypothesis does not count as viable
    # validation
    duplicate_similarity: float = 0.88
    min_text_chars: int = 12
    max_excerpt_lines: int = 80
    # decoding (the S0/S1/S2 baseline model and settings)
    model: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.0
    reasoning_effort: str = "none"
    controller_max_tokens: int = 3000
    investigator_max_tokens: int = 4096

    def to_dict(self) -> dict:
        return asdict(self)
