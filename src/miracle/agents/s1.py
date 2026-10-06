"""S1 = S0 + one fix: a response cut off by the output-token cap is not "the work is done".

S0 treated any reply without a tool call as final, so when the model was truncated mid-analysis
(finish_reason == "length") the run ended with no edit (2 of 10 dev10 runs). S1 keeps the truncated
reply in the conversation and asks the model to continue. Nothing else differs from S0: same model,
decoding, system prompt, tools, step budget (nudged turns count as steps). Do not add other changes here:
S0 vs S1 is only a controlled comparison while this is the single difference.
"""
from __future__ import annotations

from .s0 import S0Runtime

LENGTH_NUDGE = ("Your previous response was truncated by the output token limit. "
                "Continue the investigation from where you stopped. Use the available tools when appropriate.")


class S1Runtime(S0Runtime):
    def __init__(self, *args, length_nudge: str = LENGTH_NUDGE, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_nudge = length_nudge
        self.name = "s1-" + (self.client.config.model or "model").split("/")[-1].lower()

    def describe(self) -> dict:
        d = super().describe()
        d.update({"class": "S1Runtime", "length_nudge": self.length_nudge,
                  "strategy": "S1: S0 + continue (not stop) on finish_reason=length"})
        return d
