from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Canonical position vocabulary.
# - begin_context : in the leading <<<CONTEXT>>> region (prelude)
# - begin_inst    : at start of the <<<INST>>> region
# - middle_inst   : middle of the <<<INST>>> region
# - end_inst      : end of the <<<INST>>> region
# - end_context   : in the trailing <<<CONTEXT>>> region (postlude)
# - none          : no injection
POSITIONS = (
    "begin_context", "begin_inst", "middle_inst", "end_inst", "end_context", "none",
)


@dataclass
class Span:
    start: int
    end: int
    text: str


@dataclass
class SelfCheck:
    facts_preserved: bool
    placeholder_count_correct: bool
    no_new_unsafe_instruction_added: bool
    raw: dict = field(default_factory=dict)


@dataclass
class RewriteResult:
    """Output of a single rewrite call. Pre-injection; placeholders still present."""
    rewritten_text: str
    style: str
    position: str
    injection_frequency: int
    benign_frequency: int
    self_check: SelfCheck
    rewriter_model: str
    region_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    qc_passed: bool = False
    qc_notes: List[str] = field(default_factory=list)


@dataclass
class ModifiedContext:
    """
    Final transformation result. Caller-facing.

    `original_text` and `modified_text` are the two strings the downstream
    utility / ASR evaluator needs. Everything else is provenance.
    """
    original_text: str
    modified_text: str
    style: str
    position: str
    injection_frequency: int
    benign_frequency: int
    injected_spans: List[Span]
    binary_label: int        # 0 = benign, 1 = injected
    rewriter_model: str
    qc_passed: bool
    qc_notes: List[str] = field(default_factory=list)
    rewritten_text: str = ""    # tags-stripped rewrite before payload substitution
    region_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "modified_text": self.modified_text,
            "style": self.style,
            "position": self.position,
            "injection_frequency": self.injection_frequency,
            "benign_frequency": self.benign_frequency,
            "injected_spans": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.injected_spans
            ],
            "binary_label": self.binary_label,
            "rewriter_model": self.rewriter_model,
            "qc_passed": self.qc_passed,
            "qc_notes": self.qc_notes,
            "rewritten_text": self.rewritten_text,
            "region_offsets": {k: list(v) for k, v in self.region_offsets.items()},
        }
