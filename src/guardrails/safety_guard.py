from dataclasses import dataclass, field
from typing import List


@dataclass
class GuardDecision:
    blocked: bool
    valid: bool
    category: str
    confidence: float
    reason: str


@dataclass
class JudgeDecision:
    """Rich structured output from an LLM-as-a-judge classifier."""
    contains_instruction_like_span: bool
    contains_misaligned_external_instruction: bool
    expected_guard_action: str  # pass | sanitize | block | ask
    risky_span_summary: str

    def to_guard_decision(self) -> GuardDecision:
        blocked = self.expected_guard_action in ("sanitize", "block")
        confidence = (
            1.0 if self.expected_guard_action == "block" else
            0.7 if self.expected_guard_action == "sanitize" else
            0.5 if self.expected_guard_action == "ask" else
            0.0
        )
        category = (
            "prompt_injection" if self.contains_misaligned_external_instruction else
            "suspicious"        if self.contains_instruction_like_span else
            "benign"
        )
        return GuardDecision(
            blocked=blocked,
            valid=True,
            category=category,
            confidence=confidence,
            reason=self.risky_span_summary or "No risky content detected",
        )


@dataclass
class DetectedSpan:
    """A single injected text fragment identified by a sanitizer."""
    text: str    # verbatim span from the original document
    start: int   # character offset in the original text (-1 if not locatable)
    end: int     # character offset in the original text (-1 if not locatable)
    reason: str  # why this span is considered injected


@dataclass
class SanitizerDecision:
    """Unified output for all sanitizer methods: decision + spans + cleaned text."""
    is_injection: bool
    category: str                            # "benign" | "prompt_injection"
    confidence: float                        # 0.0–1.0
    detected_spans: List[DetectedSpan] = field(default_factory=list)
    sanitized_text: str = ""                 # original text with injections removed
    reason: str = ""                         # overall explanation

    def to_guard_decision(self) -> GuardDecision:
        return GuardDecision(
            blocked=self.is_injection,
            valid=True,
            category=self.category,
            confidence=self.confidence,
            reason=self.reason or (
                f"{len(self.detected_spans)} injected span(s) detected"
                if self.is_injection else "No injection detected"
            ),
        )