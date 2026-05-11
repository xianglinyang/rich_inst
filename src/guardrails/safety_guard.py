from dataclasses import dataclass


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