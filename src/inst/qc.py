from __future__ import annotations

import re
from typing import List

from src.inst.schema import RewriteResult

_PLACEHOLDER = "[[INJECTION_SPAN]]"

# Maps each canonical position to (region_key, sub_band) where sub_band is
# the allowed fractional range inside that region. For begin_context /
# end_context the whole region is acceptable. For *_inst positions we also
# require the placeholder to be in the right sub-segment of the inst region.
_POSITION_REGION = {
    "begin_context": ("context_pre",  (0.0, 1.0)),
    "begin_inst":    ("inst",         (0.0, 0.3)),
    "middle_inst":   ("inst",         (0.25, 0.75)),
    "end_inst":      ("inst",         (0.7, 1.0)),
    "end_context":   ("context_post", (0.0, 1.0)),
}

_MIN_LENGTH_RATIO = 0.4
_MAX_LENGTH_RATIO_BASE = 3.0
_MAX_LENGTH_RATIO_PER_BENIGN = 0.3

_BENIGN_LOW_RATIO = 0.4
_BENIGN_HIGH_RATIO = 3.0

_BENIGN_CUE_PATTERNS = [
    r"\bcheck\b", r"\bverify\b", r"\bcompare\b", r"\bensure\b",
    r"\bmake sure\b", r"\bconfirm\b", r"\bavoid\b", r"\bnote that\b",
    r"\bfirst\b", r"\bthen\b", r"\bif\b.*\bthen\b",
    r"\buse\b", r"\bdo not\b", r"\bdon't\b", r"\breview\b",
    r"\bidentify\b", r"\blook for\b", r"\bremember to\b",
]
_BENIGN_CUE_RE = re.compile("|".join(_BENIGN_CUE_PATTERNS), re.IGNORECASE)

_APOSTROPHE_TRANS = str.maketrans({
    "’": "'",
    "‘": "'",
    "`": "'",
    "“": '"',
    "”": '"',
})

_REFUSAL_PATTERNS = (
    ("refusal", re.compile(
        r"\bi\s+(?:can'?t|cannot|won'?t|will not|am unable|am not able)\s+"
        r"(?:assist|help|comply|provide|rewrite|rephrase|embed|create)\b"
    )),
    ("refusal", re.compile(
        r"\b(?:can'?t|cannot|unable to|not able to)\s+"
        r"(?:assist|comply|help|provide|rewrite|rephrase|embed|create)\b"
    )),
    ("refusal", re.compile(r"\b(?:i'?m sorry|i am sorry|i apologize|as an ai)\b")),
    ("refusal", re.compile(r"\bi can,\s*however\b")),
    ("safe_alternative", re.compile(
        r"\b(?:safe|harmless|benign)\s+(?:debugging\s+)?"
        r"(?:alternative|placeholder|faq entry|sample)\b"
    )),
    ("safe_alternative", re.compile(r"\blocal or internal test endpoint\b")),
)

_LEFT_BOUNDARY_CHARS = set("\n\r.!?:;([{ \"'-*")
_RIGHT_BOUNDARY_CHARS = set("\n\r.!?,;:)]} \"'")


def count_benign_cues(text: str) -> int:
    return len(_BENIGN_CUE_RE.findall(text))


def _normalize_for_qc(text: str) -> str:
    text = text.translate(_APOSTROPHE_TRANS).lower()
    return re.sub(r"\s+", " ", text).strip()


def find_refusal_artifact(text: str) -> str | None:
    """Return a QC note if text looks like a refusal or safe-substitute response."""
    normalized = _normalize_for_qc(text)
    if not normalized:
        return None
    for label, pattern in _REFUSAL_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return f"{label}: contains {match.group(0)!r}"
    return None


def _placeholder_indices(text: str) -> List[int]:
    indices: List[int] = []
    start = 0
    while True:
        idx = text.find(_PLACEHOLDER, start)
        if idx == -1:
            return indices
        indices.append(idx)
        start = idx + len(_PLACEHOLDER)


def _previous_boundary_char(text: str, idx: int) -> str:
    i = idx - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return text[i] if i >= 0 else ""


def _next_boundary_char(text: str, idx: int) -> str:
    i = idx
    while i < len(text) and text[i] in " \t":
        i += 1
    return text[i] if i < len(text) else ""


def _boundary_snippet(text: str, idx: int) -> str:
    start = max(0, idx - 45)
    end = min(len(text), idx + len(_PLACEHOLDER) + 45)
    return text[start:end].replace("\n", "\\n")


def _check_placeholder_boundaries(text: str, notes: List[str]) -> None:
    """
    Catch placeholders that will become run-on prose after substitution.

    Good examples:
      [[INJECTION_SPAN]].
      [[INJECTION_SPAN]]
      Check the next item...
    Bad example:
      [[INJECTION_SPAN]] Check the next item...
    """
    for idx in _placeholder_indices(text):
        prev_char = _previous_boundary_char(text, idx)
        next_char = _next_boundary_char(text, idx + len(_PLACEHOLDER))

        if prev_char and prev_char not in _LEFT_BOUNDARY_CHARS:
            notes.append(
                "placeholder_boundary: placeholder attached to preceding prose near "
                f"{_boundary_snippet(text, idx)!r}"
            )
        if next_char and next_char not in _RIGHT_BOUNDARY_CHARS:
            notes.append(
                "placeholder_boundary: placeholder followed by prose without punctuation "
                f"or newline near {_boundary_snippet(text, idx)!r}"
            )


def _check_region_membership(result: RewriteResult, notes: List[str]) -> None:
    """For inst styles: verify placeholder lies inside the requested region+sub-band."""
    text = result.rewritten_text
    idx = text.find(_PLACEHOLDER)
    if idx == -1:
        notes.append("region_check: placeholder not found in text")
        return

    if result.position not in _POSITION_REGION:
        notes.append(f"region_check: unknown position {result.position!r}")
        return

    region_key, (sub_lo, sub_hi) = _POSITION_REGION[result.position]
    if region_key not in result.region_offsets:
        notes.append(
            f"region_check: region {region_key!r} not present in rewrite "
            f"(found regions: {list(result.region_offsets)})"
        )
        return

    r_start, r_end = result.region_offsets[region_key]
    region_len = max(1, r_end - r_start)

    if not (r_start <= idx < r_end):
        notes.append(
            f"region_check: placeholder at char {idx} outside region {region_key!r} "
            f"[{r_start}, {r_end})"
        )
        return

    sub_offset = (idx - r_start) / region_len
    if not (sub_lo <= sub_offset <= sub_hi):
        notes.append(
            f"region_check: placeholder in correct region {region_key!r} but at "
            f"sub-offset {sub_offset:.2f}, expected [{sub_lo}, {sub_hi}] for "
            f"position={result.position!r}"
        )


def _check_narrative_position(result: RewriteResult, notes: List[str]) -> None:
    """For narrative style: verify placeholder lies in the requested half of the document."""
    text = result.rewritten_text
    idx = text.find(_PLACEHOLDER)
    if idx == -1 or not text:
        notes.append("narrative_position: placeholder not found")
        return
    offset = idx / len(text)
    bands = {"begin_context": (0.0, 0.35), "end_context": (0.65, 1.0)}
    if result.position not in bands:
        notes.append(f"narrative_position: invalid position {result.position!r} for narrative")
        return
    lo, hi = bands[result.position]
    if not (lo <= offset <= hi):
        notes.append(
            f"narrative_position: placeholder offset {offset:.2f} outside [{lo}, {hi}] "
            f"for position={result.position!r}"
        )


def run_qc(result: RewriteResult, original_context: str) -> RewriteResult:
    """
    Run deterministic QC checks. Sets qc_passed and qc_notes on `result`.

    Checks:
    1. Placeholder count matches injection_frequency.
    2. Placeholder has clean sentence/line boundaries.
    3. Region membership (inst styles) or position band (narrative style).
    4. Length ratio within [0.4×, (3.0 + 0.3*benign_frequency)×] of original.
    5. self_check flags all True.
    6. Benign cue count within [0.4×, 3.0×] of benign_frequency (skipped for narrative).
    7. Structure presence: inst styles must have all three regions (context_pre, inst, context_post).
    8. Refusal/safe-alternative artifacts are absent.
    """
    notes: List[str] = list(result.qc_notes)
    text = result.rewritten_text
    is_narrative = (result.style == "narrative")

    # 1. Placeholder count
    actual_count = text.count(_PLACEHOLDER)
    if actual_count != result.injection_frequency:
        notes.append(
            f"placeholder_count: expected {result.injection_frequency}, found {actual_count}"
        )

    # 2. Placeholder boundary check
    if actual_count > 0:
        _check_placeholder_boundaries(text, notes)

    # 3. Position / region check (only when there's a placeholder)
    if result.injection_frequency >= 1 and text:
        if is_narrative:
            _check_narrative_position(result, notes)
        else:
            _check_region_membership(result, notes)

    # 4. Length ratio (scaled by requested benign density)
    if original_context:
        ratio = len(text) / len(original_context)
        max_ratio = _MAX_LENGTH_RATIO_BASE + result.benign_frequency * _MAX_LENGTH_RATIO_PER_BENIGN
        if ratio < _MIN_LENGTH_RATIO or ratio > max_ratio:
            notes.append(
                f"length_ratio: {ratio:.2f} outside [{_MIN_LENGTH_RATIO}, {max_ratio:.2f}] "
                f"(benign_frequency={result.benign_frequency})"
            )

    # 5. Self-check flags
    sc = result.self_check
    if not sc.facts_preserved:
        notes.append("self_check: facts_preserved=False")
    if result.injection_frequency > 0 and not sc.placeholder_count_correct:
        notes.append("self_check: placeholder_count_correct=False")
    if not sc.no_new_unsafe_instruction_added:
        notes.append("self_check: no_new_unsafe_instruction_added=False")

    # 6. Benign cue count — inst styles only
    if not is_narrative and result.benign_frequency > 0 and text:
        actual_cues = count_benign_cues(text)
        lo_cues = max(1, int(result.benign_frequency * _BENIGN_LOW_RATIO))
        hi_cues = int(result.benign_frequency * _BENIGN_HIGH_RATIO)
        if not (lo_cues <= actual_cues <= hi_cues):
            notes.append(
                f"benign_density: cue count {actual_cues} outside [{lo_cues}, {hi_cues}] "
                f"for benign_frequency={result.benign_frequency}"
            )

    # 7. Structure presence — inst styles must have all three regions
    if not is_narrative and result.injection_frequency > 0:
        missing = [
            k for k in ("context_pre", "inst", "context_post")
            if k not in result.region_offsets
        ]
        if missing:
            notes.append(
                f"structure: missing regions {missing} "
                f"(found: {list(result.region_offsets)})"
            )

    # 8. Refusal/safe-alternative artifacts
    refusal_note = find_refusal_artifact(text)
    if refusal_note:
        notes.append(refusal_note)

    result.qc_notes = notes
    result.qc_passed = len(notes) == 0
    return result
