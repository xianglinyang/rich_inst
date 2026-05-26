from __future__ import annotations

from typing import List, Tuple

from src.inst.schema import Span

_PLACEHOLDER = "[[INJECTION_SPAN]]"


def substitute_payloads(
    rewritten_text: str,
    payload_texts: List[str],
) -> Tuple[str, List[Span]]:
    """
    Substitute [[INJECTION_SPAN]] placeholders with payload texts, left-to-right.

    - If len(payload_texts) is less than the number of placeholders, remaining
      placeholders are stripped (replaced with empty string).
    - If len(payload_texts) exceeds the number of placeholders, extras are ignored.
    - With empty payload_texts, all placeholders are stripped.

    Returns (modified_text, injected_spans). Each span's char offsets refer to
    `modified_text`.
    """
    text = rewritten_text
    spans: List[Span] = []

    for i, payload in enumerate(payload_texts):
        idx = text.find(_PLACEHOLDER)
        if idx == -1:
            break
        text = text[:idx] + payload + text[idx + len(_PLACEHOLDER):]
        spans.append(Span(start=idx, end=idx + len(payload), text=payload))

    # Remove any leftover placeholders (e.g. model placed extras despite the rules)
    text = text.replace(_PLACEHOLDER, "")
    return text, spans
