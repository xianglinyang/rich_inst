from __future__ import annotations

import json
import logging
import re
from typing import Dict, Literal, Optional, Tuple

from src.inst.prompt_templates import (
    PROMPTS,
    build_prompt,
    valid_positions_for,
)
from src.inst.schema import RewriteResult, SelfCheck
from src.llm_zoo import load_model
from src.llm_zoo.base_model import BaseLLM

logger = logging.getLogger(__name__)

Position = Literal[
    "begin_context", "begin_inst", "middle_inst", "end_inst", "end_context",
]

_PLACEHOLDER = "[[INJECTION_SPAN]]"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Matches a region tag possibly followed by spaces and an optional newline.
_TAG_RE = re.compile(r"<<<(CONTEXT|INST)>>>[ \t]*\n?")


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _parse_regions(text: str) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Parse <<<CONTEXT>>> / <<<INST>>> tags out of `text` and return
    (cleaned_text, region_offsets).

    Region keys: "context_pre", "inst", "context_post" when the structure is
    exactly [CONTEXT, INST, CONTEXT]. For other observed structures, returns a
    best-effort mapping (e.g., "context_pre", "inst" for two tags).

    Returns (text, {}) when no tags are found — narrative style or malformed output.
    """
    matches = list(_TAG_RE.finditer(text))
    if not matches:
        return text, {}

    cleaned_parts = []
    cursor = 0  # cursor in cleaned text
    boundaries = []  # list of (kind, cleaned_start, cleaned_end)

    # Any text before the first tag is discarded (LLM is told not to write it).
    current_kind = matches[0].group(1).lower()
    current_start = 0
    last_match_end = matches[0].end()

    for m in matches[1:]:
        chunk = text[last_match_end:m.start()]
        cleaned_parts.append(chunk)
        cursor += len(chunk)
        boundaries.append((current_kind, current_start, cursor))

        current_kind = m.group(1).lower()
        current_start = cursor
        last_match_end = m.end()

    # Trailing region after the last tag
    chunk = text[last_match_end:]
    cleaned_parts.append(chunk)
    cursor += len(chunk)
    boundaries.append((current_kind, current_start, cursor))

    cleaned_text = "".join(cleaned_parts)

    # Assign canonical keys based on observed kind sequence
    kinds = [k for k, _, _ in boundaries]
    if kinds == ["context", "inst", "context"]:
        keys = ["context_pre", "inst", "context_post"]
    elif kinds == ["context", "inst"]:
        keys = ["context_pre", "inst"]
    elif kinds == ["inst", "context"]:
        keys = ["inst", "context_post"]
    elif kinds == ["inst"]:
        keys = ["inst"]
    elif kinds == ["context"]:
        keys = ["context_pre"]
    else:
        # Multiple context/inst regions in non-canonical order — number them
        ctx_count = 0
        inst_count = 0
        keys = []
        for k in kinds:
            if k == "inst":
                keys.append("inst" if inst_count == 0 else f"inst_{inst_count}")
                inst_count += 1
            else:
                keys.append(
                    "context_pre" if ctx_count == 0
                    else "context_post" if ctx_count == 1
                    else f"context_{ctx_count}"
                )
                ctx_count += 1

    region_offsets = {k: (s, e) for k, (_, s, e) in zip(keys, boundaries)}
    return cleaned_text, region_offsets


def _parse_response(
    raw: str, injection_frequency: int
) -> Tuple[str, SelfCheck, Dict[str, Tuple[int, int]]]:
    """
    Parse the LLM JSON response.

    Returns (cleaned_rewritten_text, self_check, region_offsets).
    Raises ValueError on parse failure.
    """
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}\n---\n{cleaned[:500]}") from e

    rewritten = data.get("rewritten_context") or data.get("rewritten_text") or ""
    if not rewritten:
        raise ValueError(f"No rewritten_context in response: {list(data.keys())}")

    # Strip region tags and capture offsets
    cleaned_text, region_offsets = _parse_regions(rewritten)

    raw_check = data.get("self_check", {})
    facts_ok = bool(raw_check.get("facts_preserved", True))
    count_ok = bool(raw_check.get("placeholder_count_correct", injection_frequency == 0))
    safe_ok = bool(raw_check.get("no_new_unsafe_instruction_added", True))

    self_check = SelfCheck(
        facts_preserved=facts_ok,
        placeholder_count_correct=count_ok,
        no_new_unsafe_instruction_added=safe_ok,
        raw=raw_check,
    )
    return cleaned_text, self_check, region_offsets


def rewrite(
    context: str,
    style: str,
    model: str,
    *,
    position: Position = "middle_inst",
    injection_frequency: int = 1,
    benign_frequency: int = 5,
    max_retries: int = 2,
) -> RewriteResult:
    """
    Rewrite `context` into the requested style using an LLM.

    Parameters
    ----------
    context              : original document text
    style                : key in prompt_templates.PROMPTS (incl. "narrative")
    model                : OpenRouter model ID, e.g. "openai/gpt-4o-mini"
    position             : one of begin_context | begin_inst | middle_inst | end_inst | end_context.
                           Narrative style accepts only begin_context | end_context.
    injection_frequency  : # of [[INJECTION_SPAN]] placeholders (0 = benign rewrite)
    benign_frequency     : approx # of benign instructional sentences (ignored by narrative style)
    max_retries          : retry attempts on parse failure

    Returns
    -------
    RewriteResult with qc_passed=False set when parsing fails.
    """
    if style not in PROMPTS:
        raise ValueError(f"Unknown style {style!r}")
    if injection_frequency not in (0, 1, 2, 3):
        raise ValueError(f"injection_frequency must be 0–3, got {injection_frequency}")
    if benign_frequency < 0:
        raise ValueError(f"benign_frequency must be >= 0, got {benign_frequency}")
    if injection_frequency > 0 and position not in valid_positions_for(style):
        raise ValueError(
            f"Position {position!r} is not valid for style {style!r}. "
            f"Valid: {valid_positions_for(style)}"
        )

    llm = load_model(model, mode="api")
    prompt = build_prompt(
        style, context,
        injection_frequency=injection_frequency,
        position=position,
        benign_frequency=benign_frequency,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.invoke(prompt, return_cost=False)
            cleaned, self_check, region_offsets = _parse_response(raw, injection_frequency)
            return RewriteResult(
                rewritten_text=cleaned,
                style=style,
                position=position,
                injection_frequency=injection_frequency,
                benign_frequency=benign_frequency,
                self_check=self_check,
                rewriter_model=model,
                region_offsets=region_offsets,
            )
        except ValueError as e:
            last_err = e
            logger.warning(f"rewrite attempt {attempt + 1}/{max_retries + 1} failed: {e}")

    # Return a failed result so the pipeline can log and skip rather than crash
    return RewriteResult(
        rewritten_text="",
        style=style,
        position=position,
        injection_frequency=injection_frequency,
        benign_frequency=benign_frequency,
        self_check=SelfCheck(
            facts_preserved=False,
            placeholder_count_correct=False,
            no_new_unsafe_instruction_added=True,
            raw={},
        ),
        rewriter_model=model,
        region_offsets={},
        qc_passed=False,
        qc_notes=[f"parse_failed: {last_err}"],
    )


async def arewrite(
    context: str,
    style: str,
    model: str,
    *,
    llm: Optional[BaseLLM] = None,
    position: Position = "middle_inst",
    injection_frequency: int = 1,
    benign_frequency: int = 5,
    max_retries: int = 2,
) -> RewriteResult:
    """
    Async equivalent of `rewrite`. API calls can run concurrently when callers
    schedule multiple `arewrite` calls with a shared async model client.
    """
    if style not in PROMPTS:
        raise ValueError(f"Unknown style {style!r}")
    if injection_frequency not in (0, 1, 2, 3):
        raise ValueError(f"injection_frequency must be 0-3, got {injection_frequency}")
    if benign_frequency < 0:
        raise ValueError(f"benign_frequency must be >= 0, got {benign_frequency}")
    if injection_frequency > 0 and position not in valid_positions_for(style):
        raise ValueError(
            f"Position {position!r} is not valid for style {style!r}. "
            f"Valid: {valid_positions_for(style)}"
        )

    llm = llm or load_model(model, mode="api")
    prompt = build_prompt(
        style, context,
        injection_frequency=injection_frequency,
        position=position,
        benign_frequency=benign_frequency,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            results = await llm.batch_invoke([prompt])
            if not results or results[0] is None:
                raise RuntimeError(f"batch invoke failed [{llm.model_name}]")
            raw = results[0]
            cleaned, self_check, region_offsets = _parse_response(raw, injection_frequency)
            return RewriteResult(
                rewritten_text=cleaned,
                style=style,
                position=position,
                injection_frequency=injection_frequency,
                benign_frequency=benign_frequency,
                self_check=self_check,
                rewriter_model=model,
                region_offsets=region_offsets,
            )
        except ValueError as e:
            last_err = e
            logger.warning(f"arewrite attempt {attempt + 1}/{max_retries + 1} failed: {e}")
        except Exception as e:
            last_err = e
            logger.warning(f"arewrite attempt {attempt + 1}/{max_retries + 1} failed: {e!r}")

    return RewriteResult(
        rewritten_text="",
        style=style,
        position=position,
        injection_frequency=injection_frequency,
        benign_frequency=benign_frequency,
        self_check=SelfCheck(
            facts_preserved=False,
            placeholder_count_correct=False,
            no_new_unsafe_instruction_added=True,
            raw={},
        ),
        rewriter_model=model,
        region_offsets={},
        qc_passed=False,
        qc_notes=[f"parse_failed: {last_err}"],
    )
