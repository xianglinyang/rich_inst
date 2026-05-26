"""
src/inst — Instruction-heavy rewriter.

Public API
----------
transform(context, payloads, ...)   one-call: rewrite + QC + payload substitution
rewrite(context, style, model, ...) low-level rewrite (returns RewriteResult)
run_qc(result, original_context)    deterministic QC on a RewriteResult
substitute_payloads(rewritten, payloads)   payload substitution only

Dataclasses
-----------
ModifiedContext  — final transformation result
RewriteResult    — intermediate (placeholders still present)
Span             — char offsets of an injected payload
"""
from typing import List, Optional

from src.inst.composer import substitute_payloads
from src.inst.payload_formatter import (
    STYLE_DESCRIPTIONS,
    StylizedPayload,
    format_payload,
)
from src.inst.qc import run_qc
from src.inst.schema import ModifiedContext, RewriteResult, Span, SelfCheck
from src.inst.style_rewriter import rewrite

__all__ = [
    "transform",
    "rewrite",
    "run_qc",
    "substitute_payloads",
    "format_payload",
    "STYLE_DESCRIPTIONS",
    "ModifiedContext",
    "RewriteResult",
    "StylizedPayload",
    "Span",
    "SelfCheck",
]


def transform(
    context: str,
    payloads: Optional[List[str]],
    *,
    style: str,
    model: str,
    position: str = "middle_inst",
    benign_frequency: int = 5,
    max_retries: int = 2,
) -> ModifiedContext:
    """
    Transform `context` into an instruction-heavy variant and inject `payloads`.

    Parameters
    ----------
    context           : the original document (from the caller's dataset)
    payloads          : list of injection texts (from the caller's dataset).
                        len(payloads) determines injection_frequency.
                        Pass None or [] for a benign rewrite (no injection).
    style             : one of prompt_templates.PROMPTS keys (incl. "narrative")
    model             : OpenRouter model ID, e.g. "openai/gpt-5-mini"
    position          : begin_context | begin_inst | middle_inst | end_inst | end_context
                        Narrative style only accepts begin_context | end_context.
                        Ignored when payloads is empty.
    benign_frequency  : approx # of benign instructional sentences (ignored by narrative)
    max_retries       : retry attempts on parse failure

    Returns
    -------
    ModifiedContext — `modified_text` is what the guard should see.
    """
    payloads = payloads or []
    injection_frequency = len(payloads)

    # Pin position to a no-op label when there's no injection (avoids style-position validation)
    rewrite_position = position if injection_frequency > 0 else "begin_context"

    result = rewrite(
        context,
        style=style,
        model=model,
        position=rewrite_position,
        injection_frequency=injection_frequency,
        benign_frequency=benign_frequency,
        max_retries=max_retries,
    )
    result = run_qc(result, context)

    modified_text, spans = substitute_payloads(result.rewritten_text, payloads)

    return ModifiedContext(
        original_text=context,
        modified_text=modified_text,
        style=style,
        position=position if injection_frequency > 0 else "none",
        injection_frequency=injection_frequency,
        benign_frequency=benign_frequency,
        injected_spans=spans,
        binary_label=1 if spans else 0,
        rewriter_model=model,
        qc_passed=result.qc_passed,
        qc_notes=result.qc_notes,
        rewritten_text=result.rewritten_text,
        region_offsets=result.region_offsets,
    )
