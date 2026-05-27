#!/usr/bin/env bash
# Build instruction-heavy injection datasets for the BIPIA inputs using src.inst.
#
# Run from the project root:
#   bash scripts/run_experiment.sh smoke
#   bash scripts/run_experiment.sh small
#   bash scripts/run_experiment.sh full
#
# Output:
#   data/bipia/<tier>_variants.jsonl
#
# Requires:
#   OPENROUTER_API_KEY
#
# Useful env vars:
#   MODEL=google/gemini-3-flash-preview
#   ASYNC_BATCH_SIZE=100
#   OUTPUT=data/bipia/custom_variants.jsonl

set -euo pipefail

DATASET="bipia/email"
TASK="email"
MODEL="${MODEL:-google/gemini-3-flash-preview}"
ASYNC_BATCH_SIZE="${ASYNC_BATCH_SIZE:-100}"
TIER="${1:-small}"


case "$TIER" in
  small)
    NUM_CONTEXTS=10
    PAYLOADS_PER_CATEGORY=1
    CATEGORIES=()
    STYLES=(email)
    POSITIONS=(begin_inst middle_inst end_context)
    BENIGN_FREQUENCIES=(7)
    INJECTION_FREQUENCIES=(1)
    ;;
  full)
    NUM_CONTEXTS=50
    PAYLOADS_PER_CATEGORY=5
    CATEGORIES=()   # empty = all categories
    STYLES=(general email code_readme code_quick_start api_docs debugging web_help \
            table abstract shopping_guide travel_guide customer_support_faq narrative)
    POSITIONS=(begin_context begin_inst middle_inst end_inst end_context)
    BENIGN_FREQUENCIES=(3 5 7)
    INJECTION_FREQUENCIES=(1)
    ;;
  *)
    echo "Unknown tier '${TIER}'. Choose: smoke | small | full" >&2
    exit 1
    ;;
esac

OUTPUT="${OUTPUT:-data/${DATASET}/variants.jsonl}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

if [[ ! -f "data/${DATASET}/context.jsonl" ]]; then
  echo "ERROR: data/${DATASET}/context.jsonl not found. Run from the project root." >&2
  exit 1
fi

if [[ ! -f "data/${DATASET}/injected_command.json" ]]; then
  echo "ERROR: data/${DATASET}/injected_command.json not found. Run from the project root." >&2
  exit 1
fi

ARGS=(
  --dataset              "$DATASET"
  --output               "$OUTPUT"
  --model                "$MODEL"
  --styles               "${STYLES[@]}"
  --positions            "${POSITIONS[@]}"
  --benign-frequencies   "${BENIGN_FREQUENCIES[@]}"
  --injection-frequencies "${INJECTION_FREQUENCIES[@]}"
  --num-contexts         "$NUM_CONTEXTS"
  --payloads-per-category "$PAYLOADS_PER_CATEGORY"
  --async-batch-size     "$ASYNC_BATCH_SIZE"
  --log-level            INFO
)

if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
  ARGS+=(--categories "${CATEGORIES[@]}")
fi

echo "============================================================"
echo "  src.inst BIPIA dataset build"
echo "  Tier:             ${TIER}"
echo "  Task:             ${TASK}"
echo "  Model:            ${MODEL}"
echo "  Contexts:         ${NUM_CONTEXTS}"
echo "  Payloads/category:${PAYLOADS_PER_CATEGORY}"
echo "  Categories:       ${CATEGORIES[*]:-<all>}"
echo "  Styles:           ${STYLES[*]}"
echo "  Positions:        ${POSITIONS[*]}"
echo "  Benign freq:      ${BENIGN_FREQUENCIES[*]}"
echo "  Injection freq:   ${INJECTION_FREQUENCIES[*]}"
echo "  Async batch size: ${ASYNC_BATCH_SIZE}"
echo "  Output:           ${OUTPUT}"
echo "============================================================"

python -m src.data.build_dataset "${ARGS[@]}"
