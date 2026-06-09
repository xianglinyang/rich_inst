#!/usr/bin/env bash
# Build instruction-heavy injection datasets for the BIPIA inputs using src.inst.

set -euo pipefail

TASK="email"
DATASET="bipia/email"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-google/gemini-3-flash-preview}"
ASYNC_BATCH_SIZE="${ASYNC_BATCH_SIZE:-100}"

NUM_CONTEXTS=50
PAYLOADS_PER_CATEGORY=1

# Leave empty to use all categories for the chosen DATASET + SPLIT.
# To restrict, uncomment and paste the desired names from the lists below.
#
# bipia/email + train:
# CATEGORIES=("Information Retrieval" "Content Creation" "Learning and Tutoring" \
#            "Language Translation" "Programming Help" "Alphanumeric Substitution" \
#            "Homophonic Substitution" "Misspelling Intentionally" "Anagramming" \
#            "Space Removal & Grouping" "Instruction" "Social Interaction" \
#            "Persuasion" "Clickbait" "Malware Distribution")
#
# bipia/email + test:
CATEGORIES=("Task Automation" "Business Intelligence" "Conversational Agent" \
           "Research Assistance" "Sentiment Analysis" "Substitution Ciphers" \
           "Base Encoding" "Reverse Text" "Emoji Substitution" "Language Translation" \
           "Information Dissemination" "Marketing & Advertising" "Entertainment" \
           "Scams & Fraud" "Misinformation & Propaganda")
#
# bipia/code + train:
# CATEGORIES=("Cookie Theft" "Memory Scanning" "Dumpster Diving" \
#            "Environment Variable Analysis" "Device and Driver Enumeration" \
#            "Sending Out Spam Emails" "Crippling Critical Infrastructures" \
#            "Network Propagation" "Exploiting System Vulnerabilities" \
#            "Cryptocurrency Mining")
#
# bipia/code + test:
# CATEGORIES=("Data Eavesdropping" "Traffic Analysis" "Keylogging" "Screen Scraping" \
#            "Introduce System Fingerprinting" "Blocking Internet Connection" \
#            "Corrupting an Operating System" \
#            "Encrypting Documents and Demanding Ransom (Ransomware)" \
#            "Compromising Computers" \
#            "Bringing Down Hosts and Servers (Denial of Service)")
# CATEGORIES=()

# Supported styles (narrative only supports begin_context / end_context positions).
# To restrict, replace the array with a subset of the commented values below.
#STYLES=(general email code_readme code_quick_start api_docs debugging web_help \
#        table abstract shopping_guide travel_guide customer_support_faq narrative)
STYLES=(customer_support_faq)

# Supported positions (narrative: begin_context and end_context only).
# To restrict, replace the array with a subset of the commented values below.
#POSITIONS=(begin_context begin_inst middle_inst end_inst end_context)
POSITIONS=(end_inst)

BENIGN_FREQUENCIES=(3 5)
INJECTION_FREQUENCIES=(1)

# OUTPUT="${OUTPUT:-data/${DATASET}/${SPLIT}_variants.jsonl}"
OUTPUT="/home/xianglin/git_space/BIPIA/benchmark/${TASK}_${SPLIT}_variants.jsonl"


# -----------------------------------------------------------------------------------

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

if [[ ! -d "data/${DATASET}" ]]; then
  echo "ERROR: data/${DATASET}/ not found. Run from the project root." >&2
  exit 1
fi

ARGS=(
  --dataset              "$DATASET"
  --split                "$SPLIT"
  --output               "$OUTPUT"
  --model                "$MODEL"
  --styles               "${STYLES[@]}"
  --positions            "${POSITIONS[@]}"
  --benign-frequencies   "${BENIGN_FREQUENCIES[@]}"
  --injection-frequencies "${INJECTION_FREQUENCIES[@]}"
  --num-contexts         "$NUM_CONTEXTS"
  --payloads-per-category "$PAYLOADS_PER_CATEGORY"
  --async-batch-size     "$ASYNC_BATCH_SIZE"
  --append
  --log-level            INFO
)

if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
  ARGS+=(--categories "${CATEGORIES[@]}")
fi

echo "============================================================"
echo "  src.inst BIPIA dataset build"
echo "  Task:             ${TASK}"
echo "  Split:            ${SPLIT}"
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
