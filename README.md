# rich_inst

Prompt injection detection and sanitization baselines for LLM agents.

---

## Baselines

### Guard Model Classifiers
| Name | Port | Notes |
|---|---|---|
| PIGuard | 12390 | Fine-tuned classifier |
| PromptGuard 22M | 12392 | Meta PromptGuard small |
| PromptGuard 86M | 12393 | Meta PromptGuard large |
| ProtectAI v2 | 12391 | ProtectAI deberta-v3 |
| IntentGuard | 12394 | LLM-based intent checker (local llamafile) |
| LLMJudge | — | LLM-as-a-judge via OpenRouter |

### Sanitizers
| Name | Notes |
|---|---|
| LLMSanitizer | LLM-based via OpenRouter; `mode="one_shot"`: single call returns detected spans + cleaned text; `mode="cleaner"`: two-step detect-then-remove |
| PISanitizer | Attention-weight-based (Llama-3.1-8B-Instruct, local) |

---

## Installation

### 1. Python dependencies

```bash
pip install torch transformers accelerate openai requests fastapi uvicorn
```

### 2. PISanitizer model (Llama-3.1-8B-Instruct)

The PISanitizer logic is built into this repo (`src/guardrails/sanitizer/PISanitizer/`).
The model weights (~16 GB, bfloat16) are downloaded automatically from HuggingFace on first use — no extra clone or `PYTHONPATH` needed.

### 3. IntentGuard (local llamafile)

```bash
pip install intentguard
# The service downloads kdunee/IntentGuard-1-qwen2.5-coder-1.5b-gguf on first start
uvicorn src.guardrails.detection.IntentGuard.app:app --host 127.0.0.1 --port 12394
```

### 4. OpenRouter API key (LLMJudge, LLMSanitizer, cleaner)

```bash
export OPENROUTER_API_KEY=<your-key>
```

---

## Starting guard services

Each classifier runs as a local FastAPI service. Start the ones you need:

```bash
uvicorn src.guardrails.detection.PIGuard.app:app          --host 127.0.0.1 --port 12390
uvicorn src.guardrails.detection.ProtectAIv2.app:app      --host 127.0.0.1 --port 12391
uvicorn src.guardrails.detection.PromptGuard.app:app_22m      --host 127.0.0.1 --port 12392  # 22M
uvicorn src.guardrails.detection.PromptGuard.app:app_86m      --host 127.0.0.1 --port 12393  # 86M
uvicorn src.guardrails.detection.IntentGuard.app:app      --host 127.0.0.1 --port 12394
```

---

## Usage

```python
from src.guardrails.wrapper import detect, judge_attack, sanitize

# Unified detector — classifier or judge, same call
decision = detect(text, "PromptGuard_86M")                              # classifier
decision = detect(text, "PromptGuard_86M", scan_mode="chunk_max")       # with chunking
decision = detect(text, "openai/gpt-4o-mini", user_goal=goal)           # LLM judge

# Judge with richer structured output (JudgeDecision)
verdict = judge_attack(text, user_goal=goal, mode="authority_aware")

# Sanitizer
result = sanitize(text, "openai/gpt-4o-mini")                    # one-shot (default)
result = sanitize(text, "openai/gpt-4o-mini", mode="cleaner")    # two-step detect-then-remove
result = sanitize(text, "PISanitizer")                           # attention-weight based (local)
```

### SanitizerDecision fields

```python
result.is_injection    # bool
result.confidence      # float 0.0–1.0
result.detected_spans  # List[DetectedSpan] — each has .text, .start, .end, .reason
result.sanitized_text  # original with injections removed
result.reason          # overall explanation
```

---

## Testing

```bash
cd rich_inst
python test_sanitizers.py               # all sanitizers
python test_sanitizers.py llm cleaner   # LLM-based only
python test_sanitizers.py pisa          # PISanitizer only
```

---

## Papers

- BIPIA
- CAPTURE
- WAInjectBench
