'''Model-specific templates'''

import logging

logger = logging.getLogger(__name__)

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_END_OF_TEXT = "<|end_of_text|>"
LLAMA2_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA2_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA2_SYSTEM_TAG="<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA2_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA2_SEP_TOKEN = ""
LLAMA2_FORMATTED_PROMPT = "<s>[INST] {prompt} [/INST]"

# Llama 3 chat templates are based on
# - https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# <|begin_of_text|> is automatically added by the tokenizer
LLAMA3_BEGIN_OF_TEXT = "<|begin_of_text|>"
LLAMA3_END_OF_TEXT = "<|eot_id|>"
LLAMA3_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA3_SEP_TOKEN = ""
LLAMA3_FORMATTED_PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"

# Mistral chat templates are based on
# - https://github.com/mistralai/mistralai/blob/main/mistralai/chat.py
MISTRAL_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
MISTRAL_BEGIN_OF_TEXT = "<s>"
MISTRAL_END_OF_TEXT = "</s>"
MISTRAL_SYSTEM_TAG = "[INST]"
MISTRAL_SYSTEM = "[INST] {}".format(MISTRAL_DEFAULT_SYSTEM_PROMPT)
MISTRAL_USER_TAG="[INST]"
MISTRAL_ASSISTANT_TAG="[/INST]"
MISTRAL_SEP_TOKEN = " "
MISTRAL_FORMATTED_PROMPT = "<s> [INST] {prompt} [/INST]"

QWEN_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
QWEN_BEGIN_OF_TEXT = "<|im_start|>"
QWEN_END_OF_TEXT = "<|im_end|>"
QWEN_SYSTEM_TAG = "<|im_start|>system\n"
QWEN_SYSTEM = "<|im_start|>system\n{}<|im_end|>\n".format(QWEN_DEFAULT_SYSTEM_PROMPT)
QWEN_USER_TAG="<|im_start|>user\n"
QWEN_ASSISTANT_TAG="<|im_start|>assistant\n"
QWEN_SEP_TOKEN = "<|im_end|>"
QWEN_PAD_TOKEN = "<|endoftext|>"
QWEN_FORMATTED_PROMPT = "<|im_start|>user\n{prompt}<|im_end|>\n"


# ---------------------------------------------------------------------------
# Model context-length registry
# Keys are lowercase substrings matched against model_name.lower().
# Listed from most-specific to least-specific so the first match wins.
# ---------------------------------------------------------------------------
_MODEL_CONTEXT_LENGTHS = [
    # OpenAI
    ("gpt-4.1",              1047576),
    ("gpt-4o",               128000),
    ("gpt-4-turbo",          128000),
    ("gpt-4",                8192),
    ("gpt-3.5-turbo-16k",    16384),
    ("gpt-3.5",              4096),
    ("o1",                   200000),
    ("o3",                   200000),
    # Anthropic / Claude
    ("claude",               200000),
    # Google Gemini
    ("gemini-2.5",           1048576),
    ("gemini-2.0",           1048576),
    ("gemini-1.5",           1048576),
    ("gemini-1.0",           32768),
    # Meta Llama
    ("llama-3.3-70b",        131072),
    ("llama-3.1-405b",       131072),
    ("llama-3.1-70b",        131072),
    ("llama-3.1-8b",         131072),
    ("llama-3",              131072),
    ("llama-2",              4096),
    # Mistral
    ("mistral-large",        131072),
    ("mistral-small",        131072),
    ("mixtral-8x22b",        65536),
    ("mixtral",              32768),
    ("mistral-7b",           32768),
    # Qwen
    ("qwen2.5-72b",          131072),
    ("qwen2.5",              131072),
    ("qwen2-72b",            131072),
    ("qwen2",                131072),
    ("qwen3",                131072),
    ("qwen",                 32768),
    # DeepSeek
    ("deepseek-r1",          65536),
    ("deepseek",             65536),
    # GLM
    ("glm-4",                131072),
    ("glm",                  32768),
    # minimax

]

_DEFAULT_CONTEXT_LENGTH = 150000  # safe fallback for unknown models


def get_context_length(model_name: str) -> int:
    """Return the context-window size (in tokens) for the given model name."""
    name_lower = model_name.lower()
    for pattern, length in _MODEL_CONTEXT_LENGTHS:
        if pattern in name_lower:
            return length
    logger.warning(
        f"[model_configs] Unknown model '{model_name}'; "
        f"defaulting to context length {_DEFAULT_CONTEXT_LENGTH}."
    )
    return _DEFAULT_CONTEXT_LENGTH


def get_stop_tokens(model_name_or_path):
    model_name_lower = model_name_or_path.lower()
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return [LLAMA3_END_OF_TEXT]
    elif "mistral-7" in model_name_lower or "mistral" in model_name_lower or "zephyr_7b_r2d2" in model_name_lower:
        return [MISTRAL_END_OF_TEXT]
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return [LLAMA2_END_OF_TEXT]
    elif "qwen" in model_name_lower:
        return [QWEN_END_OF_TEXT]
    else:
        raise ValueError(f"Not implemented for model: {model_name_or_path}")

def get_formatted_prompt(model_name_or_path, prompt):
    # TODO or we implement with the tokenizer?
    model_name_lower = model_name_or_path.lower()
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return LLAMA3_FORMATTED_PROMPT.format(prompt=prompt)
    elif "mistral-7" in model_name_lower or "mistral" in model_name_lower:
        return MISTRAL_FORMATTED_PROMPT.format(prompt=prompt)
    elif "zephyr_7b_r2d2" in model_name_lower:
        return MISTRAL_FORMATTED_PROMPT.format(prompt=prompt)
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return LLAMA2_FORMATTED_PROMPT.format(prompt=prompt)
    elif "qwen" in model_name_lower:
        return QWEN_FORMATTED_PROMPT.format(prompt=prompt)
    else:
        raise ValueError(f"Not implemented for model: {model_name_or_path}")