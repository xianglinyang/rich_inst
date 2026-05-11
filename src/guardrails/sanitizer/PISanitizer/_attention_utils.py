import math
from typing import Any, Optional
import torch
import transformers.models


def _infer_model_type(model):
    keywords = {"llama": "llama", "gpt": "gpt_oss", "glm": "glm",
                "phi": "phi3", "qwen2": "qwen2", "qwen3": "qwen3", "gemma": "gemma3"}
    name = model.name_or_path.lower()
    for kw, mt in keywords.items():
        if kw in name:
            return mt
    raise ValueError(f"Unknown model: {model.name_or_path}")


def _get_helpers(model_type):
    if not hasattr(transformers.models, model_type):
        raise ValueError(f"Unknown model type: {model_type}")
    mod = getattr(getattr(transformers.models, model_type), f"modeling_{model_type}")
    return mod.apply_rotary_pos_emb, mod.repeat_kv


def _pos_ids_and_mask(model, hidden_states):
    seq_len = hidden_states[0].shape[1]
    position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
    mask = torch.ones(seq_len, seq_len + 1, device=model.device, dtype=model.dtype)
    mask = torch.triu(mask, diagonal=1) * torch.finfo(model.dtype).min
    return position_ids, mask[None, None]


def _lang_model(model, model_type):
    return model.model.language_model if model_type == "gemma3" else model.model


def get_attention_weights_one_layer(
    model: Any,
    hidden_states: Any,
    layer_index: int,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Any:
    with torch.no_grad():
        position_ids, attention_mask = _pos_ids_and_mask(model, hidden_states)
        model_type = model_type or _infer_model_type(model)
        lm = _lang_model(model, model_type)
        layer = lm.layers[layer_index]
        self_attn = layer.self_attn

        hs = layer.input_layernorm(hidden_states[layer_index])
        bsz, q_len, _ = hs.size()
        num_heads = lm.config.num_attention_heads
        num_kv_heads = lm.config.num_key_value_heads
        head_dim = self_attn.head_dim

        if model_type in ("llama", "qwen2", "qwen1.5", "gemma3", "glm"):
            q = self_attn.q_proj(hs)
            k = self_attn.k_proj(hs)
        elif model_type == "phi3":
            qkv = self_attn.qkv_proj(hs)
            qpos = num_heads * head_dim
            q = qkv[..., :qpos]
            k = qkv[..., qpos: qpos + num_kv_heads * head_dim]
        else:
            raise ValueError(f"Unknown model: {model.name_or_path}")

        q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        if model_type in ("gemma3", "qwen3"):
            q = self_attn.q_norm(q)
            k = self_attn.k_norm(k)

        if model_type == "gemma3":
            pos_emb = (lm.rotary_emb_local if self_attn.is_sliding else lm.rotary_emb)(hs, position_ids)
        else:
            pos_emb = lm.rotary_emb(hs, position_ids)
        cos, sin = pos_emb

        apply_rope, repeat_kv = _get_helpers(model_type)
        q, k = q.to("cuda:0"), k.to("cuda:0")
        cos, sin = cos.to("cuda:0"), sin.to("cuda:0")
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self_attn.num_key_value_groups)

        num_tokens = hidden_states[0].shape[1] + 1
        a_start = attribution_start if attribution_start is not None else 1
        a_end = attribution_end if attribution_end is not None else num_tokens

        causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        causal_mask = causal_mask[:, :, a_start - 1: a_end - 1]
        q = q[:, :, a_start - 1: a_end - 1]

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim) + causal_mask
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)

    return attn
