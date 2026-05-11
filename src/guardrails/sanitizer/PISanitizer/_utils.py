import copy
import warnings
import torch


def top_k_mean(tensor_list, k=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = torch.tensor(tensor_list)
        topk, _ = torch.topk(t, k=min(k, t.shape[0]), dim=0)
        return topk.float().mean(dim=0).tolist()


def process_attn(attentions, inputs):
    try:
        input_ids = inputs["input_ids"]
    except (TypeError, IndexError):
        input_ids = inputs

    layer_max_attn, layer_avg_attn, layer_top5_attn = [], [], []
    assert attentions[0].shape[2] == 1
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0, :, :, : input_ids.shape[1]].mean(dim=1)
        layer_max_attn.append(attn.max(dim=0)[0].detach().float().cpu().tolist())
        layer_avg_attn.append(attn.mean(dim=0).detach().float().cpu().tolist())
        layer_top5_attn.append(top_k_mean(attn, k=5))
    return layer_max_attn, layer_avg_attn, layer_top5_attn


def remove_indices(tensor: torch.Tensor, idx_list: list) -> torch.Tensor:
    mask = torch.ones(tensor.size(0), dtype=torch.bool)
    mask[idx_list] = False
    return tensor[mask]
