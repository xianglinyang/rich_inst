'''
RQ2 — attention analysis: does instructionalization dilute the attention paid to
the injected span?

PIGuard is a DebertaV2ForSequenceClassification whose forward pools the [CLS]
token (last_hidden_state[:, 0]) into a linear classifier, but it discards
attentions. We call the underlying `model.deberta(..., output_attentions=True)`
directly, recompute logits via `model.classifier`, and read [CLS]->token
attention (mean of the last 4 layers, mean over heads).

Metrics on the malicious track (injected span location is known exactly, since we
insert it via construct_prompt):
  IAM      = Injection Attention Mass        = sum of CLS attention over injected tokens
  BIAM     = Benign Instruction Attn Mass    = sum of CLS attention over the other content tokens
  RIA      = Relative Injection Attention    = IAM / (IAM + BIAM)
  iam_tok  = IAM / n_injected_tokens         (per-token, length-controlled)
  biam_tok = BIAM / n_benign_tokens
  len_share= n_injected_tokens / (n_injected + n_benign)   (the "expected" RIA if attention were uniform)

The headline check: does RIA(num) fall together with RQ1's P(injection)(num), and
does it drop *below* len_share (suppression beyond mere dilution)?

Run from the repo root:
    python -m src.diagnoise.common.prompts
'''
import argparse
import json
import os
import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.diagnoise.common.guardrails import MODELS, _INJECTION_LABELS
from src.diagnoise.common.prompts import (
    SAMPLES, PAYLOADS, general_instructions, construct_prompt, BENIGN_FILLER,
)

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("attn_analysis")
POSITIONS = ["begin", "middle", "end"]
MAX_NUM = min(20, len(general_instructions))
LAST_N_LAYERS = 4


class AttnAnalyzer:
    '''Attention + score for a DeBERTa-style guard whose head pools [CLS].'''

    def __init__(self, model="piguard", device=None, max_length=512, last_n=LAST_N_LAYERS):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.last_n = last_n
        model_id = MODELS.get(model, model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        ).to(self.device).eval()
        # PIGuard exposes .deberta (encoder) + .classifier; recompute logits there.
        self.encoder = getattr(self.model, "deberta", None) or getattr(self.model, "base_model")
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.inj_idx = next((i for i, l in id2label.items() if l in _INJECTION_LABELS), 1)

    @torch.no_grad()
    def forward(self, text):
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        special = enc.pop("special_tokens_mask")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.encoder(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"],
                           output_attentions=True)
        if out.attentions is None:
            raise RuntimeError("encoder returned no attentions (need attn_implementation='eager')")
        logits = self.model.classifier(out.last_hidden_state[:, 0, :])
        p_inj = F.softmax(logits, dim=-1)[0, self.inj_idx].item()

        # mean of last N layers, mean over heads, query = [CLS] (token 0)
        atts = torch.stack(out.attentions[-self.last_n:], dim=0)  # [L,1,H,S,S]
        cls_attn = atts[:, 0, :, 0, :].mean(dim=(0, 1))           # [S]
        return cls_attn.cpu(), offsets, special, p_inj


def span_token_idx(offsets, special, char_span):
    '''token indices overlapping [start,end), excluding special tokens.'''
    s, e = char_span
    idx = []
    for i, ((a, b), sp) in enumerate(zip(offsets, special)):
        if sp or a == b:        # special token / empty offset
            continue
        if a < e and b > s:     # overlap
            idx.append(i)
    return idx


def content_token_idx(offsets, special):
    return [i for i, ((a, b), sp) in enumerate(zip(offsets, special)) if not sp and b > a]


def metrics_for(analyzer, text, payload):
    cls_attn, offsets, special, p_inj = analyzer.forward(text)
    start = text.find(payload)
    inj_span = (start, start + len(payload)) if start >= 0 else (-1, -1)
    inj_idx = span_token_idx(offsets, special, inj_span)
    content_idx = content_token_idx(offsets, special)
    ben_idx = [i for i in content_idx if i not in set(inj_idx)]

    iam = float(cls_attn[inj_idx].sum()) if inj_idx else 0.0
    biam = float(cls_attn[ben_idx].sum()) if ben_idx else 0.0
    n_inj, n_ben = len(inj_idx), len(ben_idx)
    denom = iam + biam
    return {
        "p_injection": p_inj,
        "IAM": iam, "BIAM": biam,
        "RIA": iam / denom if denom else float("nan"),
        "iam_tok": iam / n_inj if n_inj else float("nan"),
        "biam_tok": biam / n_ben if n_ben else float("nan"),
        "len_share": n_inj / (n_inj + n_ben) if (n_inj + n_ben) else float("nan"),
        "n_inj": n_inj, "n_ben": n_ben,
    }


# ---------------------------------------------------------------- sweep + plot

def run_sweep(analyzer, template, payload, max_num=MAX_NUM):
    rows = []
    for num in range(0, max_num + 1):
        for pos in POSITIONS:
            text = construct_prompt(template, num, pos, malicious_instruction=payload)
            m = metrics_for(analyzer, text, payload)
            rows.append({"num": num, "position": pos, **m})
    return rows


def _mean_by_num(rows, field):
    g = defaultdict(list)
    for r in rows:
        v = r[field]
        if v == v:  # not NaN
            g[r["num"]].append(v)
    nums = sorted(g)
    return nums, [statistics.mean(g[n]) for n in nums]


def plot_sweep(rows, out_path, model="piguard"):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    nums, ria = _mean_by_num(rows, "RIA")
    _, share = _mean_by_num(rows, "len_share")
    ax1.plot(nums, ria, marker="o", color="tab:red", label="RIA (injection attention share)")
    ax1.plot(nums, share, marker="^", ls=":", color="tab:brown",
             label="len_share (uniform-attention baseline)")
    ax1.set_xlabel("number of benign instructions")
    ax1.set_ylabel("relative injection attention")
    ax1.set_ylim(-0.02, 1.02)

    ax2 = ax1.twinx()
    _, pinj = _mean_by_num(rows, "p_injection")
    ax2.plot(nums, pinj, marker="s", color="tab:blue", label="P(injection) [RQ1 score]")
    ax2.axhline(0.5, ls="--", c="grey", lw=1)
    ax2.set_ylabel("P(injection)")
    ax2.set_ylim(-0.02, 1.02)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    plt.title(f"{model}: injection attention vs. instruction count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_token(rows, out_path, model="piguard"):
    plt.figure(figsize=(8, 5))
    nums, iamt = _mean_by_num(rows, "iam_tok")
    _, biamt = _mean_by_num(rows, "biam_tok")
    plt.plot(nums, iamt, marker="o", color="tab:red", label="per-token attention, injected span")
    plt.plot(nums, biamt, marker="s", color="tab:blue", label="per-token attention, benign instr.")
    plt.xlabel("number of benign instructions")
    plt.ylabel("mean CLS attention per token")
    plt.title(f"{model}: per-token attention (length-controlled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------- matched-slot fair comparison
# BENIGN_FILLER (the held-out benign instruction occupying the injected slot) is
# the matched control; it is defined in common.prompts and imported above.


def enrichment_for(analyzer, text, item_text):
    '''Attention enrichment of `item_text` within `text`: per-token CLS attention
    to the item divided by the uniform baseline (1 / n_content_tokens). Value 1.0
    = no salience (uniform); >1 = the item draws more than its share. Invariant to
    adding uniform tokens, so mechanical dilution is removed.'''
    cls_attn, offsets, special, p_inj = analyzer.forward(text)
    start = text.find(item_text)
    if start < 0:
        return None
    item_idx = span_token_idx(offsets, special, (start, start + len(item_text)))
    n_content = len(content_token_idx(offsets, special))
    if not item_idx or not n_content:
        return None
    per_tok = float(cls_attn[item_idx].sum()) / len(item_idx)
    return {"p_injection": p_inj, "enrichment": per_tok * n_content, "per_tok": per_tok}


def run_matched_sweep(analyzer, template, payload, filler=BENIGN_FILLER, max_num=MAX_NUM):
    '''Per num/position: enrichment of the injected item vs a benign item in the
    SAME slot (matched length/position/structure). num>=1 (num=0 is degenerate:
    the item is the whole input).'''
    rows = []
    for num in range(1, max_num + 1):
        for pos in POSITIONS:
            mal = construct_prompt(template, num, pos, malicious_instruction=payload)
            ben = construct_prompt(template, num, pos, malicious_instruction=filler)
            em = enrichment_for(analyzer, mal, payload)
            eb = enrichment_for(analyzer, ben, filler)
            if em and eb:
                rows.append({
                    "num": num, "position": pos,
                    "E_inj": em["enrichment"], "E_ben": eb["enrichment"],
                    "ratio": em["enrichment"] / eb["enrichment"] if eb["enrichment"] else float("nan"),
                    "pinj_mal": em["p_injection"], "pinj_ben": eb["p_injection"],
                })
    return rows


def plot_matched(rows, out_path, model="piguard"):
    nums, e_inj = _mean_by_num(rows, "E_inj")
    _, e_ben = _mean_by_num(rows, "E_ben")
    plt.figure(figsize=(8, 5))
    plt.plot(nums, e_inj, marker="o", color="tab:red", label="injected item (same slot)")
    plt.plot(nums, e_ben, marker="s", color="tab:blue", label="benign item (same slot, matched control)")
    plt.axhline(1.0, ls="--", c="grey", lw=1, label="uniform attention (no salience)")
    plt.xlabel("number of benign instructions")
    plt.ylabel("attention enrichment  (per-token CLS attn / uniform)")
    plt.title(f"{model}: matched-slot attention enrichment (dilution-controlled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- ablation

def ablation(analyzer, template, payload, num, position="begin"):
    '''At a fixed num, compare P(injection) for: full, drop-injection, drop-benign.'''
    full = construct_prompt(template, num, position, malicious_instruction=payload)
    drop_inj = construct_prompt(template, num, position, malicious_instruction=None)  # benign only
    drop_ben = construct_prompt(template, 0, position, malicious_instruction=payload)  # injection only
    return {
        "num": num, "position": position,
        "full": analyzer.forward(full)[3],
        "drop_injection": analyzer.forward(drop_inj)[3],
        "drop_benign": analyzer.forward(drop_ben)[3],
    }


def plot_ablation(abl, out_path, model="piguard"):
    labels = ["full\n(inj + benign)", "drop injection\n(benign only)", "drop benign\n(injection only)"]
    vals = [abl["full"], abl["drop_injection"], abl["drop_benign"]]
    colors = ["tab:purple", "tab:blue", "tab:red"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(range(3), vals, color=colors)
    plt.xticks(range(3), labels)
    plt.axhline(0.5, ls="--", c="grey", lw=1, label="decision boundary")
    plt.ylabel("P(injection)")
    plt.ylim(0, 1.02)
    plt.title(f"{model}: span ablation at num={abl['num']} (pos={abl['position']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- heatmap

def _render_colored_text(text, char_score, inj_range, out_path, title, cmap_name="Reds", cpl=90):
    '''Render `text` verbatim, each character's background coloured by CLS
    attention (NaN -> white). The injected span is underlined in blue.'''
    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    finite = [s for s in char_score if s == s]
    vmax = max(finite) if finite else 1.0
    cmap = mpl.cm.get_cmap(cmap_name)
    istart, iend = inj_range

    # wrap into lines (hard breaks on '\n', soft wrap at cpl chars)
    lines = []
    cur = []
    for i, ch in enumerate(text):
        if ch == "\n":
            lines.append(cur); cur = []; continue
        if len(cur) >= cpl:
            lines.append(cur); cur = []
        cur.append((ch, char_score[i], istart <= i < iend))
    lines.append(cur)

    n_lines = len(lines)
    fig, ax = plt.subplots(figsize=(0.11 * cpl + 1.0, 0.28 * n_lines + 1.0))
    ax.set_xlim(0, cpl); ax.set_ylim(0, n_lines); ax.invert_yaxis(); ax.axis("off")
    for ln, row in enumerate(lines):
        for col, (ch, score, is_inj) in enumerate(row):
            if score == score:
                ax.add_patch(Rectangle((col, ln), 1, 1,
                                       color=cmap(score / vmax if vmax else 0.0), ec="none"))
            if is_inj:
                ax.plot([col, col + 1], [ln + 0.94, ln + 0.94], color="tab:blue", lw=1.4)
            if ch != " ":
                ax.text(col + 0.5, ln + 0.5, ch, ha="center", va="center",
                        family="monospace", fontsize=8)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, label="CLS attention")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_heatmap(analyzer, template, payload, num, out_path, model="piguard", position="begin"):
    text = construct_prompt(template, num, position, malicious_instruction=payload)
    cls_attn, offsets, special, p_inj = analyzer.forward(text)
    start = text.find(payload)
    inj_range = (start, start + len(payload)) if start >= 0 else (-1, -1)

    # per-character attention: each content token paints its char range
    char_score = [float("nan")] * len(text)
    for i, ((a, b), sp) in enumerate(zip(offsets, special)):
        if sp or b <= a:
            continue
        for c in range(a, min(b, len(text))):
            char_score[c] = float(cls_attn[i])

    title = (f"{model}: token attention (num={num}, P(inj)={p_inj:.2f})  "
             f"— blue underline = injected span")
    _render_colored_text(text, char_score, inj_range, out_path, title)


def main():
    parser = argparse.ArgumentParser(description="RQ2 attention analysis")
    parser.add_argument("--model", default="piguard")
    parser.add_argument("--payload", default="obvious_inj", choices=list(PAYLOADS))
    parser.add_argument("--sample", default="bare")
    parser.add_argument("--max-num", type=int, default=MAX_NUM)
    parser.add_argument("--ablation-num", type=int, default=10)
    args = parser.parse_args()

    sample = next(s for s in SAMPLES if s["id"] == args.sample)
    template = sample["template"]
    payload = PAYLOADS[args.payload]
    outdir = os.path.join(HERE, "figures", f"rq2_{args.model}_{args.payload}")
    os.makedirs(outdir, exist_ok=True)

    analyzer = AttnAnalyzer(args.model)

    rows = run_sweep(analyzer, template, payload, max_num=args.max_num)
    with open(os.path.join(DATA_DIR, f"rq2_{args.model}_{args.payload}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    plot_sweep(rows, os.path.join(outdir, "rq2_ria_vs_pinj.png"), model=args.model)
    plot_per_token(rows, os.path.join(outdir, "rq2_per_token.png"), model=args.model)

    matched = run_matched_sweep(analyzer, template, payload, max_num=args.max_num)
    with open(os.path.join(DATA_DIR, f"rq2_matched_{args.model}_{args.payload}.jsonl"), "w") as f:
        for r in matched:
            f.write(json.dumps(r) + "\n")
    plot_matched(matched, os.path.join(outdir, "rq2_matched_slot.png"), model=args.model)

    abl = ablation(analyzer, template, payload, args.ablation_num)
    plot_ablation(abl, os.path.join(outdir, "rq2_ablation.png"), model=args.model)

    for n in (0, args.ablation_num):
        plot_heatmap(analyzer, template, payload, n,
                     os.path.join(outdir, f"rq2_heatmap_num{n}.png"), model=args.model)

    print(f"[RQ2] figures + rq2_{args.model}_{args.payload}.jsonl written ({outdir})")
    print(f"[RQ2] ablation @num={abl['num']}: {abl}")


if __name__ == "__main__":
    main()
