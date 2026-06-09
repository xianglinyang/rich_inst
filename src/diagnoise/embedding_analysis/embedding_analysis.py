'''
RQ4 — embedding analysis: does instructionalization drag the malicious command's
PIGuard [CLS] representation toward the benign region?

We read PIGuard's [CLS] vector (`encoder(...).last_hidden_state[:, 0, :]`, the exact
input to `model.classifier`, so drift here is mechanistically tied to the decision)
for four groups, all built from the same template / list format / num sweep so the
only thing that varies is content:

  A. original           — injection only (num=0)                  malicious anchor
  B. tutorial           — benign instructions, num=1..20          benign-instruction anchor
  C. instructional mal. — injection + num benign instr., 0..20    the trajectory we watch
  D. random benign      — num declarative (non-instruction) lines benign-content anchor
  (E. benign filler in the injected slot — length-matched control for C)

Quantitative drift lives in the original high-dim [CLS] space (centroid distances +
projection onto the benign-instruction <- malicious axis). The 2-D PCA / t-SNE / UMAP
scatters are for visual comparison only — t-SNE/UMAP distort global distances, so only
PCA's inter-cluster distances are trustworthy.

Run from the repo root:
    python -m src.diagnoise.embedding_analysis.embedding_analysis --model piguard --payload obvious_inj
'''
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.diagnoise.common.guardrails import MODELS, _INJECTION_LABELS
from src.diagnoise.common.prompts import (
    SAMPLES, PAYLOADS, general_instructions, random_instructions, construct_prompt)
from src.diagnoise.common.prompts import BENIGN_FILLER
from src.diagnoise.common.bench_data import load_bipia_injections, load_dolly_tutorials

from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("embedding_analysis")
POSITIONS = ["begin", "middle", "end"]
MAX_NUM = min(20, len(general_instructions))


# --------------------------------------------------------------- random benign pool
# Declarative, non-imperative sentences (no second-person commands), so group D is
# "benign content" without the instruction-list semantics of group B. Seeded combos of
# subject x predicate give ~10^3 distinct sentences with reproducible ordering.
_SUBJECTS = [
    "The morning fog", "A small wooden boat", "The old stone bridge", "A field of barley",
    "The quiet harbor", "An empty train platform", "The library reading room", "A row of poplar trees",
    "The corner bakery", "A weathered lighthouse", "The mountain stream", "An abandoned orchard",
    "The village square", "A flock of starlings", "The river delta", "An autumn afternoon",
    "The cobblestone alley", "A bowl of ripe pears", "The distant glacier", "An old brass clock",
    "The coastal road", "A pair of grey herons", "The wheat market", "An overgrown garden",
    "The harbor lighthouse", "A sheet of thin ice", "The pine forest", "An open meadow",
    "The summer market", "A line of fishing nets", "The clay rooftops", "An evening tide",
]
_PREDICATES = [
    "drifted slowly across the bay.", "stood quietly at the edge of town.",
    "was painted a faded shade of blue.", "smelled faintly of rain and pine.",
    "stretched toward the low grey hills.", "glowed under the late afternoon sun.",
    "remained still in the windless air.", "was covered in a thin layer of frost.",
    "echoed with the calls of distant gulls.", "had weathered many quiet winters.",
    "lay wrapped in a soft morning mist.", "shimmered above the cooling sand.",
    "rested beneath a canopy of oak leaves.", "curved gently along the shoreline.",
    "filled the valley with a warm amber light.", "settled into a calm, even quiet.",
    "was lined with rows of flowering thyme.", "carried the scent of the open sea.",
    "stood out against the pale winter sky.", "faded gradually into the evening haze.",
]


def random_benign_pool(n, seed=0):
    rng = random.Random(seed)
    combos = [(s, p) for s in _SUBJECTS for p in _PREDICATES]
    rng.shuffle(combos)
    return [f"{s} {p}" for s, p in combos[:n]]


# --------------------------------------------------------------- [CLS] extractor

class CLSExtractor:
    '''Sequence-classification guard: returns the *classifier-input* vector and the true
    P(injection) for any text. A forward pre-hook on `model.classifier` captures exactly
    the vector fed to the head — this is architecture-agnostic: for PIGuard the head
    reads [CLS] directly, but for ProtectAIv2 a ContextPooler sits between [CLS] and the
    classifier, so reading raw [CLS] would skip the pooler and mis-score. We take logits
    from the true `model(**enc)` forward, so P(injection) is always correct.'''

    def __init__(self, model="piguard", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        model_id = MODELS.get(model, model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device).eval()
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.inj_idx = next((i for i, l in id2label.items() if l in _INJECTION_LABELS), 1)

        self._clf_input = {}
        self.model.classifier.register_forward_pre_hook(
            lambda module, inp: self._clf_input.__setitem__("v", inp[0].detach()))

    @torch.no_grad()
    def cls(self, text):
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits               # true forward
        p_inj = F.softmax(logits, dim=-1)[0, self.inj_idx].item()
        vec = self._clf_input["v"][0].cpu().numpy()      # the exact classifier-input vector
        return vec, p_inj


# --------------------------------------------------------------- build the dataset

def build_points(extractor, template, payload, max_num=MAX_NUM, seed=0, n_real=0,
                 template_real=False):
    '''Return (X, meta) where X is [N, H] and meta is a list of dicts.'''
    rand_pool = random_benign_pool(max_num, seed=seed)
    X, meta = [], []

    def add(text, group, num, pos):
        vec, p = extractor.cls(text)
        X.append(vec)
        meta.append({"group": group, "num": num, "position": pos, "p_injection": p})

    # A. original / injection only (single anchor; positions degenerate at num=0)
    add(construct_prompt(template, 0, "begin", malicious_instruction=payload), "A_original", 0, "-")

    for num in range(1, max_num + 1):
        # B. tutorial (benign instructions; position ignored when mal=None)
        add(construct_prompt(template, num, "begin", malicious_instruction=None),
            "B_tutorial", num, "-")
        # D. random benign content (same list format, declarative content)
        add(construct_prompt(template, num, "begin", malicious_instruction=None,
                             instructions_pool=rand_pool), "D_random_benign", num, "-")

    # C. instructional malicious trajectory (num x position) — ON-TOPIC benign filler
    for num in range(0, max_num + 1):
        for pos in POSITIONS:
            add(construct_prompt(template, num, pos, malicious_instruction=payload),
                "C_instr_malicious", num, pos)
            if num == 0:
                break  # positions degenerate at num=0

    # H. OFF-topic instructional malicious trajectory — same injection, but the benign
    # filler is `random_instructions` (imperative but topically unrelated). RQ1b shows
    # this does NOT evade at the score level; here we test whether it also fails to
    # drift toward the benign-instruction region (Lever B prediction: little/no drift).
    n_off = min(max_num, len(random_instructions))
    for num in range(0, n_off + 1):
        for pos in POSITIONS:
            add(construct_prompt(template, num, pos, malicious_instruction=payload,
                                 instructions_pool=random_instructions),
                "H_offtopic_malicious", num, pos)
            if num == 0:
                break  # positions degenerate at num=0

    # E. benign filler in the injected slot — length-matched control for C
    for num in range(1, max_num + 1):
        for pos in POSITIONS:
            add(construct_prompt(template, num, pos, malicious_instruction=BENIGN_FILLER),
                "E_benign_filler", num, pos)

    # F/G. real-world reference clouds — valid endpoint clusters. If template_real,
    # wrap each in the SAME numbered-list carrier as A-E (removes the raw-vs-templated
    # format confound); otherwise embed the raw text.
    if n_real:
        for s in load_bipia_injections(n_real, seed=seed):
            text = construct_prompt(template, 0, "begin", malicious_instruction=s) if template_real else s
            add(text, "F_real_malicious", -1, "-")
        for s in load_dolly_tutorials(n_real, seed=seed):
            text = (construct_prompt(template, 1, "begin", malicious_instruction=None,
                                     instructions_pool=[s]) if template_real else s)
            add(text, "G_real_benign", -1, "-")

    return np.array(X), meta


# --------------------------------------------------------------- drift metrics (hi-dim)

def _centroid(Xs, meta, group):
    idx = [i for i, m in enumerate(meta) if m["group"] == group]
    return Xs[idx].mean(axis=0), idx


def drift_metrics(Xs, meta, group="C_instr_malicious"):
    '''All in the standardized [CLS] space. Projection of the `group` trajectory onto the
    benign-instruction <- malicious axis (always the ON-TOPIC A->B axis, so on- and
    off-topic trajectories are comparable), normalized so 0 = malicious anchor,
    1 = benign-instruction centroid.'''
    c_mal, _ = _centroid(Xs, meta, "A_original")
    c_ben_i, _ = _centroid(Xs, meta, "B_tutorial")
    c_ben_r, _ = _centroid(Xs, meta, "D_random_benign")

    groups = {m["group"] for m in meta}
    has_real = "F_real_malicious" in groups and "G_real_benign" in groups
    if has_real:
        c_real_mal, _ = _centroid(Xs, meta, "F_real_malicious")
        c_real_ben, _ = _centroid(Xs, meta, "G_real_benign")

    axis = c_ben_i - c_mal
    axis_sq = float(axis @ axis) or 1.0

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else float("nan")

    rows = []
    by_num = defaultdict(list)
    for i, m in enumerate(meta):
        if m["group"] == group:
            by_num[m["num"]].append(i)
    for num in sorted(by_num):
        c = Xs[by_num[num]].mean(axis=0)
        p = np.mean([meta[i]["p_injection"] for i in by_num[num]])
        row = {
            "num": num,
            "drift": float((c - c_mal) @ axis / axis_sq),     # 0=mal anchor, 1=benign-instr centroid
            "dist_mal": float(np.linalg.norm(c - c_mal)),
            "dist_ben_instr": float(np.linalg.norm(c - c_ben_i)),
            "dist_ben_rand": float(np.linalg.norm(c - c_ben_r)),
            "cos_mal": cos(c, c_mal),
            "cos_ben_instr": cos(c, c_ben_i),
            "p_injection": float(p),
        }
        if has_real:
            # drift along the real malicious -> real benign axis (0=real mal, 1=real benign)
            raxis = c_real_ben - c_real_mal
            raxis_sq = float(raxis @ raxis) or 1.0
            row["drift_real"] = float((c - c_real_mal) @ raxis / raxis_sq)
            row["dist_real_mal"] = float(np.linalg.norm(c - c_real_mal))
            row["dist_real_ben"] = float(np.linalg.norm(c - c_real_ben))
        rows.append(row)
    return rows


# --------------------------------------------------------------- plotting

# Okabe-Ito colorblind-safe qualitative palette (https://jfly.uni-koeln.de/color/).
# Distinguishable under protanopia/deuteranopia/tritanopia; marker shape is a
# redundant channel so the groups remain separable in grayscale too.
_GROUP_STYLE = {
    "A_original":       dict(color="#000000",   marker="*", s=320, label="A: original (injection only)", zorder=5),
    "B_tutorial":       dict(color="#0072B2",   marker="o", s=40,  label="B: tutorial (benign instr.)"),
    "D_random_benign":  dict(color="#009E73",   marker="s", s=40,  label="D: random benign content"),
    "E_benign_filler":  dict(color="#999999",   marker="x", s=28,  label="E: benign filler (len-matched)"),
    "F_real_malicious": dict(color="#D55E00",   marker="P", s=45,  label="F: real malicious (BIPIA)", alpha=0.6),
    "G_real_benign":    dict(color="#56B4E9",   marker="D", s=35,  label="G: real benign (dolly)", alpha=0.6),
    "H_offtopic_malicious": dict(color="#CC79A7", marker="v", s=42, label="H: off-topic instr. malicious", alpha=0.85),
}

# Continuous scales: viridis (perceptually uniform, CVD-safe) for the num gradient;
# PuOr_r (purple<->orange diverging, avoids red/green) for the P(injection) boundary.
_NUM_CMAP = "viridis"
_PINJ_CMAP = "PuOr_r"   # warm/orange = flagged (P->1), cool/purple = passes (P->0)


def _scatter(ax, Y, meta, title):
    for g, st in _GROUP_STYLE.items():
        idx = [i for i, m in enumerate(meta) if m["group"] == g]
        if idx:
            ax.scatter(Y[idx, 0], Y[idx, 1], **{k: v for k, v in st.items()})
    # C trajectory coloured by num
    c_idx = [i for i, m in enumerate(meta) if m["group"] == "C_instr_malicious"]
    sc = ax.scatter(Y[c_idx, 0], Y[c_idx, 1], c=[meta[i]["num"] for i in c_idx],
                    cmap=_NUM_CMAP, s=45, marker="^", edgecolors="k", linewidths=0.3,
                    label="C: instr. malicious (by num)", zorder=4)
    # connect per-num means of C to show the path
    by_num = defaultdict(list)
    for i in c_idx:
        by_num[meta[i]["num"]].append(i)
    path = np.array([Y[by_num[n]].mean(axis=0) for n in sorted(by_num)])
    ax.plot(path[:, 0], path[:, 1], color="0.35", lw=1.2, alpha=0.8, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def make_scatters(Xs, meta, outdir, model, seed=0):
    n = len(Xs)
    perplexity = max(5, min(30, (n - 1) // 3))
    reducers = {
        "pca": ("PCA", PCA(n_components=2, random_state=seed).fit_transform(Xs)),
        "tsne": (f"t-SNE (perplexity={perplexity})",
                 TSNE(n_components=2, perplexity=perplexity, init="pca",
                      random_state=seed).fit_transform(Xs)),
    }
    try:
        import umap
        reducers["umap"] = ("UMAP (n_neighbors=15, min_dist=0.1)",
                            umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Xs))
    except Exception as e:
        print(f"[RQ4] UMAP unavailable ({e}); skipping.")

    for key, (title, Y) in reducers.items():
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sc = _scatter(ax, Y, meta, f"{model}: [CLS] embeddings — {title}")
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("C: # benign instructions (num)")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"scatter_{key}.png"), dpi=150)
        plt.close(fig)


def plot_axis_projection(Xs, meta, outdir, model):
    '''Guided 2-D view: x = malicious->benign axis (explicit safety direction, from the
    real F/G centroids if present else synthetic A/B), y = top PC of the residual after
    removing that axis. Makes the safety direction an axis instead of hoping PCA finds it.'''
    groups = {m["group"] for m in meta}
    if "F_real_malicious" in groups and "G_real_benign" in groups:
        c_mal, _ = _centroid(Xs, meta, "F_real_malicious")
        c_ben, _ = _centroid(Xs, meta, "G_real_benign")
        axlabel = "real BIPIA→dolly"
    else:
        c_mal, _ = _centroid(Xs, meta, "A_original")
        c_ben, _ = _centroid(Xs, meta, "B_tutorial")
        axlabel = "synthetic A→B"

    u = c_ben - c_mal
    uhat = u / (np.linalg.norm(u) or 1.0)
    span = float((c_ben - c_mal) @ uhat) or 1.0
    x = ((Xs - c_mal) @ uhat) / span                       # 0 = malicious centroid, 1 = benign
    resid = Xs - np.outer(Xs @ uhat, uhat)                 # remove the safety-axis component
    y = PCA(n_components=1, random_state=0).fit_transform(resid - resid.mean(axis=0))[:, 0]

    Y = np.column_stack([x, y])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = _scatter(ax, Y, meta, f"{model}: guided axis projection ({axlabel})")
    ax.axvline(0, ls=":", c="#000000", lw=1); ax.axvline(1, ls=":", c="#0072B2", lw=1)
    ax.set_xlabel(f"malicious → benign axis  ({axlabel};  0=mal centroid, 1=benign)")
    ax.set_ylabel("top residual PC (orthogonal to safety axis)")
    ax.set_xticks([0, 0.5, 1.0])
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("C: # benign instructions (num)")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_axis.png"), dpi=150)
    plt.close(fig)


def plot_pinj_colored(Xs, meta, outdir, model, seed=0):
    '''Same PCA layout, but every point is coloured by its true P(injection) (diverging,
    centred on the 0.5 boundary). Group identity is kept via marker shape. This shows the
    guard's actual decision geometry — e.g. for ProtectAIv2 the benign tutorial group is
    on the injection side while real BIPIA/dolly are on the benign side, which the
    centroid-axis view cannot show.'''
    import matplotlib as mpl
    Y = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = _PINJ_CMAP
    pj = np.array([m["p_injection"] for m in meta])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = None
    for g, st in _GROUP_STYLE.items():
        idx = [i for i, m in enumerate(meta) if m["group"] == g]
        if not idx:
            continue
        sc = ax.scatter(Y[idx, 0], Y[idx, 1], c=pj[idx], cmap=cmap, norm=norm,
                        marker=st["marker"], s=st["s"], edgecolors="k", linewidths=0.3,
                        label=st["label"])
    c_idx = [i for i, m in enumerate(meta) if m["group"] == "C_instr_malicious"]
    sc = ax.scatter(Y[c_idx, 0], Y[c_idx, 1], c=pj[c_idx], cmap=cmap, norm=norm,
                    marker="^", s=55, edgecolors="k", linewidths=0.4,
                    label="C: instr. malicious")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("P(injection)  (orange = flagged, purple = passes; white = 0.5 boundary)")
    ax.set_title(f"{model}: decision geometry — points coloured by P(injection)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_pinj.png"), dpi=150)
    plt.close(fig)


def plot_drift(rows, outdir, model, rows_off=None):
    nums = [r["num"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(nums, [r["drift"] for r in rows], marker="o", color="#CC79A7",
             label="drift, on-topic C (0=mal anchor, 1=benign-instr)")
    if rows_off:
        ax1.plot([r["num"] for r in rows_off], [r["drift"] for r in rows_off],
                 marker="v", ls="--", color="#009E73",
                 label="drift, off-topic H (random_instr filler)")
    ax1.axhline(0, ls=":", c="#000000", lw=1); ax1.axhline(1, ls=":", c="#0072B2", lw=1)
    ax1.set_xlabel("number of benign instructions in the list")
    ax1.set_ylabel("normalized drift toward benign-instruction centroid")
    ax1.set_ylim(-0.1, 1.2)

    ax2 = ax1.twinx()
    ax2.plot(nums, [r["p_injection"] for r in rows], marker="s", color="#0072B2",
             label="P(injection) on-topic C")
    if rows_off:
        ax2.plot([r["num"] for r in rows_off], [r["p_injection"] for r in rows_off],
                 marker="x", ls="--", color="#0072B2", alpha=0.6,
                 label="P(injection) off-topic H")
    ax2.axhline(0.5, ls="--", c="grey", lw=1)
    ax2.set_ylabel("P(injection)"); ax2.set_ylim(-0.02, 1.02)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    plt.title(f"{model}: representational drift vs. instruction count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift_vs_num.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="RQ4 embedding analysis")
    ap.add_argument("--model", default="piguard")
    ap.add_argument("--payload", default="obvious_inj", choices=list(PAYLOADS))
    ap.add_argument("--sample", default="bare")
    ap.add_argument("--max-num", type=int, default=MAX_NUM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-real", type=int, default=0,
                    help="add N real BIPIA injections + N dolly tutorials as endpoint clouds")
    ap.add_argument("--template-real", action="store_true",
                    help="wrap real BIPIA/dolly payloads in the carrier template (removes format confound)")
    args = ap.parse_args()

    sample = next(s for s in SAMPLES if s["id"] == args.sample)
    template, payload = sample["template"], PAYLOADS[args.payload]
    tag = f"rq4_{args.model}_{args.payload}" + ("_real" if args.n_real else "")
    if args.n_real and args.template_real:
        tag += "tpl"
    outdir = os.path.join(HERE, "figures", tag)
    os.makedirs(outdir, exist_ok=True)

    extractor = CLSExtractor(args.model)
    X, meta = build_points(extractor, template, payload, max_num=args.max_num,
                           seed=args.seed, n_real=args.n_real, template_real=args.template_real)
    print(f"[RQ4] built {len(X)} points, dim={X.shape[1]}")

    # standardize (DeBERTa [CLS] is anisotropic) — all metrics + reductions on Xs
    Xs = StandardScaler().fit_transform(X)

    rows = drift_metrics(Xs, meta)
    has_off = any(m["group"] == "H_offtopic_malicious" for m in meta)
    rows_off = drift_metrics(Xs, meta, group="H_offtopic_malicious") if has_off else None
    make_scatters(Xs, meta, outdir, args.model, seed=args.seed)
    plot_axis_projection(Xs, meta, outdir, args.model)
    plot_pinj_colored(Xs, meta, outdir, args.model, seed=args.seed)
    plot_drift(rows, outdir, args.model, rows_off=rows_off)

    # persist embeddings + metadata
    np.savez_compressed(os.path.join(DATA_DIR, f"{tag}.npz"),
                        X=X, Xs=Xs,
                        group=np.array([m["group"] for m in meta]),
                        num=np.array([m["num"] for m in meta]),
                        position=np.array([m["position"] for m in meta]),
                        p_injection=np.array([m["p_injection"] for m in meta]))
    with open(os.path.join(DATA_DIR, f"{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "group": "C_instr_malicious"}) + "\n")
        if rows_off:
            for r in rows_off:
                f.write(json.dumps({**r, "group": "H_offtopic_malicious"}) + "\n")

    print(f"[RQ4] figures + {tag}.(npz|jsonl) written ({outdir})")
    print(f"[RQ4] on-topic  C: drift @0 -> {rows[0]['drift']:.3f}, @{rows[-1]['num']} -> {rows[-1]['drift']:.3f} ; "
          f"P(inj) @0 -> {rows[0]['p_injection']:.3f}, @{rows[-1]['num']} -> {rows[-1]['p_injection']:.3f}")
    if rows_off:
        print(f"[RQ4] off-topic H: drift @0 -> {rows_off[0]['drift']:.3f}, @{rows_off[-1]['num']} -> {rows_off[-1]['drift']:.3f} ; "
              f"P(inj) @0 -> {rows_off[0]['p_injection']:.3f}, @{rows_off[-1]['num']} -> {rows_off[-1]['p_injection']:.3f}")


if __name__ == "__main__":
    main()
