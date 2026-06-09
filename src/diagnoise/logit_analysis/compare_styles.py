'''
Compare framing styles for RQ1: does surrounding the malicious instruction with
benign *narrative prose* evade the guard as well as a numbered *instruction list*?

Same benign sentences and same injected payload either way; only the framing
differs (list = "1. X\\n2. Y"; narrative = "X Y Z" prose paragraph). Reads
results_<model>.jsonl (which carries a `style` field) and overlays P(injection)
vs num, one line per style, for the malicious track (benign track dashed).

Run from the repo root:
    python -m src.diagnoise.logit_analysis.compare_styles --model piguard
'''
import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("logit_analysis")
# colour = benign content pool; linestyle = framing style
CONTENT_COLOR = {
    "login": "tab:blue", "login_descr": "tab:cyan",
    "random_instr": "tab:red", "random": "tab:orange",
}
STYLE_LS = {"list": "-", "narrative": "--"}
STYLE_MARKER = {"list": "o", "narrative": "^"}


def load(model):
    path = os.path.join(DATA_DIR, f"results_{model}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pinj_vs_num(rows, track, style, content, payload=None):
    g = defaultdict(list)
    for r in rows:
        if r["track"] != track or r.get("style", "list") != style or r.get("content", "login") != content:
            continue
        if track == "malicious" and payload and r.get("payload") != payload:
            continue
        g[r["num"]].append(r["p_injection"])
    nums = sorted(g)
    return nums, [statistics.mean(g[n]) for n in nums]


def main():
    parser = argparse.ArgumentParser(description="Compare content (login/random) x style (list/narrative)")
    parser.add_argument("--model", default="piguard")
    parser.add_argument("--payload", default="obvious_inj")
    parser.add_argument("--track", default="malicious", choices=["malicious", "benign"])
    parser.add_argument("--styles", nargs="+", default=None,
                        help="restrict to these framing styles (default: all present)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    tag = f"_{'-'.join(args.styles)}" if args.styles else ""
    out = args.out or os.path.join(
        HERE, "figures", f"compare_content_style_{args.model}_{args.payload}_{args.track}{tag}.png")

    rows = load(args.model)
    order = ["login", "login_descr", "random_instr", "random"]
    present = {r.get("content", "login") for r in rows}
    contents = [c for c in order if c in present] + sorted(present - set(order))
    styles = args.styles or sorted({r.get("style", "list") for r in rows})

    plt.figure(figsize=(8.5, 5.5))
    for content in contents:
        for style in styles:
            nums, vals = pinj_vs_num(rows, args.track, style, content, payload=args.payload)
            if not nums:
                continue
            plt.plot(nums, vals, color=CONTENT_COLOR.get(content),
                     ls=STYLE_LS.get(style, "-"), marker=STYLE_MARKER.get(style, "o"),
                     label=f"{content} / {style}")

    plt.axhline(0.5, ls=":", c="grey", lw=1, label="decision boundary")
    plt.xlabel("number of benign sentences")
    plt.ylabel("P(injection)  -> higher = flagged")
    plt.title(f"{args.model}: benign content x framing ({args.track} track, {args.payload})")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[styles] wrote {out}")


if __name__ == "__main__":
    main()
