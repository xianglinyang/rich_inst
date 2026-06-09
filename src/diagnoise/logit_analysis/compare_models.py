'''
Cross-model comparison for RQ1: overlay the malicious-track P(benign) vs.
instruction count for several guard models in one figure.

Reads results_<model>.jsonl (produced by cal_logits.py --model <model>) for each
requested model and plots the mean P(benign) on the malicious track (averaged
over carriers and positions). Higher = the guard passes the injection.

Run from the repo root:
    python -m src.diagnoise.logit_analysis.compare_models
    python -m src.diagnoise.logit_analysis.compare_models --models piguard protectaiv2 promptguard86m
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
DEFAULT_MODELS = ["piguard", "protectaiv2", "promptguard86m"]


def load(model):
    path = os.path.join(DATA_DIR, f"results_{model}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pben_vs_num(rows, track, payload=None):
    g = defaultdict(list)
    for r in rows:
        if r["track"] != track:
            continue
        if payload is not None and r.get("payload") != payload:
            continue
        g[r["num"]].append(1.0 - r["p_injection"])
    nums = sorted(g)
    return nums, [statistics.mean(g[n]) for n in nums]


def payloads_in(rows):
    return sorted({r.get("payload", "none") for r in rows if r["track"] == "malicious"})


# distinct linestyle per payload so (model=colour, payload=linestyle) reads clearly
_PAYLOAD_STYLE = {"cmd_exec": "-", "obvious_inj": "--"}
_PAYLOAD_MARKER = {"cmd_exec": "o", "obvious_inj": "s"}


def main():
    parser = argparse.ArgumentParser(description="Overlay RQ1 results across guard models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out", default=os.path.join(HERE, "figures", "compare_models.png"))
    args = parser.parse_args()

    plt.figure(figsize=(9, 5.5))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = []
    for i, model in enumerate(args.models):
        rows = load(model)
        if rows is None:
            print(f"[cmp] skip {model}: no results_{model}.jsonl")
            continue
        color = cycle[i % len(cycle)]
        for payload in payloads_in(rows):
            nums, mal = pben_vs_num(rows, "malicious", payload=payload)
            plt.plot(nums, mal, color=color,
                     ls=_PAYLOAD_STYLE.get(payload, "-"),
                     marker=_PAYLOAD_MARKER.get(payload, "o"),
                     label=f"{model} / {payload}")
        plotted.append(model)

    plt.axhline(0.5, ls=":", c="grey", lw=1, label="decision boundary")
    plt.xlabel("number of benign instructions")
    plt.ylabel("P(benign)  -> higher = guard passes the injection")
    plt.title("Instructionalization vs. guard model & payload (malicious track)")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[cmp] wrote {args.out} for models: {', '.join(plotted)}")


if __name__ == "__main__":
    main()
