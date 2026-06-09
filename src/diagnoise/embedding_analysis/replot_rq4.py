'''
Regenerate RQ4 figures from cached embeddings — no model forward passes required.

`embedding_analysis.py` saves the standardized [CLS] embeddings + metadata to
`rq4_<tag>.npz` and the per-num drift rows to `rq4_<tag>.jsonl`. This script reloads
those and re-runs only the plotting code, so the figures pick up the current
(colorblind-safe) styling in `embedding_analysis._GROUP_STYLE` / `_NUM_CMAP` /
`_PINJ_CMAP` without needing a GPU, the guard models, or network access.

Run from the repo root:
    python -m src.diagnoise.embedding_analysis.replot_rq4              # all cached tags
    python -m src.diagnoise.embedding_analysis.replot_rq4 rq4_piguard_obvious_inj
'''
import json
import os
import sys

import numpy as np

from src.diagnoise.embedding_analysis.embedding_analysis import (
    make_scatters, plot_axis_projection, plot_pinj_colored, plot_drift,
)

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("embedding_analysis")


def _meta_from_npz(d):
    '''Rebuild the list-of-dicts `meta` the plot functions expect from the npz arrays.'''
    return [
        {"group": str(g), "num": int(n), "position": str(p), "p_injection": float(pj)}
        for g, n, p, pj in zip(d["group"], d["num"], d["position"], d["p_injection"])
    ]


def _model_from_tag(tag):
    # tag = rq4_<model>_<payload>[_real|_realtpl]
    parts = tag.split("_")
    return parts[1] if len(parts) > 1 else tag


def replot(tag, seed=0):
    npz_path = os.path.join(DATA_DIR, f"{tag}.npz")
    if not os.path.exists(npz_path):
        print(f"[replot] skip {tag}: {npz_path} not found")
        return
    d = np.load(npz_path, allow_pickle=True)
    Xs = d["Xs"]
    meta = _meta_from_npz(d)
    model = _model_from_tag(tag)
    outdir = os.path.join(HERE, "figures", tag)
    os.makedirs(outdir, exist_ok=True)

    make_scatters(Xs, meta, outdir, model, seed=seed)
    plot_axis_projection(Xs, meta, outdir, model)
    plot_pinj_colored(Xs, meta, outdir, model, seed=seed)

    jsonl_path = os.path.join(DATA_DIR, f"{tag}.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        plot_drift(rows, outdir, model)

    print(f"[replot] {tag}: figures rewritten -> {outdir} ({len(Xs)} points)")


def main():
    tags = sys.argv[1:]
    if not tags:
        tags = sorted(
            f[:-4] for f in os.listdir(DATA_DIR)
            if f.startswith("rq4_") and f.endswith(".npz")
        )
    for tag in tags:
        replot(tag)


if __name__ == "__main__":
    main()
