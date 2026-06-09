'''
RQ1 visualization: read results.jsonl and plot guard behaviour vs. the number of
benign instructions, for the benign and malicious tracks.

Produces figures (x = number of benign instructions):
  1. risk_score : mean P(injection) with +/-1 std band   -> does risk drop?
  2. pass_rate  : fraction predicted benign               -> does pass rate rise?
  3. confidence : mean predicted-class confidence         -> single-input confidence
  4. logit      : mean raw injection-class logit          -> single-input logit

Run from the repo root:
    python -m src.diagnoise.logit_analysis.plot
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

TRACKS = ["benign", "malicious"]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows):
    '''(track, num) -> {mean_p, std_p, pass_rate, mean_margin, n}.'''
    groups = defaultdict(list)
    for r in rows:
        groups[(r["track"], r["num"])].append(r)

    agg = {}
    for key, items in groups.items():
        ps = [it["p_injection"] for it in items]
        confs = [it["confidence"] for it in items]
        logits = [it["logit"] for it in items]              # injection-class logit
        logits_ben = [it["logit_benign"] for it in items]   # benign-class logit
        diffs = [it["logit_benign"] - it["logit_injection"] for it in items]  # benign - injection
        passes = [1.0 if it["passes"] else 0.0 for it in items]
        # Signed by the guard's decision: + when predicted malicious/injection,
        # - when predicted benign (i.e. the input "passes"). Magnitude is the
        # predicted (winning) class's score, so the sign always matches the label.
        signed_conf = [(-c if it["passes"] else c) for c, it in zip(confs, items)]
        signed_logit = [
            (-it["logit_benign"] if it["passes"] else it["logit_injection"])
            for it in items
        ]
        agg[key] = {
            "mean_p": statistics.mean(ps),
            "std_p": statistics.pstdev(ps) if len(ps) > 1 else 0.0,
            "mean_conf": statistics.mean(confs),
            "std_conf": statistics.pstdev(confs) if len(confs) > 1 else 0.0,
            "mean_logit": statistics.mean(logits),
            "std_logit": statistics.pstdev(logits) if len(logits) > 1 else 0.0,
            "mean_logit_ben": statistics.mean(logits_ben),
            "std_logit_ben": statistics.pstdev(logits_ben) if len(logits_ben) > 1 else 0.0,
            "mean_diff": statistics.mean(diffs),
            "std_diff": statistics.pstdev(diffs) if len(diffs) > 1 else 0.0,
            "mean_signed_conf": statistics.mean(signed_conf),
            "std_signed_conf": statistics.pstdev(signed_conf) if len(signed_conf) > 1 else 0.0,
            "mean_signed_logit": statistics.mean(signed_logit),
            "std_signed_logit": statistics.pstdev(signed_logit) if len(signed_logit) > 1 else 0.0,
            "pass_rate": statistics.mean(passes),
            "n": len(items),
        }
    return agg


def _series(agg, track, field):
    nums = sorted({num for (t, num) in agg if t == track})
    return nums, [agg[(track, num)][field] for num in nums]


def plot_risk_score(agg, out_path):
    plt.figure(figsize=(7, 4.5))
    for track in TRACKS:
        nums, means = _series(agg, track, "mean_p")
        if not nums:
            continue
        _, stds = _series(agg, track, "std_p")
        plt.plot(nums, means, marker="o", label=track)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        plt.fill_between(nums, lo, hi, alpha=0.15)
    plt.axhline(0.5, ls="--", c="grey", lw=1, label="decision threshold")
    plt.xlabel("number of benign instructions")
    plt.ylabel("risk score  P(injection)")
    plt.title("PIGuard risk score vs. instruction count")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pass_rate(agg, out_path):
    plt.figure(figsize=(7, 4.5))
    for track in TRACKS:
        nums, rates = _series(agg, track, "pass_rate")
        if not nums:
            continue
        plt.plot(nums, rates, marker="o", label=track)
    plt.xlabel("number of benign instructions")
    plt.ylabel("guard pass rate (fraction predicted benign)")
    plt.title("PIGuard pass rate vs. instruction count")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_signed(agg, out_path, metric="confidence", model="guard"):
    '''Single figure: sign = guard decision (+malicious / -benign), magnitude = metric.

    metric: "confidence" (predicted-class probability) or "logit" (injection logit).
    A line above 0 means the guard flags the input; dipping below 0 means it now
    passes (lets the input through).
    '''
    if metric == "confidence":
        mean_field, std_field = "mean_signed_conf", "std_signed_conf"
        ylabel = "signed confidence  (+injection / -benign)"
    elif metric == "logit":
        mean_field, std_field = "mean_signed_logit", "std_signed_logit"
        ylabel = "signed logit  (+injection / -benign)"
    else:
        raise ValueError("metric must be 'confidence' or 'logit'")

    plt.figure(figsize=(7, 4.5))
    for track in TRACKS:
        nums, means = _series(agg, track, mean_field)
        if not nums:
            continue
        _, stds = _series(agg, track, std_field)
        plt.plot(nums, means, marker="o", label=f"{track} input")
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        plt.fill_between(nums, lo, hi, alpha=0.15)
    plt.axhline(0.0, ls="--", c="grey", lw=1, label="decision boundary")
    plt.xlabel("number of benign instructions")
    plt.ylabel(ylabel)
    plt.title(f"{model} signed {metric} vs. instruction count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confidence(agg, out_path):
    plt.figure(figsize=(7, 4.5))
    for track in TRACKS:
        nums, means = _series(agg, track, "mean_conf")
        if not nums:
            continue
        _, stds = _series(agg, track, "std_conf")
        plt.plot(nums, means, marker="o", label=track)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        plt.fill_between(nums, lo, hi, alpha=0.15)
    plt.xlabel("number of benign instructions")
    plt.ylabel("guard confidence (predicted-class probability)")
    plt.title("PIGuard confidence vs. instruction count")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_logit(agg, out_path, model="guard"):
    '''Three panels: injection-class logit, benign-class logit, and their
    difference (decision margin = injection - benign). Raw values, no sign
    rewriting. In the margin panel, 0 is the actual decision boundary.'''
    panels = [
        ("mean_logit", "std_logit", "injection-class logit", None),
        ("mean_logit_ben", "std_logit_ben", "benign-class logit", None),
        ("mean_diff", "std_diff", "difference (benign - injection)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, (mean_f, std_f, ylabel, boundary) in zip(axes, panels):
        for track in TRACKS:
            nums, means = _series(agg, track, mean_f)
            if not nums:
                continue
            _, stds = _series(agg, track, std_f)
            ax.plot(nums, means, marker="o", label=track)
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(nums, lo, hi, alpha=0.15)
        if boundary is not None:
            ax.axhline(boundary, ls="--", c="grey", lw=1, label="decision boundary")
        ax.set_xlabel("number of benign instructions")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"{model} logits vs. instruction count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pben_by(rows, group_field, track="malicious"):
    '''mean P(benign) grouped by (group_value, num); returns {group_value: (nums, means)}.'''
    g = defaultdict(list)
    for r in rows:
        if track is not None and r["track"] != track:
            continue
        g[(r[group_field], r["num"])].append(1.0 - r["p_injection"])
    out = defaultdict(dict)
    for (gv, num), vals in g.items():
        out[gv][num] = statistics.mean(vals)
    return {gv: (sorted(d), [d[n] for n in sorted(d)]) for gv, d in out.items()}


def plot_by_case(rows, out_path, model="guard"):
    '''P(malicious)=P(injection) vs num, for the benign and malicious scenarios,
    per carrier. With a single carrier this is just two lines (benign/malicious);
    with several, colour = carrier and linestyle = scenario.'''
    cases = sorted({r["sample_id"] for r in rows})
    single = len(cases) <= 1
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    track_style = {"malicious": "-", "benign": "--"}
    track_marker = {"malicious": "o", "benign": "s"}
    track_color = {"malicious": cycle[1], "benign": cycle[0]}

    plt.figure(figsize=(7, 4.5))
    for ci, case in enumerate(cases):
        for track in ("malicious", "benign"):
            sub = [r for r in rows if r["sample_id"] == case and r["track"] == track]
            if not sub:
                continue
            g = defaultdict(list)
            for r in sub:
                g[r["num"]].append(r["p_injection"])
            nums = sorted(g)
            means = [statistics.mean(g[n]) for n in nums]
            color = track_color[track] if single else cycle[ci % len(cycle)]
            label = track if single else f"{case} ({track})"
            plt.plot(nums, means, color=color, ls=track_style[track],
                     marker=track_marker[track], label=label)

    plt.axhline(0.5, ls=":", c="grey", lw=1, label="decision boundary")
    plt.xlabel("number of benign instructions")
    plt.ylabel("P(malicious) = P(injection)  -> higher = flagged")
    plt.title(f"{model}: P(malicious) vs. instruction count")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_position_by_case(rows, out_path):
    '''Position effect of the injected command: small multiples, one panel per
    carrier, lines = injection position (begin/middle/end). Malicious track only.'''
    mal = [r for r in rows if r["track"] == "malicious"]
    cases = sorted({r["sample_id"] for r in mal})
    positions = ["begin", "middle", "end"]

    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.5), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = [r for r in mal if r["sample_id"] == case]
        for pos in positions:
            series = _pben_by([r for r in case_rows if r["position"] == pos], "position", track=None)
            if pos not in series:
                continue
            nums, means = series[pos]
            ax.plot(nums, means, marker="o", label=pos)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_title(case)
        ax.set_xlabel("number of benign instructions")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(title="payload position")
    axes[0].set_ylabel("P(benign)  -> higher = guard passes")
    fig.suptitle("Effect of injected-command position, per carrier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RQ1 results")
    parser.add_argument("--model", default="piguard",
                        help="guard model name; selects results_<model>.jsonl and figures/<model>/")
    parser.add_argument("--in", dest="in_path", default=None,
                        help="override input JSONL (default: results_<model>.jsonl)")
    parser.add_argument("--outdir", default=None,
                        help="override output dir (default: figures/<model>)")
    parser.add_argument("--metric", choices=["confidence", "logit"], default="confidence",
                        help="magnitude shown in the single signed figure")
    parser.add_argument("--payload", default=None,
                        help="restrict malicious track to one payload (e.g. cmd_exec, obvious_inj)")
    parser.add_argument("--style", default="list",
                        help="restrict to one framing style (list | narrative); blank for all")
    parser.add_argument("--content", default="login",
                        help="restrict to one benign content pool (login | random); blank for all")
    parser.add_argument("--all", action="store_true",
                        help="also emit the separate risk/pass/confidence/logit figures")
    args = parser.parse_args()

    if args.in_path is None:
        args.in_path = os.path.join(DATA_DIR, f"results_{args.model}.jsonl")
    if args.outdir is None:
        suffix = f"_{args.payload}" if args.payload else ""
        args.outdir = os.path.join(HERE, "figures", args.model + suffix)

    rows = load(args.in_path)
    if args.style:  # restrict to one framing style (rows lacking the field pass through)
        rows = [r for r in rows if r.get("style", "list") == args.style]
    if args.content:  # restrict to one benign content pool
        rows = [r for r in rows if r.get("content", "login") == args.content]
    if args.payload:  # keep benign track + only the requested malicious payload
        rows = [r for r in rows if r["track"] == "benign" or r.get("payload") == args.payload]
    agg = aggregate(rows)
    os.makedirs(args.outdir, exist_ok=True)

    if args.metric == "confidence":
        out = os.path.join(args.outdir, "rq1_signed_confidence.png")
        plot_signed(agg, out, metric="confidence", model=args.model)
    else:  # logit: report the real injection-class logit, no decision sign
        out = os.path.join(args.outdir, "rq1_logit.png")
        plot_logit(agg, out, model=args.model)
    print(f"[RQ1] wrote {out}")

    # Breakdowns: per carrier, and the injected-command position effect.
    by_case = os.path.join(args.outdir, "rq1_by_case.png")
    plot_by_case(rows, by_case, model=args.model)
    print(f"[RQ1] wrote {by_case}")
    by_pos = os.path.join(args.outdir, "rq1_position_by_case.png")
    plot_position_by_case(rows, by_pos)
    print(f"[RQ1] wrote {by_pos}")

    if args.all:
        plot_risk_score(agg, os.path.join(args.outdir, "rq1_risk_score.png"))
        plot_pass_rate(agg, os.path.join(args.outdir, "rq1_pass_rate.png"))
        plot_confidence(agg, os.path.join(args.outdir, "rq1_confidence.png"))
        plot_logit(agg, os.path.join(args.outdir, "rq1_logit.png"))
        print(f"[RQ1] also wrote breakdown figures to {args.outdir}")


if __name__ == "__main__":
    main()
