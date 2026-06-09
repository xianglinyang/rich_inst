'''
RQ1 summary metrics per guard model and payload.

For each model x payload, on the malicious track (mean over carriers/positions):
  - pass@0    : P(guard passes) with NO benign instructions  (raw detectability)
  - pass@max  : P(guard passes) at the max instruction count
  - asr@0/max : attack success rate = fraction of (carrier x position) samples the
                guard lets through (pred == benign) at num 0 / max
  - instr_gain: asr@max - asr@0  (how much instructionalization helps the attacker)

"obvious-vs-subtle gap" = the difference in detectability between the overt
injection (obvious_inj) and the subtle command payload (cmd_exec).

Run from the repo root:
    python -m src.diagnoise.logit_analysis.summary
'''
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("logit_analysis")
MODELS = ["piguard", "protectaiv2", "promptguard86m"]


def load(model):
    path = os.path.join(DATA_DIR, f"results_{model}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(rows, payload):
    mal = [r for r in rows if r["track"] == "malicious" and r.get("payload") == payload]
    nums = sorted({r["num"] for r in mal})
    if not nums:
        return None
    lo, hi = nums[0], nums[-1]
    by = defaultdict(list)
    for r in mal:
        by[r["num"]].append(r)

    def pben(n):
        return statistics.mean(1.0 - r["p_injection"] for r in by[n])

    def asr(n):  # attack success rate = fraction predicted benign (passes)
        return statistics.mean(1.0 if r["passes"] else 0.0 for r in by[n])

    return {
        "pass@0": pben(lo), "pass@max": pben(hi),
        "asr@0": asr(lo), "asr@max": asr(hi),
        "instr_gain": asr(hi) - asr(lo), "max_num": hi,
    }


def main():
    payloads = ["cmd_exec", "obvious_inj"]
    print(f"{'model':14} {'payload':12} {'pass@0':>7} {'pass@max':>9} "
          f"{'asr@0':>6} {'asr@max':>8} {'instr_gain':>11}")
    table = {}
    for model in MODELS:
        rows = load(model)
        if rows is None:
            print(f"{model:14} (no results file)")
            continue
        table[model] = {}
        for p in payloads:
            m = metrics(rows, p)
            if m is None:
                continue
            table[model][p] = m
            print(f"{model:14} {p:12} {m['pass@0']:7.3f} {m['pass@max']:9.3f} "
                  f"{m['asr@0']:6.2f} {m['asr@max']:8.2f} {m['instr_gain']:+11.2f}")

    print("\nObvious-vs-subtle detectability gap (asr@0: subtle - obvious; "
          "higher = subtle payload slips by more):")
    for model, t in table.items():
        if "cmd_exec" in t and "obvious_inj" in t:
            gap = t["cmd_exec"]["asr@0"] - t["obvious_inj"]["asr@0"]
            print(f"  {model:14} {gap:+.2f}  "
                  f"(subtle asr@0={t['cmd_exec']['asr@0']:.2f}, "
                  f"obvious asr@0={t['obvious_inj']['asr@0']:.2f})")

    out = os.path.join(DATA_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n[sum] wrote {out}")


if __name__ == "__main__":
    main()
