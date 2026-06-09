'''
RQ1 driver: does instructionalization shift PIGuard's decision boundary toward
"benign/pass"?

For every base sample we sweep the number of benign instructions `num` from 0 to
len(general_instructions). For each `num` we score two tracks with PIGuard:

  - benign     : the carrier document + `num` benign instructions (no injection)
  - malicious  : same, but with one injected command placed at begin/middle/end

Each (track, num) therefore aggregates over {samples} x {positions}, giving a
real pass rate (a fraction) rather than a single 0/1 outcome. Results are written
as one JSON object per scored prompt to `results.jsonl`; plot.py consumes them.

Run from the repo root:
    python -m src.diagnoise.logit_analysis.cal_logits
'''
import argparse
import json
import os
from typing import List

from src.diagnoise.common.guardrails import MODELS, load_scorer
from src.diagnoise.common.prompts import (
    SAMPLES,
    PAYLOADS,
    STYLES,
    CONTENT_POOLS,
    general_instructions,
    construct_prompt,
)

POSITIONS = ["begin", "middle", "end"]
MAX_NUM = min(20, len(general_instructions))

HERE = os.path.dirname(os.path.abspath(__file__))
from src.diagnoise.common.paths import data_dir
DATA_DIR = data_dir("logit_analysis")


def results_path(model: str) -> str:
    return os.path.join(DATA_DIR, f"results_{model}.jsonl")


def build_prompts(max_num: int):
    '''Return (prompts, metas) for the full RQ1 sweep.'''
    prompts: List[str] = []
    metas: List[dict] = []

    for sample in SAMPLES:
        template = sample["template"]
        for content, pool in CONTENT_POOLS.items():
            cap = min(max_num, len(pool))
            for style in STYLES:
                for num in range(0, cap + 1):
                    common = {"sample_id": sample["id"], "num": num,
                              "style": style, "content": content}
                    # Benign track: no injection. Skip num=0 only when it would
                    # be a degenerate empty string (bare carrier).
                    benign_prompt = construct_prompt(template, num, "end",
                                                     malicious_instruction=None,
                                                     instructions_pool=pool, style=style)
                    if num >= 1 or benign_prompt.strip():
                        prompts.append(benign_prompt)
                        metas.append({**common, "track": "benign",
                                      "position": "none", "payload": "none"})

                    # Malicious track (num=0 is the bare payload): same benign
                    # content + one injected payload, over each payload/position.
                    for pname, ptext in PAYLOADS.items():
                        for pos in POSITIONS:
                            prompts.append(
                                construct_prompt(template, num, pos,
                                                 malicious_instruction=ptext,
                                                 instructions_pool=pool, style=style)
                            )
                            metas.append({**common, "track": "malicious",
                                          "position": pos, "payload": pname})

    return prompts, metas


def run(model: str = "piguard", out_path: str = None, max_num: int = MAX_NUM,
        batch_size: int = 16) -> str:
    out_path = out_path or results_path(model)
    prompts, metas = build_prompts(max_num)
    print(f"[RQ1] model={model} scoring {len(prompts)} prompts "
          f"({len(SAMPLES)} samples x num 0..{max_num} x tracks/positions)")

    scorer = load_scorer(model)
    scores = scorer.score_many(prompts, batch_size=batch_size)

    with open(out_path, "w") as f:
        for meta, score in zip(metas, scores):
            f.write(json.dumps({**meta, **score.as_row()}) + "\n")

    print(f"[RQ1] wrote {len(scores)} rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="RQ1 logit sweep over a guard model")
    parser.add_argument("--model", default="piguard",
                        help=f"guard model: one of {list(MODELS)} or a raw HF id")
    parser.add_argument("--out", default=None,
                        help="output JSONL path (default: results_<model>.jsonl)")
    parser.add_argument("--max-num", type=int, default=MAX_NUM,
                        help=f"max benign instruction count (<= {MAX_NUM})")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(model=args.model, out_path=args.out, max_num=args.max_num, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
