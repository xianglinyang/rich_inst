'''
Real-world reference clusters for the RQ4 embedding analysis.

Two endpoint clouds, sampled reproducibly:
  - load_bipia_injections : pure malicious commands (BIPIA text + code attacks, local)
  - load_dolly_tutorials  : benign tutorial-style instructions (dolly-15k, HF)

These give *valid* (large, real) malicious and benign anchors to compare the synthetic
`construct_prompt` trajectory against, instead of the few hand-written anchors.
'''
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BIPIA_DIR = os.path.join(REPO, "data", "bipia")

# self-contained, instruction-like dolly categories (no required `context` passage)
DOLLY_TUTORIAL_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")


def load_bipia_injections(n=150, seed=0, kinds=("text", "code")):
    '''Flatten BIPIA *_attack_*.json ({category: [payloads]}) into a deduped list of
    pure injection strings, then sample `n` with a fixed seed.'''
    payloads = []
    for kind in kinds:
        for path in glob.glob(os.path.join(BIPIA_DIR, f"{kind}_attack_*.json")):
            with open(path) as f:
                d = json.load(f)
            for items in d.values():
                payloads.extend(s.strip() for s in items if isinstance(s, str) and s.strip())
    payloads = sorted(set(payloads))
    rng = random.Random(seed)
    rng.shuffle(payloads)
    return payloads[:n] if n else payloads


def load_dolly_tutorials(n=150, seed=0, categories=DOLLY_TUTORIAL_CATEGORIES):
    '''Sample `n` self-contained instruction strings from dolly-15k (HF download).'''
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    cats = set(categories)
    insts = sorted({r["instruction"].strip() for r in ds
                    if r["category"] in cats and r["instruction"].strip()})
    rng = random.Random(seed)
    rng.shuffle(insts)
    return insts[:n] if n else insts


if __name__ == "__main__":
    mal = load_bipia_injections(5)
    ben = load_dolly_tutorials(5)
    print("BIPIA injections (sample):")
    for s in mal:
        print("  -", s[:90].replace("\n", " "))
    print("Dolly tutorials (sample):")
    for s in ben:
        print("  -", s[:90])
