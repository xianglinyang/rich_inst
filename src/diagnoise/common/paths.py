'''
Central path helpers for the diagnoise analyses.

Two kinds of artifact, two destinations:
  - Raw outputs (.jsonl / .npz / .json)  -> OUTSIDE the repo, under DATA_ROOT.
  - Figures (.png)                       -> INSIDE the repo, under
                                            src/diagnoise/<analysis>/figures/.

Each analysis passes its package name ("logit_analysis", "embedding_analysis",
"attn_analysis") so artifacts land in a per-analysis subfolder.

Override the data root with the RICH_INST_DATA_ROOT env var if needed.
'''
import os

HERE = os.path.dirname(os.path.abspath(__file__))      # .../src/diagnoise/common
DIAGNOISE = os.path.dirname(HERE)                       # .../src/diagnoise

DATA_ROOT = os.environ.get("RICH_INST_DATA_ROOT", "/data2/xianglin/rich_inst")


def data_dir(analysis):
    '''Raw-output dir for `analysis`, created on demand. Lives outside the repo.'''
    d = os.path.join(DATA_ROOT, analysis)
    os.makedirs(d, exist_ok=True)
    return d


def fig_dir(analysis, *sub):
    '''Figure dir for `analysis` (optionally a sub-path), created on demand.
    Lives inside the repo so figures are version-controlled alongside the code.'''
    d = os.path.join(DIAGNOISE, analysis, "figures", *sub)
    os.makedirs(d, exist_ok=True)
    return d
