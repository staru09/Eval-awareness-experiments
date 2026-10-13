"""Join the real-user rows (label 0) of one score file with every row of another, so the
reference classify_auroc.py can compare a benchmark with a fixed WildChat group.

    python reference/combine.py WILDCHAT.npz BENCHMARK.npz OUT.npz
"""
import sys

import numpy as np

wild, bench, out = sys.argv[1:]
w, b = np.load(wild, allow_pickle=True), np.load(bench, allow_pickle=True)
keep = w["labels"] == 0
rows = {k: np.concatenate([w[k][keep], b[k]]) for k in w.files if k.startswith("scores_") or k in ("labels", "source", "prompt_ids")}
np.savez(out, **rows, **{k: w[k] for k in w.files if k not in rows})
