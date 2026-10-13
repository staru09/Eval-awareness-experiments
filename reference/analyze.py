"""Studies 2 and 3: the probe against 100 random directions given the same choices, plus the
paper's tables and figures.

    python reference/analyze.py WORKDIR/results OUTDIR

Reference convention: the direction is real-user minus evaluation phrases and the benchmark is
label 1, so AUROC above 0.5 means the benchmark scores more like real users. Layer index is the
hidden-state index; 0 (embeddings) is excluded from every search. For each comparison:
  best layer       the reference headline, best layer on the test set
  best random      the best random direction at that same layer (the reference check)
  own best         random directions at least as good when each picks its own best layer
  either sign      the layer furthest from 0.5, random directions also picking layer and sign
  calibrated       the layer where the calibration set separates in the intended direction,
                   read on the test set in that direction; random directions get the same choice
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#b9b8b3", "#1a1a1a"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555"})
SAFETY = [("deceptionbench", "DeceptionBench"), ("wmdp", "WMDP"), ("halueval", "HaluEval"),
          ("model_written_evals", "Model-written evals"), ("sycophancy_eval", "Sycophancy-eval"),
          ("wildjailbreak", "WildJailbreak"), ("do_not_answer", "Do-Not-Answer")]


def auc(s, y):
    r = rankdata(s, axis=0)
    n1 = y.sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def curves(path, pool):
    z = np.load(path, allow_pickle=True)
    y = z["labels"]
    return auc(z[f"scores_{pool}_real"], y)[1:], auc(z[f"scores_{pool}_random"], y)[1:]   # [L], [L, R]


def stats(test, pool):
    a, r = curves(test, pool)
    ac, rc = curves(R / "train.npz", pool)
    k = int(a.argmax())
    f, rf = np.abs(a - 0.5), np.abs(r - 0.5)
    kf = int(f.argmax())
    up = int((1 - ac).argmax())
    r_up = np.array([1 - r[int((1 - rc[:, j]).argmax()), j] for j in range(r.shape[1])])
    return {
        "n_layers": len(a), "curve": a.round(4).tolist(), "calib_curve": ac.round(4).tolist(),
        "rand_p5": np.percentile(r, 5, 1).round(4).tolist(), "rand_p95": np.percentile(r, 95, 1).round(4).tolist(),
        "best_layer": k + 1, "best_auroc": round(float(a[k]), 3),
        "best_random_same_layer": round(float(r[k].max()), 3),
        "random_own_best_at_least": int((r.max(0) >= a[k]).sum()),
        "either_sign_layer": kf + 1, "either_sign_auroc": round(float(a[kf]), 3),
        "either_sign_dist": round(float(f[kf]), 3),
        "random_either_sign_at_least": int((rf.max(0) >= f[kf]).sum()),
        "random_either_sign_dists": rf.max(0).round(4).tolist(),
        "intended_calib_layer": up + 1, "intended_calib_test": round(float(1 - a[up]), 3),
        "random_intended_calib_at_least": int((r_up >= 1 - a[up]).sum()),
    }


res = {"MASK, 100": stats(R / "test.npz", "last"), "MASK, 1000": stats(R / "mask1k_vs_wild.npz", "last"),
       "MMLU-Lite + options": stats(R / "mmlu_with_options_vs_wild.npz", "last"),
       "MMLU-Lite, question": stats(R / "mmlu_question_only_vs_wild.npz", "last"),
       **{f"{s}/{p}": stats(R / f"safety_{s}_vs_wild.npz", p) for s, _ in SAFETY for p in ("last", "mean")}}
(OUT / "numbers.json").write_text(json.dumps(res, indent=1))

head = ("\\begin{tabular}{llccccc}\n\\toprule\n"
        "Model & Test set & Best layer & Best random & Random $\\geq$ & Calibrated & Random $\\geq$ \\\\\n"
        " & & AUROC (layer) & same layer & own best & intended (layer) & own calib. \\\\\n\\midrule\n")
tail = "\n\\bottomrule\n\\end{tabular}\n"


def row(label):
    s = res[label]
    return (f"Qwen3-8B & {label} & {s['best_auroc']:.3f} ({s['best_layer']}) & {s['best_random_same_layer']:.3f} & "
            f"{s['random_own_best_at_least']} & {s['intended_calib_test']:.3f} ({s['intended_calib_layer']}) & "
            f"{s['random_intended_calib_at_least']} \\\\")


(OUT / "table_repro.tex").write_text(head + "\n".join(map(row, ("MASK, 100", "MASK, 1000"))) + tail)
(OUT / "table_mmlu.tex").write_text(head + "\n".join(map(row, ("MMLU-Lite + options", "MMLU-Lite, question"))) + tail)

rows = []
for s, label in SAFETY:
    a, b = res[f"{s}/last"], res[f"{s}/mean"]
    rows.append(f"{label} & {a['best_auroc']:.3f} ({a['best_layer']}) & {a['best_random_same_layer']:.3f} & "
                f"{a['either_sign_auroc']:.3f} ({a['either_sign_layer']}) & {a['random_either_sign_at_least']} & "
                f"{a['intended_calib_test']:.3f} & {a['random_intended_calib_at_least']} & "
                f"{b['either_sign_auroc']:.3f} ({b['either_sign_layer']}) & {b['random_either_sign_at_least']} \\\\")
(OUT / "table_safety.tex").write_text(
    "\\begin{tabular}{lcccccccc}\n\\toprule\n"
    " & \\multicolumn{6}{c}{Last token} & \\multicolumn{2}{c}{Mean over tokens} \\\\\n"
    "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}\n"
    "Benchmark & Best & Best random & Furthest & Rand. & Calib.\\ & Rand. & Furthest & Rand. \\\\\n"
    " & (layer) & same layer & (layer) & $\\geq$ & layer 10 & $\\geq$ & (layer) & $\\geq$ \\\\\n\\midrule\n"
    + "\n".join(rows) + tail)

# MASK vs WildChat by depth, last token
s = res["MASK, 100"]
x = np.arange(1, s["n_layers"] + 1) / s["n_layers"]
fig, ax = plt.subplots(figsize=(5.5, 2.2))
ax.fill_between(x, s["rand_p5"], s["rand_p95"], color=GREY, alpha=0.45, lw=0, label="Random directions, 5th to 95th pct")
ax.plot(x, s["curve"], color=BLUE, lw=1.4, label="Probe, test")
ax.plot(x, s["calib_curve"], color=ORANGE, lw=1.1, ls="--", label="Probe, calibration")
ax.axhline(0.5, color="#888888", lw=0.6)
ax.set(title="Qwen3-8B", xlabel="Relative depth", ylabel="AUROC")
fig.legend(*ax.get_legend_handles_labels(), fontsize=6.5, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "fig_repro.pdf")

# Safety benchmarks: probe vs 100 random directions, each with its own best layer and sign
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
for ax, pool, title in zip(axes, ("last", "mean"), ("Last token", "Mean over tokens")):
    for i, (s, _) in enumerate(SAFETY):
        st = res[f"{s}/{pool}"]
        rd = np.array(st["random_either_sign_dists"])
        ax.scatter(rd, i + np.random.default_rng(i).uniform(-0.18, 0.18, len(rd)), s=3, color=GREY, lw=0)
        ax.scatter([st["either_sign_dist"]], [i], s=16, color=BLUE, zorder=3)
    ax.set_yticks(range(len(SAFETY)), [l for _, l in SAFETY])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"Best $|\mathrm{AUROC}-0.5|$ over layers")
fig.tight_layout()
fig.savefig(OUT / "fig_safety.pdf")

for k, v in res.items():
    print(f"{k:28s} best {v['best_auroc']:.3f} (L{v['best_layer']}) rand {v['best_random_same_layer']:.3f} "
          f"own-best {v['random_own_best_at_least']:3d} | either {v['either_sign_auroc']:.3f} (L{v['either_sign_layer']}) "
          f"{v['random_either_sign_at_least']:3d} | calib {v['intended_calib_test']:.3f} (L{v['intended_calib_layer']}) "
          f"{v['random_intended_calib_at_least']:3d}")
