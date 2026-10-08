import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe import DATA, auroc_from_ranks, bands, ranks

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#b9b8b3", "#1a1a1a"
NAMES = {"qwen3_8b": "Qwen3-8B", "olmo3_7b_think": "Olmo-3-7B-Think"}
# (key, axis label, table label)
SOURCES = [("mask", "MASK", "MASK"), ("model_written_evals", "Model-written", "Model-written evals"),
           ("bbq", "BBQ", "BBQ"), ("mmlu", "MMLU", "MMLU"), ("gsm8k", "GSM8K", "GSM8K")]
COMPARISONS = [("RQ1 tier A vs tier B", "A vs.\\ B: tests vs.\\ users"),
               ("RQ2 tier C vs tier B", "C vs.\\ B: dataset prose vs.\\ users"),
               ("RQ2 support tier A vs tier C", "A vs.\\ C: tests vs.\\ dataset prose"),
               ("replication mask vs wildchat", "Replication: MASK vs.\\ WildChat$^\\dagger$")]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555"})


def top_legend(fig, ax, ncol, top):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.4, w_pad=2.0, rect=(0, 0, 1, top))


def fig_layers(runs, tier, out):
    """AUROC by relative depth over the 5th-95th percentile of random directions at each layer."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5, 1.9), sharey=True, squeeze=False)
    for ax, (key, d) in zip(axes[0], runs):
        z = np.load(d / "scores_prompt.npz", allow_pickle=True)
        for keep, hi, color, label in (
                ((tier == "A") | (tier == "B"), tier == "A", BLUE, "Tests vs. users (A vs. B)"),
                ((tier == "C") | (tier == "B"), tier == "C", ORANGE, "Dataset prose vs. users (C vs. B)")):
            y = hi[keep].astype(int)
            c = auroc_from_ranks(ranks(z["scores_mean_real"][keep]), y)[0]
            r = auroc_from_ranks(ranks(z["scores_mean_random"][keep]), y)[0]
            x = np.arange(len(c)) / (len(c) - 1)
            ax.fill_between(x, np.percentile(r, 5, axis=1), np.percentile(r, 95, axis=1),
                            color=color, alpha=0.13, lw=0)
            ax.plot(x, c, color=color, lw=1.4, label=label)
        ax.axhline(0.5, color="#888888", lw=0.6, ls=":")
        ax.set_title(NAMES.get(key, key), fontsize=8.5, color=INK)
        ax.set_xlabel("Relative layer depth")
        ax.set_xlim(0, 1)
    axes[0][0].set_ylabel("AUROC")
    top_legend(fig, axes[0][0], 2, 0.9)
    fig.savefig(out / "fig_layers.pdf")
    plt.close(fig)


def fig_sources(runs, analyses, out):
    """Each evaluation dataset: probe, random floor and best text-only baseline."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5, 1.75), sharey=True, squeeze=False)
    ys = np.arange(len(SOURCES))[::-1]
    for ax, (key, _) in zip(axes[0], runs):
        a = analyses[key]["prompt / mean"]
        for yv, (s, _, _) in zip(ys, SOURCES):
            c = a[f"RQ3 {s} vs tier B"]
            first = yv == ys[0]
            ax.plot([0.5, c["headline_band_auroc"]], [yv, yv], color=GREY, lw=0.8, zorder=1)
            ax.scatter(c["random_floor_best_of_100"], yv, marker="|", s=90, color=INK, zorder=3,
                       label="Random floor" if first else None)
            ax.scatter(max(c["text_baselines"].values()), yv, marker="x", s=18, color=ORANGE, zorder=3,
                       label="Best text-only" if first else None)
            ax.scatter(c["headline_band_auroc"], yv, s=26, color=BLUE, zorder=4,
                       label="Probe" if first else None)
        ax.axvline(0.5, color="#888888", lw=0.6, ls=":")
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label, _ in SOURCES])
        ax.set_xlim(0.35, 1.0)
        ax.set_ylim(-0.6, len(SOURCES) - 0.4)
        ax.set_xlabel("AUROC vs. real users")
        ax.set_title(NAMES.get(key, key), fontsize=8.5, color=INK)
    top_legend(fig, axes[0][0], 3, 0.88)
    fig.savefig(out / "fig_sources.pdf")
    plt.close(fig)


def fig_null(runs, rows, tier, out):
    """The matched control: each random direction searches layers exactly as the probe does."""
    words = np.array([len(r["text"].split()) for r in rows], dtype=float)
    comps = (("Tests vs. users", (tier == "A") | (tier == "B"), tier == "A"),
             ("Dataset prose vs. users", (tier == "C") | (tier == "B"), tier == "C"))
    fig, axes = plt.subplots(len(comps), len(runs), figsize=(5.5, 3.0), sharex=True, squeeze=False)
    for j, (key, d) in enumerate(runs):
        z = np.load(d / "scores_prompt.npz", allow_pickle=True)
        for i, (title, keep, hi) in enumerate(comps):
            ax = axes[i][j]
            y = hi[keep].astype(int)
            probe = bands(auroc_from_ranks(ranks(z["scores_mean_real"][keep]), y)[0]).max()
            rand = bands(auroc_from_ranks(ranks(z["scores_mean_random"][keep]), y)[0].T).max(1)
            ax.hist(rand, bins=np.arange(0.5, 0.80, 0.01), color=GREY, edgecolor="white", lw=0.4,
                    label="100 random directions, best band")
            ax.axvline(probe, color=BLUE, lw=1.6, label="Probe, best band")
            ax.axvline(auroc_from_ranks(ranks(words[keep]), y)[0], color=ORANGE, lw=1.2, ls="--",
                       label="Word count alone")
            p = ((rand >= probe).sum() + 1) / (len(rand) + 1)
            ax.text(0.98, 0.92, f"probe {probe:.3f}\np = {p:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.8, color=INK)
            if i == 0:
                ax.set_title(NAMES.get(key, key), fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(title, fontsize=7.2)
            if i == len(comps) - 1:
                ax.set_xlabel("Best 3-layer band AUROC")
    top_legend(fig, axes[0][0], 3, 0.92)
    fig.savefig(out / "fig_null.pdf")
    plt.close(fig)


def tables(runs, analyses, out):
    """LaTeX tables read straight from each analysis_phase1.json; no number is typed."""
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAMES.get(k, k)}}}" for k, _ in runs)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs)))
    mark = {"beats all three controls": "\\,\\checkmark",
            "no separation in the expected direction": "\\,$\\downarrow$"}

    lines = [f"\\begin{{tabular}}{{l{'cc' * len(runs)}}}", "\\toprule", f" & {head} \\\\", rules,
             "Comparison & " + " & ".join("AUROC & Rand." for _ in runs) + " \\\\", "\\midrule"]
    for key, label in COMPARISONS:
        readout = "prompt / last" if key.startswith("replication") else "prompt / mean"
        cells = [analyses[k][readout][key] for k, _ in runs]
        lines.append(f"{label} & " + " & ".join(
            f"{c['headline_band_auroc']:.3f}{mark.get(c['verdict'], '')} & {c['random_floor_best_of_100']:.3f}"
            for c in cells) + " \\\\")
    (out / "table_results.tex").write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")

    lines = [f"\\begin{{tabular}}{{l{'cc' * len(runs)}c}}", "\\toprule", f" & {head} & \\\\", rules,
             "Dataset & " + " & ".join("AUROC (95\\% CI) & Rand." for _ in runs) + " & Best text-only \\\\",
             "\\midrule"]
    for s, _, label in SOURCES:
        cells = [analyses[k]["prompt / mean"][f"RQ3 {s} vs tier B"] for k, _ in runs]
        name, val = max(cells[0]["text_baselines"].items(), key=lambda kv: kv[1])   # same prompts
        lines.append(f"{label} & " + " & ".join(
            f"{c['headline_band_auroc']:.3f}{mark.get(c['verdict'], '')} "
            f"({c['band_95ci'][0]:.3f}--{c['band_95ci'][1]:.3f}) & {c['random_floor_best_of_100']:.3f}"
            for c in cells) + f" & {val:.3f} {name} \\\\")
    (out / "table_sources.tex").write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="model_key=run_dir, e.g. qwen3_8b=runs/qwen3")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    runs = [(k, Path(d)) for k, d in (a.split("=", 1) for a in args.runs)]
    analyses = {k: json.loads((d / "analysis_phase1.json").read_text()) for k, d in runs}
    rows = json.loads((DATA / "phase1.json").read_text(encoding="utf-8"))
    tier = np.array([r["category"] for r in rows])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_layers(runs, tier, out)
    fig_sources(runs, analyses, out)
    fig_null(runs, rows, tier, out)
    tables(runs, analyses, out)
    print("wrote 3 figures and 2 tables to", out)


if __name__ == "__main__":
    main()
