import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/runs/llama3_poc"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED, N_RANDOM, N_SHUFFLE = 0, 100, 100
EVAL_WORDS = re.compile(r"\b(test\w*|evaluat\w*|benchmark\w*)\b", re.I)


def pairs():
    """(eval phrases, deployment phrases). In the pairs file the eval side is negative_example."""
    txt = (ROOT / "data/contrastive_pairs_16.txt").read_text(encoding="utf-8")
    return re.findall(r"negative_example: (.+)", txt), re.findall(r"positive_example: (.+)", txt)


def prep(args):
    from datasets import load_dataset
    from prep import HI, LO, clean
    rng = random.Random(SEED)
    keep = lambda t: LO <= len(t) <= HI and not EVAL_WORDS.search(t)
    phase1 = json.loads(Path(args.phase1).read_text(encoding="utf-8"))
    pool = {s: [r["text"] for r in phase1 if r["source"] == s and keep(r["text"])] for s in ("gsm8k", "wildchat")}
    tqa = [clean(q) for q in load_dataset("truthfulqa/truthful_qa", "generation", split="validation")["question"]]
    pool["truthfulqa"] = [t for t in tqa if keep(t)]
    rows = [{"source": s, "label": int(s != "wildchat"), "text": t}
            for s, n in (("gsm8k", 25), ("truthfulqa", 25), ("wildchat", 50)) for t in rng.sample(pool[s], n)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in pool:
        n = sorted(len(r["text"]) for r in rows if r["source"] == s)
        print(f"{s:10s} {len(n)} of {len(pool[s])} in band | chars min/median/max {n[0]}/{n[len(n) // 2]}/{n[-1]}")


def extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    ev, dep = pairs()
    texts = ev + dep + [r["text"] for r in json.loads((OUT / "val.json").read_text(encoding="utf-8"))]
    last, mean = [], []
    for i, t in enumerate(texts):
        s = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
        if i == 0:
            print("model sees:", repr(s))
        ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            h = torch.stack(model(ids, output_hidden_states=True).hidden_states)[:, 0].float()   # [33, T, d]
        last.append(h[:, -1].cpu().numpy())
        mean.append(h[:, 1:].mean(1).cpu().numpy())      # BOS skipped: its norm dwarfs every other token
    np.savez(OUT / "acts.npz", last=np.stack(last), mean=np.stack(mean), n_pairs=len(ev),
             revision=str(getattr(model.config, "_commit_hash", "")))


def auroc(s, y):
    """s: [n, ...] scores, y: [n] 0/1. Label 1 (eval) expected higher."""
    r = rankdata(s, axis=0)
    n1 = y.sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def direction(e, d):
    """e, d: [k, layers, d] -> unit eval-minus-deployment direction and midpoint, per layer."""
    diff = e.mean(0) - d.mean(0)
    with np.errstate(invalid="ignore"):
        return diff / np.linalg.norm(diff, axis=-1, keepdims=True), (e.mean(0) + d.mean(0)) / 2


def score(h, v, m):
    return np.einsum("nld,ld->nl", h - m, v)


def analyze(args):
    rng = np.random.default_rng(SEED)
    z = np.load(OUT / "acts.npz")
    val = json.loads((OUT / "val.json").read_text(encoding="utf-8"))
    y = np.array([r["label"] for r in val])
    src = np.array([r["source"] for r in val])
    k = int(z["n_pairs"])
    res = {"model": MODEL, "revision": str(z["revision"]), "n_val": {s: int((src == s).sum()) for s in set(src)},
           "chars_auroc": round(float(auroc(np.array([len(r["text"]) for r in val]), y)), 4)}
    for pool in ("last", "mean"):
        a = z[pool]
        e, d, h = a[:k], a[k:2 * k], a[2 * k:]
        v, m = direction(e, d)
        s = score(h, v, m)
        curve = auroc(s, y)
        L = 1 + int(curve[1:].argmax())      # layer 0 is the embedding, constant at the last token

        r = rng.standard_normal((N_RANDOM, a.shape[-1]))
        rand = auroc(np.einsum("nld,rd->nlr", h, r / np.linalg.norm(r, axis=1, keepdims=True)), y)   # [layers, R]
        shuf = []
        for _ in range(N_SHUFFLE):
            flip = (rng.random(k) < 0.5)[:, None, None]
            shuf.append(auroc(score(h, *direction(np.where(flip, d, e), np.where(flip, e, d))), y))
        shuf = np.array(shuf)                                                                           # [S, layers]
        lopo = np.array([(direction(np.delete(e, i, 0), np.delete(d, i, 0))[0] * v).sum(-1) for i in range(k)])

        sl, half_gap = s[:, L], np.linalg.norm(e[:, L].mean(0) - d[:, L].mean(0)) / 2
        thr = np.unique(sl)
        tpr, fpr = (sl[y == 1, None] >= thr).mean(0), (sl[y == 0, None] >= thr).mean(0)
        j = int((tpr - fpr).argmax())
        tau5 = float(np.quantile(sl[y == 0], 0.95))
        res[pool] = {
            "best_layer": L, "val_auroc": round(float(curve[L]), 4),
            "random_at_best_layer": {"p95": round(float(np.percentile(rand[L], 95)), 4), "max": round(float(rand[L].max()), 4)},
            "shuffle_at_best_layer": {"p95": round(float(np.percentile(shuf[:, L], 95)), 4), "max": round(float(shuf[:, L].max()), 4)},
            "per_source_auroc_at_best_layer": {s_: round(float(auroc(sl[(src == s_) | (y == 0)], y[(src == s_) | (y == 0)])), 4)
                                               for s_ in ("gsm8k", "truthfulqa")},
            "early_layers_1_3_max_auroc": round(float(curve[1:4].max()), 4),
            "train_sanity_at_best_layer": {"eval_above_0": int((score(e, v, m)[:, L] > 0).sum()),
                                           "deploy_below_0": int((score(d, v, m)[:, L] < 0).sum()), "of": k},
            "lopo_cosine_at_best_layer": {"min": round(float(lopo[:, L].min()), 4), "mean": round(float(lopo[:, L].mean()), 4),
                                          "pairs_below_0.9": [i + 1 for i in np.where(lopo[:, L] < 0.9)[0]]},
            "tau_youden": round(float(thr[j]), 4), "tau_in_half_gaps": round(float(thr[j] / half_gap), 4),
            "balanced_acc_at_tau": round(float((tpr[j] + 1 - fpr[j]) / 2), 4),
            "balanced_acc_at_0": round(float(((sl[y == 1] > 0).mean() + (sl[y == 0] <= 0).mean()) / 2), 4),
            "tau_5pct_fpr": round(tau5, 4), "tpr_at_5pct_fpr": round(float((sl[y == 1] > tau5).mean()), 4),
            "curves": {"val_auroc": curve.round(4).tolist(),
                       "random_p95": np.percentile(rand, 95, axis=1).round(4).tolist(),
                       "random_max": rand.max(1).round(4).tolist(),
                       "shuffle_p95": np.percentile(shuf, 95, axis=0).round(4).tolist(),
                       "adjacent_layer_cosine": (v[:-1] * v[1:]).sum(-1).round(4).tolist(),
                       "lopo_min_cosine": lopo.min(0).round(4).tolist()}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    plot(res)
    print(json.dumps({p: {k_: v_ for k_, v_ in res[p].items() if k_ != "curves"} for p in ("last", "mean")}, indent=1))
    print("chars AUROC", res["chars_auroc"])


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, pool in enumerate(("last", "mean")):
        c, x = res[pool]["curves"], np.arange(len(res[pool]["curves"]["val_auroc"]))
        a = ax[0, col]
        a.plot(x, c["val_auroc"], "k-", lw=2, label="probe")
        a.plot(x, c["random_p95"], "C0--", label="random direction, 95th pct")
        a.plot(x, c["random_max"], "C0:", label="random direction, max of 100")
        a.plot(x, c["shuffle_p95"], "C1--", label="shuffled pairs, 95th pct")
        a.axhline(0.5, color="grey", lw=0.8)
        a.axvline(res[pool]["best_layer"], color="grey", ls=":")
        a.set(title=f"{pool} token: val AUROC (eval vs WildChat)", ylabel="AUROC")
        b = ax[1, col]
        b.plot(x[1:], c["adjacent_layer_cosine"], "k-", label="cos(v_L, v_L-1)")
        b.plot(x, c["lopo_min_cosine"], "C2-", label="leave-one-pair-out, min cos")
        b.set(xlabel="layer (0 = embeddings)", ylabel="cosine")
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "auroc_vs_layer.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description="Llama 3 8B Instruct POC: direction from the 16 pairs, layer and threshold on val")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep", help="25 GSM8K + 25 TruthfulQA (eval) vs 50 WildChat (deployment)").add_argument("--phase1", required=True)
    sub.add_parser("extract", help="GPU: last-token and mean hidden states, every layer")
    sub.add_parser("analyze", help="AUROC per layer, controls, L*, threshold, plot")
    args = ap.parse_args()
    {"prep": prep, "extract": extract, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
