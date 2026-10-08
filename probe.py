import argparse
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WIDTH, N_PERM, N_BOOT, SEED = 3, 2000, 1000, 0


# ---------------------------------------------------------------- GPU run

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def summary(path):
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    toks = sorted(e["n_generated_tokens"] for e in t)
    print(f"saved {len(t)} | hit the token limit {sum(e.get('finish_reason') == 'length' for e in t)}"
          f" | no reasoning {sum(not e['reasoning_trace'] for e in t)}"
          f" | gen tokens min/median/max {toks[0]}/{toks[len(toks) // 2]}/{toks[-1]}", flush=True)


def prompt_ids(model, text):
    from evalaware.models import ModelSpec, load_tokenizer, prompt_only
    tok = load_tokenizer(model)
    prompt = prompt_only(tok, text, ModelSpec(hf_id=model, enable_thinking=True))
    return tok(prompt, add_special_tokens=False)["input_ids"]


def capture(args):
    """One prompt through HF or vLLM, saved as [layers, tokens, d] for the agreement check."""
    import torch
    text = json.loads((DATA / "phase1.json").read_text(encoding="utf-8"))[args.prompt_id]["text"]
    ids = prompt_ids(args.model, text)
    if args.backend == "hf":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
        with torch.no_grad():
            hidden = m(torch.tensor([ids], device=m.device), output_hidden_states=True).hidden_states
        h = torch.stack([x[0] for x in hidden[1:]]).float().cpu().numpy()
    else:
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states, load_hidden_states)
        from vllm.inputs import TokensPrompt
        from evalaware.activations import _final_norm
        n_layers = AutoConfig.from_pretrained(args.model).get_text_config().num_hidden_layers
        tmp = Path(args.out).parent / "_vllm_hidden"
        tmp.mkdir(parents=True, exist_ok=True)
        n = len(ids) + 8
        llm = LLM(model=args.model, dtype="bfloat16", max_model_len=n, max_num_batched_tokens=n,
                  enable_chunked_prefill=False, enable_prefix_caching=False, gpu_memory_utilization=0.5,
                  speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                      "draft_model_config": {"hf_config": {
                                          "eagle_aux_hidden_state_layer_ids": list(range(1, n_layers + 1))}}},
                  kv_transfer_config=KVTransferConfig(
                      kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer",
                      kv_connector_extra_config={"shared_storage_path": str(tmp)}))
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1))[0]
        path = out.kv_transfer_params["hidden_states_path"]
        h = load_hidden_states(path)["hidden_states"].float()
        cleanup_hidden_states(path)
        h[:, -1] = _final_norm(args.model)(h[:, -1])         # vLLM saves the last layer before the norm
        h = h.permute(1, 0, 2).numpy()
    np.save(args.out, h)


def build(args):
    """evalaware.probe.build, as `evalaware probe` calls it, but with the pairs read from this repo:
    the installed evalaware package does not ship its data/ folder."""
    from evalaware import datasets, probe as contrastive
    from evalaware.models import load_model
    model, tok = load_model(args.model, "bfloat16")
    contrastive.build(model, tok, datasets.load_pairs(DATA / "contrastive_pairs_16.txt"), Path(args.out),
                      args.model, n_random=100, seed=42)


def check_agreement(out, probe_path):
    import torch
    a, b = np.load(out / "hf.npy"), np.load(out / "vllm.npy")
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    real = torch.load(probe_path, map_location="cpu", weights_only=False)["real"]
    gap = max(abs(float(a[l].mean(0) @ real[l + 1].numpy()) - float(b[l].mean(0) @ real[l + 1].numpy()))
              for l in range(a.shape[0]))
    print(f"HF vs vLLM: lowest cosine {cos.min():.5f} at layer {int(cos.min(1).argmin())}, "
          f"largest whole-prompt probe score gap {gap:.3f}", flush=True)
    return float(cos.min())


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # nvcc on the VM is too old to build it
    gen = (f"evalaware generate --model-path {args.model} --backend vllm --capture-backend vllm "
           f"--max-new-tokens 32768 {args.sampling}")
    up = f"hf upload {args.repo} {{src}} {args.dest}/{{dst}} --repo-type dataset"

    sh(f"{sys.executable} {__file__} build --model {args.model} --out {out}/probe.pt")
    for backend in ("hf", "vllm"):                      # separate processes, separate GPU memory
        sh(f"{sys.executable} {__file__} capture --backend {backend} --model {args.model} "
           f"--prompt-id {args.check_prompt_id} --out {out}/{backend}.npy")
    if check_agreement(out, out / "probe.pt") < 0.99:
        raise SystemExit("HF and vLLM hidden states differ; not running")

    sh(f"{gen} --dataset {DATA}/phase1_smoke.json --out-dir {out}/smoke --gpu-mem 0.65")
    summary(out / "smoke/reasoning_traces.json")
    sh(up.format(src=f"{out}/probe.pt", dst="probe.pt"))

    # Two passes per chunk: generation wants most of the GPU for its cache; capture
    # needs ~10 GB free per long sequence. Models with full multi-head attention
    # (Olmo-3) cannot hold a 32k-token sequence in vLLM's capture cache: use --capture hf.
    cap = "--gpu-mem 0.65" if args.capture == "vllm" else "--capture-backend hf"
    for c in range(1, 6):
        d = f"{out}/c{c}"
        sh(f"{gen} --dataset {DATA}/phase1_c{c}.json --out-dir {d} --no-activations --gpu-mem 0.88")
        summary(f"{d}/reasoning_traces.json")
        sh(f"{gen} --dataset {DATA}/phase1_c{c}.json --out-dir {d} {cap}")
        shutil.make_archive(d, "zip", out, f"c{c}")
        sh(up.format(src=f"{d}.zip", dst=f"c{c}.zip"))      # nothing stays only on the VM

    traces = []
    for kind in ("activations_prompt", "activations"):
        (out / "all" / kind).mkdir(parents=True, exist_ok=True)
        for c in range(1, 6):
            for f in (out / f"c{c}" / kind).glob("*.npz"):
                dst = out / "all" / kind / f.name
                if not dst.exists():
                    dst.symlink_to(f)
    for c in range(1, 6):
        traces += json.loads((out / f"c{c}/reasoning_traces.json").read_text(encoding="utf-8"))
    traces.sort(key=lambda e: e["prompt_id"])
    (out / "reasoning_traces_all.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    for kind, name in (("activations_prompt", "scores_prompt.npz"), ("activations", "scores_reply.npz")):
        sh(f"evalaware score --probe {out}/probe.pt --activations-dir {out}/all/{kind} "
           f"--dataset {DATA}/phase1.json --out {out}/{name}")
    for f in ("scores_prompt.npz", "scores_reply.npz", "reasoning_traces_all.json"):
        sh(up.format(src=f"{out}/{f}", dst=f))


# ---------------------------------------------------------------- analysis

def ranks(x):
    return rankdata(x, axis=0)


def auroc_from_ranks(r, y):
    """r: [n, ...] ranks; y: [n] or [k, n] 0/1. AUROC per trailing column, label 1 expected higher."""
    y = np.atleast_2d(y).astype(float)
    n1 = y.sum(1)
    n0 = y.shape[1] - n1
    rs = np.tensordot(y, r, axes=(1, 0))
    shape = (-1,) + (1,) * (rs.ndim - 1)
    return (rs - (n1 * (n1 + 1) / 2).reshape(shape)) / (n1 * n0).reshape(shape)


def bands(curve, width=WIDTH):
    k = np.ones(width) / width
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), -1, curve)


def text_features(rows):
    t = [r["text"] for r in rows]
    return {"chars": np.array([len(x) for x in t]),
            "words": np.array([len(x.split()) for x in t]),
            "punctuation": np.array([sum(ch in string.punctuation for ch in x) for x in t]),
            "digits": np.array([sum(ch.isdigit() for ch in x) for x in t])}


def compare(real, rand, y, feats, rng, prechosen):
    """Best 3-layer band, and every control given the same layer search."""
    n = real.shape[0]
    r_real = ranks(real)
    curve = auroc_from_ranks(r_real, y)[0]
    b = bands(curve)
    start = int(b.argmax())
    best, layers = float(b[start]), list(range(start, start + WIDTH))
    rand_best = bands(auroc_from_ranks(ranks(rand), y)[0].T).max(1)
    perm_best = bands(auroc_from_ranks(r_real, np.stack([rng.permutation(y) for _ in range(N_PERM)]))).max(1)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            boot.append(auroc_from_ranks(ranks(real[i][:, layers]), y[i])[0].mean())
    pos, neg = real[y == 1, layers[1]], real[y == 0, layers[1]]
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return {
        "n_expected_higher": int(y.sum()), "n_other": int(n - y.sum()),
        "headline_band_auroc": round(best, 4),
        "band_layers": layers, "band_layers_paper_numbering": [l + 1 for l in layers],
        "band_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "single_best_layer": int(curve.argmax()), "single_best_auroc": round(float(curve.max()), 4),
        "lowest_layer_auroc": round(float(curve.min()), 4),
        "prechosen": {str(l): round(float(curve[l]), 4) for l in prechosen},
        "random_floor_best_of_100": round(float(rand_best.max()), 4),
        "random_p": round(float(((rand_best >= best).sum() + 1) / (len(rand_best) + 1)), 4),
        "shuffle_p95": round(float(np.percentile(perm_best, 95)), 4),
        "shuffle_p": round(float(((perm_best >= best).sum() + 1) / (N_PERM + 1)), 4),
        "text_baselines": {k: round(float(auroc_from_ranks(ranks(v.astype(float)), y)[0]), 4)
                           for k, v in feats.items()},
        "cohens_d_at_band_middle": round(float((pos.mean() - neg.mean()) / pooled), 3),
        "catch_rate_at_5pct_false_alarms": round(float((pos > np.quantile(neg, 0.95)).mean()), 3),
        "curve": [round(float(c), 4) for c in curve],
    }


def verdict(c):
    if c["headline_band_auroc"] <= 0.5:
        return "no separation in the expected direction"
    floors = (c["random_floor_best_of_100"], c["shuffle_p95"], max(c["text_baselines"].values()))
    return "beats all three controls" if c["headline_band_auroc"] > max(floors) else "loses to a control"


def analyze(args):
    rows = json.loads((DATA / "phase1.json").read_text(encoding="utf-8"))
    tier = np.array([r["category"] for r in rows])
    source = np.array([r["source"] for r in rows])
    prechosen = tuple(int(x) for x in args.prechosen.split(","))
    rng = np.random.default_rng(SEED)
    # each comparison: (prompts kept, which of them is expected to score higher)
    spec = {"RQ1 tier A vs tier B": ((tier == "A") | (tier == "B"), tier == "A"),
            "RQ2 tier C vs tier B": ((tier == "B") | (tier == "C"), tier == "C"),
            "RQ2 support tier A vs tier C": ((tier == "A") | (tier == "C"), tier == "A")}
    for s in ("mask", "model_written_evals", "bbq", "mmlu", "gsm8k"):
        spec[f"RQ3 {s} vs tier B"] = ((source == s) | (tier == "B"), source == s)
    spec["replication mask vs wildchat"] = ((source == "mask") | (source == "wildchat"), source == "mask")
    spec["tier D wording only (bench_deploy vs casual_deploy, 25 each)"] = (
        (source == "bench_deploy") | (source == "casual_deploy"), source == "bench_deploy")

    out = {}
    for readout, name in (("prompt", "scores_prompt.npz"), ("prompt+reply", "scores_reply.npz")):
        z = np.load(Path(args.run_dir) / name, allow_pickle=True)
        assert (z["prompt_ids"] == np.arange(len(rows))).all()
        for pooling in ("mean", "last"):
            real, rand = z[f"scores_{pooling}_real"], z[f"scores_{pooling}_random"]
            res = {}
            for comp, (keep, hi) in spec.items():
                c = compare(real[keep], rand[keep], hi[keep].astype(int),
                            text_features([rows[i] for i in np.where(keep)[0]]), rng, prechosen)
                c["verdict"] = verdict(c)
                res[comp] = c
            out[f"{readout} / {pooling}"] = res
    dest = Path(args.run_dir) / "analysis_phase1.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, res in out.items():
        print(f"\n=== {key}")
        for comp, c in res.items():
            print(f"{comp:62s} {c['headline_band_auroc']:.3f} ({c['band_95ci'][0]:.3f}-{c['band_95ci'][1]:.3f})"
                  f"  random {c['random_floor_best_of_100']:.3f}  {c['verdict']}")
    print("\nwrote", dest)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU pipeline: probe, agreement check, generate, capture, score, push")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repo", required=True, help="Hub dataset repo to push to")
    r.add_argument("--dest", required=True, help="folder inside the repo")
    r.add_argument("--capture", choices=["vllm", "hf"], default="vllm")
    r.add_argument("--sampling", default="--temperature 0.6 --top-p 0.95")
    r.add_argument("--check-prompt-id", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--backend", choices=["hf", "vllm"], required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--prompt-id", type=int, required=True)
    c.add_argument("--out", required=True)
    a = sub.add_parser("analyze", help="AUROC against layer-matched controls")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--prechosen", default="10,14", help="decoder layers fixed in advance")
    args = ap.parse_args()
    {"run": run, "build": build, "capture": capture, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
