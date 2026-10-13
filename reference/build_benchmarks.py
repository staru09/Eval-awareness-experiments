"""Study 3 prompt files.

  data/mmlu/{with_options,question_only}.jsonl  100 Global-MMLU-Lite questions (English test split, seed 0)
  data/safety/<set>.jsonl                       up to 100 prompts per safety/deception benchmark (label 1)
                                                and 100 held-out WildChat prompts (label 0)

Safety filters: duplicates removed; English for the whole prompt and for every phrase of 20+
characters, no letters outside the Latin script; 80 to 400 characters; seed 0.

    python reference/build_benchmarks.py data
"""
import json
import random
import re
import statistics
import sys
from pathlib import Path

from datasets import load_dataset
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0
DATA = Path(sys.argv[1])
LO, HI, N, MIN_N = 80, 400, 100, 50


def write(path, texts, label, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"text": t, "label": label, "source": source, "prompt_id": i}, ensure_ascii=False) + "\n")


def mc(q, choices):
    return q + "\n" + "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices))


def english(t):
    try:
        return detect(t) == "en"
    except Exception:
        return False


def strictly_english(t):
    if any(ch.isalpha() and ord(ch) > 0x24F for ch in t):
        return False
    return english(t) and all(english(c) for c in re.split(r"[.,;:!?\n]+", t) if len(c.strip()) >= 20)


def sycophancy():
    """The repo's files have different columns, so `load_dataset` fails; read them one by one."""
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "meg-tong/sycophancy-eval"
    rows = [json.loads(l) for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".jsonl")
            for l in open(hf_hub_download(repo, f, repo_type="dataset"), encoding="utf-8")]
    return [r["prompt"][0]["content"] for r in rows if len(r["prompt"]) == 1 and r["prompt"][0]["type"] == "human"]


def wildchat():
    """Held-out real users: the stand-in pool minus the 200 calibration prompts."""
    cal = {json.loads(l)["text"] for l in open(DATA / "paper/train.jsonl", encoding="utf-8")}
    pool = (json.loads(l)["question"].strip() for l in open(DATA / "wildchat_pool.jsonl", encoding="utf-8"))
    return [q for q in pool if q not in cal]


SAFETY = {
    "deceptionbench": lambda: load_dataset("PKU-Alignment/DeceptionBench", split="test")["outer_prompt"],
    "sycophancy_eval": sycophancy,
    "model_written_evals": lambda: load_dataset("Anthropic/model-written-evals", split="train")["question"],
    "halueval": lambda: load_dataset("pminervini/HaluEval", "qa", split="data")["question"],
    "wmdp": lambda: [mc(r["question"], r["choices"]) for c in ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
                     for r in load_dataset("cais/wmdp", c, split="test")],
    "wildjailbreak": lambda: [r["adversarial"] for r in load_dataset("allenai/wildjailbreak", "eval", split="train",
                                                                      delimiter="\t", keep_default_na=False)
                              if r["data_type"] == "adversarial_harmful"],
    "do_not_answer": lambda: load_dataset("LibrAI/do-not-answer", split="train")["question"],
    "wildchat": wildchat,
}

# Global-MMLU-Lite: the same 100 of the 400 English test questions, with and without options
d = load_dataset("CohereLabs/Global-MMLU-Lite", "en", split="test")
idx = sorted(random.Random(0).sample(range(len(d)), 100))
rows = [d[i] for i in idx]
write(DATA / "mmlu/with_options.jsonl",
      [f"{r['question']}\n" + "\n".join(f"{k}. {r['option_' + k.lower()]}" for k in "ABCD") for r in rows], 1, "gmmlu")
write(DATA / "mmlu/question_only.jsonl", [r["question"] for r in rows], 1, "gmmlu")

for name, load in SAFETY.items():
    raw = list(dict.fromkeys(str(t).strip() for t in load() if t and str(t).strip()))
    pool = [t for t in raw if LO <= len(t) <= HI and strictly_english(t)]
    if len(pool) < MIN_N:
        print(f"SKIP {name}: only {len(pool)} prompts pass the filters")
        continue
    texts = random.Random(0).sample(pool, min(N, len(pool)))
    write(DATA / f"safety/{name}.jsonl", texts, int(name != "wildchat"), name)
    n = sorted(len(t) for t in texts)
    print(f"{name:20s} {len(texts):3d} of {len(pool):5d} | chars min/median/max {n[0]}/{int(statistics.median(n))}/{n[-1]}")
