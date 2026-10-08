import collections
import glob
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0
DATA = Path(__file__).resolve().parent / "data"
LO, HI = 100, 300
PER_SOURCE = 100
SEED = 0

SPACE_PUNCT = re.compile(r"\s+([,.!?;:'])")
CONTRACTION = re.compile(r"\b(\w+)\s+n't\b")
SPEAKER = re.compile(r"^\s*[mf]\s*:\s*", re.IGNORECASE)


def snap(repo, pattern):
    return snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])


def clean(text, speaker=False):
    # Tokenisation artefacts only; a prompt's framing is the variable under study.
    t = str(text).replace("\n", " ")
    if speaker:
        t = SPEAKER.sub("", t)
    t = CONTRACTION.sub(r"\1n't", t)
    return re.sub(r"\s+", " ", SPACE_PUNCT.sub(r"\1", t)).strip()


def is_english(t):
    try:
        return detect(t) == "en"
    except Exception:
        return False


# ---- tier D: the format control, 25 per group, length-binned ----

D_REPO = "viliana-dev/eval-awareness-2x2"
D_GROUPS = {"bench_eval": ("bench_eval", "formatted_prompt", 1),
            "casual_eval": ("casual_eval_mutual", "first_turn", 1),
            "bench_deploy": ("bench_deploy_rewritten", "rewritten", 0),
            "casual_deploy": ("casual_deploy_matched", "first_turn", 0)}
D_BINS = [(100, 160), (160, 240), (240, 320)]


def tier_d():
    root = snap(D_REPO, "*/*.parquet")
    rng = random.Random(SEED)
    rows = []
    for group, (folder, column, label) in D_GROUPS.items():
        df = pd.read_parquet(sorted(glob.glob(f"{root}/{folder}/*.parquet"))[0])
        seen, pool = set(), []
        for v in df[column]:
            t = clean(v, speaker=True)
            if t.lower() in seen or not (D_BINS[0][0] <= len(t) <= D_BINS[-1][1]):
                continue
            if "<<" in t or ">>" in t or not is_english(t):
                continue
            seen.add(t.lower())
            pool.append(t)
        picked = []
        for lo, hi in D_BINS:
            band = [t for t in pool if lo <= len(t) < hi]
            rng.shuffle(band)
            picked += band[:9]
        rng.shuffle(picked)
        rows += [(t, label, group) for t in picked[:25]]
    rng.shuffle(rows)
    return [(clean(t), label, group) for t, label, group in rows]


# ---- tiers A to C: one reader per source, each returning raw strings ----

def parquets(repo, pattern, fn):
    root = snap(repo, pattern)
    return [x for f in sorted(glob.glob(f"{root}/{pattern}")) for x in fn(pd.read_parquet(f))]


def model_written_evals():
    root = snap("Anthropic/model-written-evals", "advanced-ai-risk/human_generated_evals/*.jsonl")
    out = []
    for f in sorted(glob.glob(f"{root}/advanced-ai-risk/human_generated_evals/*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(line)["question"] for line in fh if line.strip()]
    return out


def oasst2():
    # opening prompter turns only; a reply is a different kind of text
    root = snap("OpenAssistant/oasst2", "*all.messages.jsonl.gz")
    out = []
    for f in sorted(glob.glob(f"{root}/*all.messages.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("role") == "prompter" and not r.get("parent_id") and r.get("lang") == "en":
                    out.append(r["text"])
    return out


def first_user(messages):
    return next((m.get("content") for m in messages if m.get("role") == "user"), None)


SOURCES = [
    ("mask", "A", 1, lambda: parquets("cais/MASK", "*/*.parquet", lambda d: list(d["user_prompt"]))),
    ("model_written_evals", "A", 1, model_written_evals),
    ("bbq", "A", 1, lambda: parquets("walledai/BBQ", "data/*.parquet",
                                     lambda d: [f"{c} {q}" for c, q in zip(d["context"], d["question"])])),
    ("mmlu", "A", 1, lambda: parquets("cais/mmlu", "all/test-*.parquet", lambda d: list(d["question"]))),
    ("gsm8k", "A", 1, lambda: parquets("openai/gsm8k", "main/*.parquet", lambda d: list(d["question"]))),
    ("wildchat", "B", 0, lambda: parquets("allenai/WildChat-1M", "data/train-00000-of-00014.parquet",
                                          lambda d: [c[0]["content"] for c in d["conversation"] if len(c)])),
    ("lmsys", "B", 0, lambda: parquets("lmsys/lmsys-chat-1m", "data/train-00000-of-00006-*.parquet",
                                       lambda d: [c[0]["content"] for c in d["conversation"] if len(c)])),
    ("sharegpt", "B", 0, lambda: parquets(D_REPO, "casual_deploy_matched/*.parquet",
                                          lambda d: list(d["first_turn"]))),
    ("oasst2", "C", 0, oasst2),
    ("no_robots", "C", 0, lambda: parquets("HuggingFaceH4/no_robots", "data/*.parquet",
                                           lambda d: [first_user(m) for m in d["messages"]])),
]


def main():
    DATA.mkdir(exist_ok=True)
    d_rows = tier_d()
    taken = {t.lower() for t, _, _ in d_rows}       # one prompt, one place
    rng = random.Random(SEED)
    rows = []
    for name, tier, label, reader in SOURCES:
        raw = [r for r in reader() if r is not None]
        rng.shuffle(raw)
        picked = []
        for r in raw:
            t = clean(r)
            if (t.lower() in taken or not (LO <= len(t) < HI) or " " not in t or "<<" in t
                    or not is_english(t)):
                continue
            taken.add(t.lower())
            picked.append(t)
            if len(picked) == PER_SOURCE:
                break
        if len(picked) < PER_SOURCE:
            raise SystemExit(f"{name}: only {len(picked)} prompts pass the filters")
        rows += [{"text": t, "label": label, "source": name, "category": tier} for t in picked]
    rows += [{"text": t, "label": label, "source": group, "category": "D"} for t, label, group in d_rows]

    # Fixed order, then one seeded shuffle, so every chunk mixes all sources.
    rows.sort(key=lambda r: r["text"])
    random.Random(SEED).shuffle(rows)
    rows = [{"id": f"p{i:04d}", "prompt_id": i, **r} for i, r in enumerate(rows)]

    (DATA / "phase1.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in range(5):
        (DATA / f"phase1_c{c + 1}.json").write_text(
            json.dumps(rows[c * 220:(c + 1) * 220], ensure_ascii=False), encoding="utf-8")
    (DATA / "phase1_smoke.json").write_text(json.dumps(rows[:3], ensure_ascii=False), encoding="utf-8")
    print(len(rows), "prompts,", sum(r["label"] for r in rows), "tests | tiers",
          dict(sorted(collections.Counter(r["category"] for r in rows).items())))


if __name__ == "__main__":
    main()
