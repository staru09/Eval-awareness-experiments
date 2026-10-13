"""Stand-in for the reference paper's unreleased 1,000-prompt WildChat pool.

First user turn, labelled English, assistant reply not opening with a refusal, duplicates
removed, from the first WildChat-1M file; 1,000 sampled with seed 0.

    python reference/wildchat_pool.py data/wildchat_pool.jsonl
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REFUSAL = re.compile(r"^\W*(I'?m sorry|I am sorry|I apologi[sz]e|I can(no|')t|I'm not able|I am not able|As an AI)", re.I)

root = snapshot_download("allenai/WildChat-1M", repo_type="dataset", allow_patterns=["data/train-00000-*"])
df = pd.read_parquet(next(Path(root, "data").glob("train-00000-*")))
pool, seen = [], set()
for conv, lang in zip(df["conversation"], df["language"]):
    if lang != "English" or len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
        continue
    q, a = conv[0]["content"].strip(), conv[1]["content"].strip()
    if q and q not in seen and not REFUSAL.search(a):
        seen.add(q)
        pool.append(q)
print(len(df), "conversations,", len(pool), "pass the filters")
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.writelines(json.dumps({"question": q}, ensure_ascii=False) + "\n" for q in random.Random(0).sample(pool, 1000))
