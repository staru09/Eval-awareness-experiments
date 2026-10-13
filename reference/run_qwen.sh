#!/usr/bin/env bash
# Studies 2 and 3 on Qwen3-8B with the reference pipeline (cls_corr_mi, commit 8ad6a8e), unchanged.
#
#   PY=<python with torch> PY_OLD=<python with datasets==3.6.0> bash reference/run_qwen.sh WORKDIR
#
# Writes prompts to WORKDIR/data, scores and AUROC files to WORKDIR/results.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mkdir -p "${1:-work}" && cd "${1:-work}" && pwd)
PY=${PY:-python}; PY_OLD=${PY_OLD:-python}
D=$W/data; R=$W/results; mkdir -p "$D" "$R"
MODEL=(--model-key qwen3_8b --model-path Qwen/Qwen3-8B)

[ -d "$W/evaluation-awareness" ] || git clone -q https://github.com/evaluation-awareness/evaluation-awareness "$W/evaluation-awareness"
git -C "$W/evaluation-awareness" checkout -q 8ad6a8e
REF=$W/evaluation-awareness/cls_corr_mi

# 1. Prompts. The reference build_prompts.py loads datasets by old names, which need datasets 3.6.
[ -f "$D/wildchat_pool.jsonl" ] || $PY "$HERE/wildchat_pool.py" "$D/wildchat_pool.jsonl"
[ -f "$D/paper/train.jsonl" ] || (cd "$REF" && $PY_OLD build_prompts.py --wildchat-jsonl "$D/wildchat_pool.jsonl" \
  --mask-root "$D/no_local_mask" --out-dir "$D/paper" --seed 0)
[ -f "$D/safety/wildchat.jsonl" ] || $PY "$HERE/build_benchmarks.py" "$D"

# 2. Probe with 100 random directions, then scores.
score() { (cd "$REF" && $PY score_prompts.py "${MODEL[@]}" --probe "$R/probe.pt" --prompts-jsonl "$1" --out "$2" --batch-size "$3"); }
[ -f "$R/probe.pt" ] || (cd "$REF" && $PY compute_probe.py "${MODEL[@]}" --out "$R/probe.pt" --n-random 100 --batch-size 4)
score "$D/paper/train.jsonl" "$R/train.npz" 8
score "$D/paper/test.jsonl" "$R/test.npz" 4
score "$D/paper/mask_1k.jsonl" "$R/mask_1k.npz" 4
$PY "$HERE/combine.py" "$R/test.npz" "$R/mask_1k.npz" "$R/mask1k_vs_wild.npz"
for f in with_options question_only; do
  score "$D/mmlu/$f.jsonl" "$R/mmlu_$f.npz" 4
  $PY "$HERE/combine.py" "$R/test.npz" "$R/mmlu_$f.npz" "$R/mmlu_${f}_vs_wild.npz"
done
for f in "$D"/safety/*.jsonl; do
  score "$f" "$R/safety_$(basename "$f" .jsonl).npz" 64
done
for f in "$D"/safety/*.jsonl; do
  n=$(basename "$f" .jsonl); [ "$n" = wildchat ] && continue
  $PY "$HERE/combine.py" "$R/safety_wildchat.npz" "$R/safety_$n.npz" "$R/safety_${n}_vs_wild.npz"
done

# 3. The reference's own AUROC report (best test layer, random max at that layer) for every comparison.
for t in "$R/test.npz" "$R"/*_vs_wild.npz; do
  for p in last mean; do
    (cd "$REF" && $PY classify_auroc.py --train-scores "$R/train.npz" --test-scores "$t" --pooling $p \
      --out "$R/auroc_$(basename "$t" .npz)_$p.json")
  done
done
echo "done: $R"
