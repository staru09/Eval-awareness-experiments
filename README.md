# Is Linear Probing Enough for Evaluation Awareness?

Code to replicate the three studies of the paper on Qwen3-8B. The probe is the 16-pair
contrastive probe of Hua et al. (2025), `data/contrastive_pairs_16.txt`.

| Study | What it tests | Scripts |
|---|---|---|
| 1 | 1,100 length- and language-matched prompts: evaluations, real users, dataset prose | `prep.py`, `probe.py`, `plot.py` |
| 2 | The reference pipeline (Heidari et al. 2026) on its own data recipe: MASK vs WildChat | `reference/` |
| 3 | Global-MMLU-Lite and seven safety and deception benchmarks vs WildChat | `reference/` |

Prompt files are rebuilt from the original datasets and are not included, because WildChat,
LMSYS-Chat-1M and ShareGPT cannot be redistributed.

## Setup

We used one H100 GPU and a Hugging Face token (`hf auth login`). Accept the terms of the gated
datasets on Hugging Face first: `lmsys/lmsys-chat-1m` (Study 1) and `allenai/wildjailbreak`
(Study 3).

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
# Studies 2 and 3 also need a second environment: the reference repo loads
# GSM8K and TruthfulQA by old dataset names that only datasets 3.6 accepts.
uv venv .venv_old
uv pip install --python .venv_old/bin/python "datasets==3.6.0" "huggingface_hub<1.0" pandas pyarrow
```

## Study 1

```bash
python prep.py                                   # data/phase1.json, 1,100 prompts
python probe.py run --model Qwen/Qwen3-8B --out runs/qwen3 \
    --repo <your-hf-dataset-repo> --dest qwen3   # probe, generation, hidden states, scores
python probe.py analyze --run-dir runs/qwen3     # AUROC against matched random directions
python plot.py qwen3_8b=runs/qwen3 --out figures # Figures 2 to 4, Tables 2 and 3
```

`probe.py run` generates the model's reasoning for every prompt with vLLM (up to 32,768
tokens) and uploads each chunk to the Hugging Face repo you name, so it takes several hours.

## Studies 2 and 3

```bash
PY=.venv/bin/python PY_OLD=.venv_old/bin/python bash reference/run_qwen.sh work
python reference/analyze.py work/results figures
```

`run_qwen.sh` does the following:
1. Clones the reference repository at commit `8ad6a8e`.
2. Builds the prompts:
   - `reference/wildchat_pool.py`: a stand-in for the reference's unreleased WildChat pool;
   - the reference's own `build_prompts.py` (seed 0): calibration and MASK test sets;
   - `reference/build_benchmarks.py`: Global-MMLU-Lite and the seven safety sets, with English and 80 to 400 character filters.
3. Builds the probe with 100 random directions and scores every prompt with the reference code.
4. Runs the reference `classify_auroc.py` on every comparison.

It takes about 30 minutes, most of it downloading the model and datasets. `analyze.py` then
compares the probe with 100 random directions given the same choice of layer and sign, and
writes Tables 4 to 6 and Figures 5 and 6.

## Expected results

| Comparison | Probe | Random directions as good |
|---|---|---|
| Study 1: evaluations vs users, best 3-layer band | 0.619 | best of 100 reaches 0.668 (p = 0.35) |
| Study 2: MASK vs WildChat, best test layer (15) | 0.690 | 90 of 100 |
| Study 3: DeceptionBench, mean over tokens, layer 6 | 0.126 | 0 of 100 (p = 0.01) |

**How to read AUROC.** In the reference code AUROC above 0.5 means the benchmark scores more
like real users, the opposite of what an evaluation detector should do.

**Small differences are expected.** Your numbers can differ in the third decimal place:
- dataset revisions on Hugging Face can change;
- the WildChat pool is our stand-in, not the reference's;
- GPU arithmetic is not bit-exact across machines and batch sizes.
