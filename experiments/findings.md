# Findings: does the 16-pair probe separate evaluation from deployment?

Models: Meta-Llama-3-8B-Instruct and Qwen3-8B (thinking mode on).
Probe: the 16 contrastive pairs from Hua et al. (2025), one direction per layer.
Reference: Heidari, Memarian and Rabusseau, "Evaluation Awareness in Language Models"
(arXiv 2608.21766), code at github.com/evaluation-awareness/evaluation-awareness (`cls_corr_mi/`, commit 8ad6a8e).

## How to read the numbers

The paper's direction is "real user minus evaluation", and its code counts the evaluation
prompts as the side that should score higher. So in the paper's numbers, AUROC above 0.5
means the evaluation prompts look **more like real users**. Our own script (`llama3_poc.py`)
uses the opposite sign, so there AUROC below 0.5 means the same thing.

## What we ran

1. **Our check (`llama3_poc.py`), Llama only.** 25 GSM8K + 25 TruthfulQA vs 50 WildChat, all
   100 to 300 characters. 100 random directions and 100 shuffled-pair directions as controls.
2. **Their pipeline, unchanged, on our 100 prompts.** Only change: a Llama entry in `models.yaml`.
3. **Their pipeline on the paper's data recipe.** Built with their `build_prompts.py` (seed 0):
   200 WildChat + 100 GSM8K + 100 TruthfulQA for calibration, 100 MASK + 100 WildChat for test,
   and all 1,000 MASK prompts. Run for Llama and Qwen with 100 random directions.

## Results

**Our check (Llama).** The evaluation prompts score on the real-user side at almost every layer.
- Last token: AUROC is below 0.5 at all 32 layers, lowest 0.049 at layer 10.
  GSM8K alone gives 0.05 and TruthfulQA alone 0.04.
- The best layer in the intended direction is 0.484 (layer 7).
  95% of random directions at that layer score below 0.871.
- Encoding the 16 phrases without the chat template, as the paper does, changes single layers
  but not the picture: 29 of 32 layers stay below 0.5.

**Their pipeline, paper data, last token** (above 0.5 means MASK looks more like a real user):

| | Llama 3 8B | Qwen3-8B |
|---|---|---|
| Best layer picked on test (their method) | 0.714 (layer 30) | 0.690 (layer 15) |
| Best random direction at that same layer | 0.817 | 0.791 |
| Random directions that beat the probe when each gets its own best layer | 78 of 100 | 90 of 100 |
| Layer picked on the calibration set instead | 0.533 | 0.536 |
| Layers where the probe beats every random direction | 0 of 32 | 0 of 36 |

Results on all 1,000 MASK prompts are the same: 0.735 Llama, 0.668 Qwen.
The one probe-specific signal is Llama's average over tokens at layer 30. It scores 0.82 to 0.83
and beats 99 to 100 of 100 random directions, but it points the wrong way (MASK looks like a
real user). It also disappears at the layer picked on calibration (0.49 to 0.52).

## Compared with the paper

- **The raw number matches.** The paper reports above 0.7 for Qwen3-8B; we get 0.690 from one seed.
- **The conclusion does not.** "Separates evaluation from deployment" holds only if three
  choices are all allowed:
  1. The best layer is picked on the test set itself.
  2. Either direction counts as separation. The paper says lower scores mean evaluation, yet its
     reported AUROC above 0.7 means MASK scores higher, that is, more deployment-like.
  3. The probe is compared with the **average** random direction (always about 0.5). Single
     random directions at the same layer range from about 0.23 to 0.82.

**Conclusion so far.** MASK and WildChat can be told apart in the activations: even random
directions reach about 0.8. But the direction built from the 16 phrases does no better than a
random direction once both get the same layer search, and it points the wrong way. We find no
evidence that the probe reads "being evaluated" in Llama 3 8B or Qwen3-8B.

## What is missing and why

| Missing | Why |
|---|---|
| The paper's exact WildChat prompts | Their 1,000-prompt pool is not released. We used a stand-in: the first WildChat-1M file, filtered as the paper describes (English, first user turn, no system message, no refusal), with 1,000 sampled using seed 0. GSM8K, TruthfulQA and MASK follow their script, but dataset versions are not stated, so an exact match cannot be confirmed. |
| 10 seeds | We ran seed 0 only. The paper averages 10 test resamples. |
| Verbalization (does the model say it is being tested?) | Their repo has no code to generate answers or run the judge. The paper's rubric is only in Appendix C.1. Generation (3 answers per MASK prompt, temperature 1.0) and judging are deferred. |
| Probe vs judge correlation | `correlation_mi.py` is in their repo and ready, but it needs judge scores. MASK probe scores (`mask_1k.npz`) are already computed. |
| Steering | Not started. It needs generation and the judge first. |
| A classifier trained on the activations | Not run. It would show how easy MASK vs WildChat is to separate with any line, as a reference for the probe. |
| Length control on the paper data | The paper's sets are not length-matched. Median length in characters: TruthfulQA 47, GSM8K 233, WildChat 234 to 291 (up to about 28,000), MASK 292. Our own 100-prompt set is length-matched. |
| Other paper models | Qwen3-32B, Olmo3, Gemma 4 and Nemotron not run. Llama 3 8B is not in the paper. |
| Qwen's average-over-tokens curve | It is almost flat, at 0.51 to 0.53, across layers 7 to 30. A likely cause is one large-activation token (`<|im_start|>`) dominating the average. Not yet verified. |

## Pipeline notes

- **Old dataset names.** `build_prompts.py` loads `gsm8k` and `truthful_qa` by their old names, which the current Hugging Face library rejects. We ran it with `datasets==3.6.0`.
- **Double start token.** `score_prompts.py` tokenizes text that already went through the chat template with the default settings. For Llama this adds a second `<|begin_of_text|>`. The 16 phrases are encoded without the template, with one start token.
- **Layer and cutoff choice.** Their `classify_auroc.py` picks the Youden cutoff on the calibration set but the peak layer on the test set.
