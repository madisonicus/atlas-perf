# Abliterating open models ourselves on atlas

Research note, 2026-11-27. Web research only; nothing has been run on atlas yet.

## Bottom line

Nothing stops us. Both base models are Apache-2.0 and **not gated** on Hugging Face (checked via HF API 2026-11-27: `google/gemma-4-31B-it`, `Qwen/Qwen3.8-27B`, `gated: false`). The tooling is mature and an MLX-native port has been measured on an M5 Ultra at about a minute per trial for a 27B model, so a full run is a few hours, not days.

**Recommendation:** pilot on Qwen3.8-27B with Heretic-MLX (pinned commit, code read first), daytime, outside the 22:00–05:00 night batch window. Target the same uniform int4 layout we already serve, keep vision and MTP tensors, and judge the result with a refusal judge plus capability checks, not Heretic's keyword count. If it holds up, do Gemma 4 31B next, which removes the gated third-party Heretic dependency.

(Update 2026-11-27: Ben approved; pilot starts on Qwen3.8-9B, same hybrid architecture, then 27B.)

## State of the art (checked live)

- **Heretic** (p-e-w/heretic, AGPL-3.0) is the standard tool. Latest release v2.1.0 (2026-11-08). v2.0 (2026-11-01) made **Arbitrary-Rank Ablation (ARA)** the default modifier; classic directional ablation with projected abliteration and norm preservation (MPOA, from Jim Lai's work) remains available. It runs an Optuna TPE search that co-minimizes refusals and KL divergence from the original. v2 configs are not compatible with v1.
- **Norm-preserving biprojected abliteration** (Lai, Nov 2025) changes only the direction of weight rows, not their magnitude, and keeps more reasoning ability than plain abliteration. Built into Heretic as MPOA.
- **Independent evaluation** (Young, arXiv 2611.17846, 2026-11-13) ran Heretic 2.0 on exactly our families (Qwen3.8-27B, Gemma-4-31B and others). Findings that matter for us:
  - Newer models are not uniformly harder to abliterate; it depends on model and method. ARA beat direction-based methods on Gemma, and did relatively worse on Qwen.
  - Keyword refusal counts disagree with a judge (WildGuard) by about 7 points per model.
  - **First-token KL does not bound capability loss.** Low KL alone does not prove the model is unharmed.

## Running on atlas

- **Upstream Heretic is PyTorch.** On Apple Silicon it uses MPS, which is slow, and for Qwen3.8's hybrid GatedDeltaNet layers MPS has no kernels, so most layers fall back to CPU (one report: ~1.5 h per trial on a smaller Qwen hybrid). Gemma 4 31B (dense) works on MPS but needs fixes (MPS fallback for `linalg_qr`, a NaN fix for masked logits in the KL scorer) and takes several hours for 200 trials, per practitioner reports.
- **Heretic-MLX** (scoltzero/heretic-mlx, AGPL-3.0, created 2026-11-23) ports Heretic 2.2's ARA and MPOA to MLX. Author's own M5 Ultra 256 GB numbers on Qwen3.8-27B: 49–81 s per full ARA trial; works directly on MXFP4, MXFP8, MLX affine 4/5/6/8-bit and bf16 without a second copy of the weights; preserves all `vision_tower.*` and `language_model.mtp.*` tensors; exports in the same quant layout (`preserve`), a ~164 MB LoRA adapter, or higher-precision merged weights; resumable; writes a `reproduce.json`. Caveat: it is days old, one author, unreviewed. Read the code and pin a commit before running it. (Our review: [reviews/2026-11-27-heretic-mlx-code-review.md](../reviews/2026-11-27-heretic-mlx-code-review.md).)
- Other MLX options: nateGeorge/mlx-abliterate (MIT, grid search, classic directional ablation only) and akaiHuang/mlx-heretic (older Heretic algorithm). Heretic-MLX is the most complete.
- **Time estimate:** 100 trials × ~1 min ≈ 2 h per model on M5 Ultra, plus evaluation. Unverified on atlas; a `--smoke` run (3 trials, ~5 min per the author) confirms speed first.
- **Memory:** 27B at 4-bit is ~15 GB, bf16 ~55 GB; the search keeps one copy. It runs in its own MLX process, not through Ollama, so the priority tiers do not apply: it competes with Ollama for GPU and memory while it runs. Schedule it when atlas is otherwise quiet and never in the night batch window.

## What doing it ourselves gains

1. **No gated or third-party dependency.** The Gemma 4 31B Heretic we wanted is gated with manual approval; the base model is not.
2. **Our quant recipe from the start.** Abliterate, then export directly in the uniform int4 layout atlas serves (or abliterate bf16 and quantize with our recipe). No re-quantizing someone else's weights.
3. **Keep vision.** The third-party Qwen3.8 Heretic we run lost vision; Heretic-MLX leaves vision tensors untouched.
4. **Our own objectives.** Heretic's scorer plugins let us add a judge model and a capability benchmark to the search, and other configs exist (anti-slop, anti-sycophancy) if wanted.
5. **Reproducible and publishable.** `reproduce.json` plus our eval results are a natural lab-repo post.

## Draft heads (MTP / DFlash)

- ARA and abliteration only touch attention output and MLP down projections; the MTP head tensors and separate drafter models are left as they were. They were trained on the original model's hidden states and outputs, so acceptance drops somewhat after the edit. How much is unmeasured anywhere I could find.
- Gemma 4's drafter is a separate ~0.4B "assistant" model, not baked into the 31B weights. Qwen3.8's MTP tensors live in the checkpoint and are preserved.
- Because the edit is small (target KL ≈ 0.05–0.15), the first step is to **measure acceptance** on the abliterated model with the existing drafter. Only if it drops meaningfully is retraining worth it: DFlash-style drafters have been adapted to new targets with ~10k updates on the target's own outputs, and on-policy distillation (Draft-OPD, AdaFlash) beats plain SFT. That is a training job, a separate project, and should wait for numbers.

## Evaluation plan (pilot)

1. Baseline the stock model: refusal rate on held-out harmful prompts judged by a classifier (WildGuard or Heretic's BERT classifier scorer), plus a quick capability set (MMLU-Pro subset, GSM8K, a coding set, IFEval) and a vision spot check.
2. Run Heretic-MLX ARA, 100 trials, 400/400 prompts, export `preserve` in our int4 layout.
3. Same evals on the result, plus compare head-to-head against the third-party Heretic build we run now (Heretic-MLX has `--evaluate-model` for this).
4. Measure decode speed and draft acceptance with our patched Ollama.
5. Keep it if refusals are at or below the third-party build and capability is within noise. Publish method and numbers in the lab repo; keep the old model until the new one wins.

## Sources

- Heretic: https://github.com/p-e-w/heretic and releases https://github.com/p-e-w/heretic/releases
- Heretic-MLX: https://github.com/scoltzero/heretic-mlx
- Young 2026, "Is Abliteration Getting Harder?": https://arxiv.org/html/2611.17846
- Lai, norm-preserving biprojected abliteration: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
- Gemma 4 12B Heretic on Apple Silicon write-up (MPS fixes, keeping multimodal tensors): https://huggingface.co/blog/junafinity/gemma-4-12b-abliterated
- mlx-abliterate: https://github.com/nateGeorge/mlx-abliterate ; mlx-heretic: https://github.com/akaiHuang/mlx-heretic
- Gemma 4 Apache-2.0: https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html
- DFlash: https://arxiv.org/pdf/2602.06036 ; Draft-OPD: https://arxiv.org/pdf/2605.29343
