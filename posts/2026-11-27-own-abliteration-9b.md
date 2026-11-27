# Our own abliterated 9B, in one afternoon

*2026-11-27. Mac Studio M5 Ultra, 96 GB. Every number below was measured on
this machine today. Data: [results/abliteration-9b.json](../results/abliteration-9b.json).
Method: [research/abliteration.md](../research/abliteration.md).*

Until today, our uncensored models came from strangers on Hugging Face. We
wanted our own builds: our quantization, the vision and draft-head tensors
kept, and a result we can reproduce. Today we made one for a 9B model, and
it took less than an hour of machine time.

## What we did

1. **Read the tool first.** heretic-mlx is an MLX port of Heretic. It is
   four days old and has one author. We read it for network calls, uploads
   and shell-outs, and found none outside the documented ones. Review:
   [reviews/2026-11-27-heretic-mlx-code-review.md](../reviews/2026-11-27-heretic-mlx-code-review.md).
   We pinned commit `e8ed39b` and ran it as a separate user without admin
   rights or a Hugging Face token.
2. **Search.** Arbitrary-Rank Ablation (ARA) learns a rank-50 correction
   for each attention and MLP output projection. It keeps answers to
   harmless prompts the same and moves harmful prompts toward harmless
   ones. Optuna tried 100 settings in 54 minutes (37 s each) on the bf16
   weights.
3. **Build.** We copied the edited projections back into the original
   checkpoint (27 of 64 changed, layers 12–26) and built a uniform int4
   model with our patched Ollama, next to an untouched build of the same
   base.

## Results

Refusals, judged by a classifier (not keyword counts), on 100 harmful and
100 benign JailbreakBench prompts:

| Model | Harmful refused | Benign refused |
|---|---|---|
| Stock base | 95 | 39 |
| **Ours, int4** | **13** | **7** |
| A third-party "heretic" 9B we had used | 97 | 37 |

Capability screen, int4, same settings for both:

| Model | GSM8K /100 | MMLU-Pro /140 | HumanEval /50 | Decode tok/s (prose / code / chat) |
|---|---|---|---|---|
| Stock base | 95 | 87 | 49 | 224 / 242 / 210 |
| **Ours** | **95** | **80** | **47** | **212 / 275 / 206** |

- Refusals fell from 95 to 13. The model also stopped refusing benign
  requests that the base refused (39 to 7), such as a poem about the
  Holocaust.
- Math is unchanged. MMLU-Pro is 7 lower. Paired per question that is 4
  against 11, p = 0.12: a possible small loss, not proven at this size.
- Speed is the same. The MTP draft head still works: acceptance 0.59
  against 0.63 on the base.
- Vision works: both builds read the shapes, colors and text of a test
  image.

## What surprised us

The third-party heretic 9B that we used refused as much as the stock base
with thinking off. We do not know what it was built from. This is the
best argument for building our own: we now know exactly what went in.

## What is not measured yet

- A larger capability set, IFEval, and answers with thinking on.
- The base is a community distill (Qwen3.5-9B trained on Qwen3.8 outputs).
  Qwen publishes no official Qwen3.8 9B. It has the same architecture as
  Qwen3.8 27B, which is the next target, then Gemma 4 31B.
