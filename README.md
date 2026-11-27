# atlas-perf

Faster local LLM inference on one Mac Studio (Apple M5 Ultra, 96 GB, 1.2 TB/s),
by patching the open-source stack (Ollama 0.40.18 and its MLX runner) and
measuring every change. Started 2026-11-26.

**Goal:** cut the time from send to last word in half against the Phase 0
baseline, with no quality loss, and offer each patch that helps upstream.
Part goal for decode: +50% single-stream tok/s on `qwen3.8:27b` (average of
prose, code, chat), from 99.7 tok/s.

## Method

- Idle machine only. Temperature 0, 400 output tokens, three prompt kinds.
- Arms run interleaved, A/B/A/B, so drift hits every arm alike.
- Noise floor (A/A, same build, 20 runs): the median-to-median gap is under
  0.1% at the 95th percentile. Gains of 0.5% are real.
- Small gains count, and they can interact: patches are tested as a stack
  and then left out one at a time, not only alone.
- Quality gate: 20 tool calls and six blind-graded tasks (`../bakeoff`).

## Results log

| Date | Change | Measured | Note |
|---|---|---|---|
| 2026-11-27 | Baseline, official `qwen3.8:27b-nvfp4` | 99.7 tok/s avg (prose 91.7, code 124.1, chat 83.3); plain decode without MTP 52.4 | `bench/decode.py` |
| 2026-11-27 | Peak bandwidth | 1,150 GB/s read | `bench/bw.py`; plain decode uses 66% of it |
| 2026-11-27 | Kernel timing, one decode step | matmuls 14.3 of 19.1 ms; 2–4 rows cost 1.3–2.2x one row on nvfp4; affine 4-bit 1.1x from 4 rows | `bench/qmv.py` |
| 2026-11-27 | Uniform nvfp4 (no 8-bit layers) | 107.5 vs 100.2 (+7.3%) | patch 0002, `OLLAMA_QUANTIZE_UNIFORM=1` |
| 2026-11-27 | Uniform int4 + pad 2–3 rows to 4 | 111.6 vs 100.2 (+11.4%); code 157.8 | patch 0002, `OLLAMA_MLX_PAD_ROWS=1` |
| 2026-11-27 | Padding alone (int4) | 109.9 vs 105.1 (+4.6%) | |
| 2026-11-27 | Fused output gates (qwen3.5) | 109.6 vs 109.1 (+0.5%) | patch 0002, `OLLAMA_MLX_FUSE=1`; kept |
| 2026-11-27 | Quality, uniform int4 | 20/20 tool calls; blind tasks 0.82 vs 0.83 official | same grader |
| 2026-11-27 | Newer MLX 0.32.5 (libmlx from the PyPI wheel, swapped into the dev build) | official 99.7 (no change) | runs cleanly; MLX version shows 0.32.5 |
| 2026-11-27 | Interaction: padding on MLX 0.32.5 | code 139.3 with padding vs 147.6 without (−5.6%) | MLX 0.32.5 has its own small-row path; padding now hurts |
| 2026-11-27 | MLX 0.32.5, int4u, no padding | 111.4 avg | same +11% as old MLX with padding |
| 2026-11-27 | MLX 0.32.5, fusion on vs off (int4u, no pad) | 108.3 vs 110.2 | fusion −1.7% here; it was +0.5% on old MLX |
| 2026-11-27 | MLX 0.32.5, uniform nvfp4 + fusion | 108.3 (prose 102.3, code 132.4, chat 90.2) | nvfp4 leads prose and chat, int4 leads code |

## Patches (`patches/ollama`, against v0.40.18)

1. `OLLAMA_MLX_SPECULATIVE=0`: turn draft heads off, to measure them.
2. `OLLAMA_QUANTIZE_UNIFORM=1` (create), `OLLAMA_MLX_PAD_ROWS=1` and
   `OLLAMA_MLX_FUSE=1` (runner).

Built Go-only on the Mac (`go build`), with the MLX libraries of the shipped
Ollama next to the binary. No Xcode needed.

## What we learned so far

- Patches interact. The same padding patch is +4.6% on the MLX that Ollama
  0.40.18 ships and −5.6% on code with MLX 0.32.5. Fusion flips sign too. So
  every patch is judged inside the stack it ships in.
- The format that wins depends on the text: nvfp4 is faster on prose and chat,
  int4 on code, where MTP drafts deeper and cheap 4-row verifies pay off.
- Best so far: about +11% (111 tok/s) against 100. The +50% goal needs more
  accepted tokens per weight pass, not only kernels.

## Upstream leads

- MLX: no fast fp4 kernel for 2–5 rows on NAX GPUs (`use_qmv_mma` min_rows 6);
  affine has one from 4 rows (#4787, #4827). A fp port for 2–5 rows would
  speed up MTP verification and batched decode.
- Ollama: prefix cache restore stops at 131,072 tokens; no disk paging; no KV
  quantization on MLX; no request priority.
