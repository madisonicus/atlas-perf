# atlas-perf

Faster local LLM inference on one Mac Studio (Apple M5 Ultra, 96 GB, 1.2 TB/s),
by patching the open-source stack (Ollama 0.40.18 and its MLX runner) and
measuring every change. Started 2026-11-26. Meant to move to its own repo.

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

## Patches (`patches/ollama`, against v0.40.18)

1. `OLLAMA_MLX_SPECULATIVE=0`: turn draft heads off, to measure them.
2. `OLLAMA_QUANTIZE_UNIFORM=1` (create), `OLLAMA_MLX_PAD_ROWS=1` and
   `OLLAMA_MLX_FUSE=1` (runner).

Built Go-only on the Mac (`go build`), with the MLX libraries of the shipped
Ollama next to the binary. No Xcode needed.

## Upstream leads

- MLX: no fast fp4 kernel for 2–5 rows on NAX GPUs (`use_qmv_mma` min_rows 6);
  affine has one from 4 rows (#4787, #4827). A fp port for 2–5 rows would
  speed up MTP verification and batched decode.
- Ollama: prefix cache restore stops at 131,072 tokens; no disk paging; no KV
  quantization on MLX; no request priority.
