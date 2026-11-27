# [Metal] fp4 quantized_matmul has no fast path for 2–5 rows on NAX GPUs (M5 Ultra)

**Target:** ml-explore/mlx issues. Draft, not filed.

On an M5 Ultra (MLX 0.32.5), `quantized_matmul` with nvfp4 weights costs
almost as much per row at 2–5 rows as at one row, while affine 4-bit drops
to about 1.1x one row from 4 rows. 2–5 rows is exactly the size of
speculative-decoding verification (MTP, DFlash) and small batched decode.

All quantized matmuls of one Qwen3.8 27B decode step (64 dense MLP, 48
linear-attention and 16 attention layers, lm_head), distinct weights per
layer, chained in one graph, median of 20 (`bench/qmv.py`):

| rows | nvfp4 ms | affine 4-bit ms |
|---|---|---|
| 1 | 14.35 | 14.35 |
| 2 | 21.11 | 18.36 |
| 3 | 24.39 | 25.91 |
| 4 | 31.94 | 16.26 |
| 5 | 38.48 | 17.16 |
| 6 | 27.82 | 17.55 |
| 8 | 27.84 | 19.32 |

`use_qmv_mma` sets `min_rows = 6` for fp modes on NAX, and affine 4-bit gets
`qmv_nax` from 4 rows (#4787, #4827). Affine 2–3 rows are also slow.

End to end in Ollama 0.40.18, a uniform int4 build of the same model decodes
+11% faster than the nvfp4 build with MTP on, and code +27%, from cheaper
verification. A fp `qmv_mma`/`qmv_nax` path for 2–5 rows, or a lower
`min_rows` where it wins, would give this to nvfp4 models.

Hardware: Mac Studio, M5 Ultra (80-core GPU), 96 GB, macOS 27.0.1.
