"""Peak GPU memory bandwidth on atlas with MLX: read N bytes, write N bytes.

    uvx --with mlx python bw.py      # on atlas, idle

Reports GB/s for a copy (read + write) and a reduction (read only), median
of 20 runs, for 1, 4 and 8 GiB arrays.
"""
import statistics, time
import mlx.core as mx

def bench(fn, n_bytes, reps=20):
    fn(); mx.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); mx.synchronize(); ts.append(time.perf_counter() - t)
    return n_bytes / statistics.median(ts) / 1e9, n_bytes / min(ts) / 1e9

for gib in (1, 4, 8, 16):
    n = gib * (1 << 30) // 2          # bf16 elements
    x = mx.ones((gib * 1024, 512 * 1024), dtype=mx.bfloat16); mx.eval(x)  # 2D: no int32 overflow
    copy_med, copy_best = bench(lambda: mx.eval(x * 1), 2 * n * 2)
    red_med, red_best = bench(lambda: mx.eval(mx.sum(x)), n * 2)
    print(f"{gib} GiB  copy {copy_med:.0f} GB/s (best {copy_best:.0f})  read-only {red_med:.0f} GB/s (best {red_best:.0f})")
    del x
