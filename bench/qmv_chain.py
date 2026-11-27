"""Quantized matmuls of one Qwen3.8 27B decode step, chained in model order.

    uvx --with mlx python qmv_chain.py [mode] [rows...]

Unlike qmv.py, each layer's input is the previous layer's output, as in the
real forward, so the GPU cannot overlap independent matmuls. Layer pattern:
3 linear-attention layers, then 1 full-attention layer, 16 times. Each layer
has its MLP. Median of 20 steps.
"""
import statistics, sys, time
import mlx.core as mx

MODE = sys.argv[1] if len(sys.argv) > 1 else "affine"
ROWS = [int(b) for b in sys.argv[2:]] or [1, 2, 3, 4, 5, 6, 8]
CFG = {"nvfp4": dict(group_size=16, bits=4), "affine": dict(group_size=64, bits=4)}[MODE]
mode = "affine" if MODE == "affine" else MODE

def q(out_f, in_f):
    w = mx.random.normal((out_f, in_f)).astype(mx.bfloat16) * 0.02
    t = mx.quantize(w, mode=mode, **CFG)
    mx.eval(t)
    return t

def mm(x, t):
    return mx.quantized_matmul(x, t[0], t[1], t[2] if len(t) > 2 else None, transpose=True, mode=mode, **CFG)

layers = []
for i in range(64):
    if i % 4 == 3:
        mix = ("attn", q(12288, 5120), q(1024, 5120), q(1024, 5120), q(5120, 6144))
    else:
        mix = ("gdn", q(10240, 5120), q(6144, 5120), q(5120, 6144))
    layers.append((mix, q(17408, 5120), q(17408, 5120), q(5120, 17408)))
head = q(248320, 5120)

def step(x):
    for mix, gate, up, down in layers:
        if mix[0] == "attn":
            qo = mm(x, mix[1]); k = mm(x, mix[2]); v = mm(x, mix[3])
            a = qo[:, :6144] + mx.concatenate([k, v] * 3, axis=-1)
            x = x + mm(a, mix[4])
        else:
            qkv = mm(x, mix[1]); z = mm(x, mix[2])
            a = qkv[:, :6144] * z
            x = x + mm(a, mix[3])
        h = mx.sigmoid(mm(x, gate)) * mm(x, up)
        x = x + mm(h, down)
    return mm(x, head)

for r in ROWS:
    x = (mx.random.normal((r, 5120)) * 0.1).astype(mx.bfloat16)
    mx.eval(step(x)); mx.synchronize()
    ts = []
    for _ in range(20):
        t = time.perf_counter(); mx.eval(step(x)); mx.synchronize(); ts.append(time.perf_counter() - t)
    print(f"{MODE} rows {r}: {statistics.median(ts)*1e3:6.2f} ms per step")
