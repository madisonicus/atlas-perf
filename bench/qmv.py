"""Where does a decode step's time go? Quantized matmuls of qwen3.8 27B on MLX.

    uvx --with mlx python qmv.py [mode] [batch...]

Builds random weights with the model's real shapes and counts (64 MLP, 48
linear-attention and 16 full-attention layers, lm_head), quantized like the
Ollama build. Times one full "step" of all these matmuls chained in one
graph (no per-op sync), median of 20, and reports bytes moved per second.
Compare with plain decode: 52.4 tok/s = 19.1 ms per step.
"""
import statistics, sys, time
import mlx.core as mx

MODE = sys.argv[1] if len(sys.argv) > 1 else "nvfp4"
BATCHES = [int(b) for b in sys.argv[2:]] or [1, 2, 4, 8]
CFG = {"nvfp4": dict(group_size=16, bits=4), "mxfp4": dict(group_size=32, bits=4),
       "affine": dict(group_size=64, bits=4), "mxfp8": dict(group_size=32, bits=8)}[MODE]
mode = "affine" if MODE == "affine" else MODE
SHAPES = [(64, 17408, 5120), (64, 17408, 5120), (64, 5120, 17408),      # gate, up, down
          (48, 10240, 5120), (48, 6144, 5120), (48, 5120, 6144),        # linear attn qkv, z, out
          (16, 12288, 5120), (16, 1024, 5120), (16, 1024, 5120), (16, 5120, 6144),  # q, k, v, o
          (1, 248320, 5120)]                                            # lm_head

# One distinct weight per layer, as in the real model: a shared weight would
# sit in the chip's caches and overstate the bandwidth.
layers, nbytes = [], 0
for count, out_f, in_f in SHAPES:
    for _ in range(count):
        w = mx.random.normal((out_f, in_f)).astype(mx.bfloat16)
        q = mx.quantize(w, mode=mode, **CFG)
        mx.eval(q)
        nbytes += sum(a.nbytes for a in q)
        layers.append((1, in_f, q))
        del w
print(f"{MODE}: quantized weights read per step {nbytes/1e9:.2f} GB")

def step(xs):
    outs = []
    for count, in_f, q in layers:
        wq, scales = q[0], q[1]
        biases = q[2] if len(q) > 2 else None
        for _ in range(count):
            outs.append(mx.quantized_matmul(xs[in_f], wq, scales, biases, transpose=True, mode=mode, **CFG))
    return outs

for b in BATCHES:
    xs = {d: mx.random.normal((b, d)).astype(mx.bfloat16) for d in (5120, 6144, 17408)}
    mx.eval(xs)
    mx.eval(step(xs)); mx.synchronize()
    ts = []
    for _ in range(20):
        t = time.perf_counter(); mx.eval(step(xs)); mx.synchronize(); ts.append(time.perf_counter() - t)
    med = statistics.median(ts)
    print(f"batch {b}: {med*1e3:6.2f} ms per step, {nbytes/med/1e9:6.0f} GB/s, {b/med:6.1f} tok/s if matmuls were all")
