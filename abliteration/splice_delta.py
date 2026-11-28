"""Apply an edit found on a quantized MLX model to the original bf16 checkpoint.

The search runs on a 4-bit MLX model to save memory. heretic-mlx then
exports with `--export-modified-mode bf16`: every edited output projection
is stored as bf16 (base dequantized + correction), every other layer stays
4-bit. Here we take the correction only,

    delta = W_edited (bf16 export) - dequantize(W_base, 4-bit source)

and add it to the same tensor of the original Hugging Face bf16 checkpoint.
All other tensors (vision tower, MTP head, norms) stay byte-identical.

  python splice_delta.py ORIG_HF_DIR BASE_MLX_DIR EXPORT_DIR OUT_DIR
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx

TARGET = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.(linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)\.weight$"
)


def load_dir(d: Path) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


def main() -> None:
    orig_dir, base_dir, exp_dir, out_dir = (Path(a) for a in sys.argv[1:5])
    qcfg = json.loads((base_dir / "config.json").read_text())["quantization"]
    exp = load_dir(exp_dir)
    base = load_dir(base_dir)

    deltas: dict[str, mx.array] = {}
    for k, v in exp.items():
        m = TARGET.match(k)
        if not m or (k[: -len("weight")] + "scales") in exp:
            continue  # not a target, or still quantized (not edited)
        p = k[: -len("weight")]
        q = qcfg.get(p[:-1]) if isinstance(qcfg.get(p[:-1]), dict) else qcfg  # per-layer override
        w0 = mx.dequantize(base[k], base[p + "scales"], base.get(p + "biases"),
                           group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", qcfg.get("mode", "affine")))
        d = (v.astype(mx.float32) - w0.astype(mx.float32)).astype(mx.bfloat16)  # bf16 keeps memory low
        mx.eval(d)
        deltas[f"model.language_model.layers.{m.group(1)}.{m.group(2)}.weight"] = d
    del exp, base
    if not deltas:
        sys.exit("no edited bf16 tensors in export")

    out_dir.mkdir(parents=True)
    for f in orig_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out_dir / f.name)

    done, rel = 0, []
    for f in sorted(orig_dir.glob("*.safetensors")):
        tensors, meta = mx.load(str(f), return_metadata=True)
        for k in list(tensors):
            if k in deltas:
                w = tensors[k]
                if deltas[k].shape != w.shape:
                    sys.exit(f"shape mismatch {k}: {deltas[k].shape} vs {w.shape}")
                new = (w.astype(mx.float32) + deltas[k].astype(mx.float32)).astype(w.dtype)
                rel.append((mx.linalg.norm(deltas[k].astype(mx.float32)) / mx.linalg.norm(w.astype(mx.float32))).item())
                tensors[k] = new
                done += 1
        mx.save_safetensors(str(out_dir / f.name), tensors, metadata=meta)
        mx.clear_cache()
    if done != len(deltas):
        sys.exit(f"applied {done} of {len(deltas)} deltas")
    print(json.dumps({"edited_tensors": done, "rel_change_min": min(rel), "rel_change_max": max(rel)}))


if __name__ == "__main__":
    main()
