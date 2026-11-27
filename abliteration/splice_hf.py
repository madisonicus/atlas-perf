"""Put heretic-mlx's edited weights back into the original Hugging Face checkpoint.

heretic-mlx exports in MLX naming and layout (for example
`language_model.model.layers.N...`, conv1d axes moved). Ollama's importer
wants the original Hugging Face form. The edit only changes plain Linear
weights (attention output and MLP down projections), and those have the
same layout in both forms. So we copy the original checkpoint and replace
only those tensors. Everything else (vision tower, MTP head, norms) stays
byte-identical to the original.

  python splice_hf.py ORIGINAL_HF_DIR HERETIC_EXPORT_DIR OUT_DIR
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
    orig_dir, exp_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    exp = load_dir(exp_dir)
    edits: dict[str, mx.array] = {}
    for k, v in exp.items():
        m = TARGET.match(k)
        if m:
            edits[f"model.language_model.layers.{m.group(1)}.{m.group(2)}.weight"] = v
    if not edits:
        sys.exit("no target tensors found in export")

    out_dir.mkdir(parents=True)
    for f in orig_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out_dir / f.name)

    changed, same, seen = 0, 0, set()
    for f in sorted(orig_dir.glob("*.safetensors")):
        tensors, meta = mx.load(str(f), return_metadata=True)
        for k in list(tensors):
            if k in edits:
                new = edits[k].astype(tensors[k].dtype)
                if new.shape != tensors[k].shape:
                    sys.exit(f"shape mismatch {k}: {new.shape} vs {tensors[k].shape}")
                if mx.array_equal(new, tensors[k]).item():
                    same += 1
                else:
                    changed += 1
                tensors[k] = new
                seen.add(k)
        mx.save_safetensors(str(out_dir / f.name), tensors, metadata=meta)
    missing = set(edits) - seen
    if missing:
        sys.exit(f"{len(missing)} edited tensors not in original, e.g. {sorted(missing)[:3]}")
    print(json.dumps({"target_tensors": len(edits), "changed": changed, "unchanged": same}))


if __name__ == "__main__":
    main()
