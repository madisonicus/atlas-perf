"""Draw the README charts, light and dark, from the numbers in the results log.

    uv run --with matplotlib python charts/make_charts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(surface="#ffffff", text="#191919", text2="#666666", grid="#e6e6e6",
                  series=["#257de0", "#f47230", "#00b894"]),
    "dark": dict(surface="#0d1117", text="#efefef", text2="#9c9c9c", grid="#30363d",
                 series=["#338ced", "#e76624", "#00a888"]),
}

def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

def fig(t):
    f, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    f.patch.set_facecolor(t["surface"])
    style(ax, t)
    return f, ax

def title(ax, t, main, sub):
    ax.set_title(main, loc="left", color=t["text"], fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["text2"], fontsize=9)

def decode(t, name):
    builds = ["Official\nnvfp4", "Uniform\nnvfp4", "Uniform int4\n+ row padding", "Uniform int4\n+ DFlash2 + fused"]
    data = {"Prose": [94.2, 103.8, 91.2, 99.9], "Code": [121.5, 123.1, 157.8, 167.4], "Chat": [84.9, 95.5, 85.9, 85.7]}
    f, ax = fig(t)
    w = 0.26
    for i, (k, v) in enumerate(data.items()):
        xs = [x + (i - 1) * (w + 0.02) for x in range(len(builds))]
        ax.bar(xs, v, width=w, color=t["series"][i], label=k, zorder=3)
    for x, v in enumerate(data["Code"]):
        ax.text(x, v + 2, f"{v:.0f}", ha="center", color=t["text2"], fontsize=8)
    ax.set_xticks(range(len(builds)), builds, color=t["text"], fontsize=9)
    ax.set_ylabel("tok/s", color=t["text2"])
    ax.set_ylim(0, 185)
    leg = ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    for tx in leg.get_texts(): tx.set_color(t["text"])
    title(ax, t, "Decode speed of Qwen3.8 27B by build", "One stream, temperature 0, 400 tokens. Medians of 5–6 interleaved runs, idle M5 Ultra.")
    f.tight_layout(); f.savefig(name, facecolor=t["surface"]); plt.close(f)

def rows(t, name):
    r = [1, 2, 3, 4, 5, 6, 7, 8]
    nv = [14.35, 21.11, 24.39, 31.94, 38.48, 27.82, 28.14, 27.84]
    af = [14.35, 18.36, 25.91, 16.26, 17.16, 17.55, 18.67, 19.32]
    f, ax = fig(t)
    ax.plot(r, nv, color=t["series"][0], lw=2, marker="o", ms=5, label="nvfp4", zorder=3)
    ax.plot(r, af, color=t["series"][1], lw=2, marker="o", ms=5, label="affine 4-bit", zorder=3)
    ax.text(5.15, 38.48, "nvfp4", color=t["text"], fontsize=9, va="center")
    ax.text(8.15, 19.32, "affine 4-bit", color=t["text"], fontsize=9, va="center")
    ax.set_xlim(0.7, 9.2); ax.set_ylim(0, 42)
    ax.set_xticks(r, [str(x) for x in r], color=t["text"])
    ax.set_xlabel("rows per matmul (tokens verified at once)", color=t["text2"])
    ax.set_ylabel("ms per decode step (matmuls only)", color=t["text2"])
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for tx in leg.get_texts(): tx.set_color(t["text"])
    title(ax, t, "Checking 2–5 drafted tokens at once is slow on nvfp4", "All quantized matmuls of one Qwen3.8 27B step, MLX 0.32.5. One row = 87% of peak bandwidth.")
    f.tight_layout(); f.savefig(name, facecolor=t["surface"]); plt.close(f)

def parallel(t, name):
    n = [1, 2, 4]
    off = [92, 133, 195]; best = [98, 149, 171]
    f, ax = fig(t)
    ax.plot(n, off, color=t["series"][0], lw=2, marker="o", ms=6, label="Official nvfp4 (MTP head)", zorder=3)
    ax.plot(n, best, color=t["series"][1], lw=2, marker="o", ms=6, label="Uniform int4 + DFlash2 + fused", zorder=3)
    ax.set_xticks(n, ["1", "2", "4"], color=t["text"])
    ax.set_xlabel("parallel requests", color=t["text2"]); ax.set_ylabel("total tok/s", color=t["text2"])
    ax.set_ylim(0, 220)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for tx in leg.get_texts(): tx.set_color(t["text"])
    title(ax, t, "The best build depends on the mode", "Total output with 1, 2 and 4 requests at once. Wall clock, prefill included. Medians of 3.")
    f.tight_layout(); f.savefig(name, facecolor=t["surface"]); plt.close(f)

def bandwidth(t, name):
    labels = ["Peak (MLX read test)", "Matmuls of one step", "Whole plain decode step"]
    vals = [1150, 1006, 757]
    f, ax = fig(t)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.barh(labels[::-1], vals[::-1], color=t["series"][0], height=0.5, zorder=3)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 15, i, f"{v:,} GB/s", va="center", color=t["text"], fontsize=9)
    ax.set_xlim(0, 1400); ax.tick_params(axis="y", colors=t["text"])
    ax.set_xlabel("GB/s", color=t["text2"])
    title(ax, t, "Plain decode uses 66% of the memory bandwidth", "Qwen3.8 27B reads 14.45 GB per step. Apple's spec: 1.2 TB/s.")
    f.tight_layout(); f.savefig(name, facecolor=t["surface"]); plt.close(f)

for mode, t in THEMES.items():
    decode(t, f"charts/decode-{mode}.svg")
    rows(t, f"charts/rows-{mode}.svg")
    parallel(t, f"charts/parallel-{mode}.svg")
    bandwidth(t, f"charts/bandwidth-{mode}.svg")
print("done")
