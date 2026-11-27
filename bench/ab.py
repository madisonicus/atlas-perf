"""A/B decode on atlas: list of (label, base url, model). Interleaved, N runs."""
import json, statistics, sys, urllib.request
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = [("official nvfp4", "http://127.0.0.1:11434", "qwen3.8:27b-nvfp4"),
        ("uniform nvfp4", "http://127.0.0.1:11500", "qwen3.8:27b-nvfp4u"),
        ("uniform int4 + pad", "http://127.0.0.1:11500", "qwen3.8:27b-int4u")]
PROMPTS = {"prose": "Explain in about 300 words how ZFS snapshots work and why they are cheap.",
           "code": "Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.",
           "chat": "My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters."}
def run(base, model, p):
    b = {"model": model, "stream": False, "think": False, "keep_alive": -1, "messages": [{"role": "user", "content": p}], "options": {"num_predict": 400, "temperature": 0}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/api/chat", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["eval_count"] / (r["eval_duration"] / 1e9)
for _, base, m in ARMS: run(base, m, "Say OK.")
res = {}
for i in range(RUNS):
    arms = ARMS if i % 2 == 0 else ARMS[::-1]
    for k, p in PROMPTS.items():
        for label, base, m in arms:
            res.setdefault((label, k), []).append(run(base, m, p))
for label, _, _ in ARMS:
    meds = [statistics.median(res[(label, k)]) for k in PROMPTS]
    print(f"{label:20} " + "  ".join(f"{k} {v:6.1f}" for k, v in zip(PROMPTS, meds)) + f"   avg {sum(meds)/3:6.1f}")
