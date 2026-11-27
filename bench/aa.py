"""A/B decode on atlas: list of (label, base url, model). Interleaved, N runs."""
import json, statistics, sys, urllib.request
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = [("A", "http://127.0.0.1:11500", "qwen3.8:27b-int4u"),
        ("B", "http://127.0.0.1:11500", "qwen3.8:27b-int4u")]
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
import random
for k in PROMPTS:
    a, b = res[("A", k)], res[("B", k)]
    print(k, "A median", round(statistics.median(a),1), "B median", round(statistics.median(b),1), "per-run spread (min-max) A", round(min(a),1), round(max(a),1))
    # half-split noise: median of 10 random runs vs median of the other 10, 200 draws
    allv = a + b; diffs = []
    for _ in range(200):
        random.shuffle(allv); x, y = allv[:len(a)//2], allv[len(a)//2:len(a)]
        diffs.append(abs(statistics.median(x)/statistics.median(y) - 1) * 100)
    diffs.sort(); print(k, "noise: 95th pct of median-vs-median gap for n=%d: %.1f%%" % (len(a)//2, diffs[190]))
for label, _, _ in ARMS:
    meds = [statistics.median(res[(label, k)]) for k in PROMPTS]
    print(f"{label:20} " + "  ".join(f"{k} {v:6.1f}" for k, v in zip(PROMPTS, meds)) + f"   avg {sum(meds)/3:6.1f}")
