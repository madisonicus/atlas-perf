"""Phase 0 decode test: MTP on (prod Ollama :11434) against MTP off (dev
Ollama :11500 with OLLAMA_MLX_SPECULATIVE=0), interleaved, on an idle atlas.

    python3 decode.py [model] [runs]

Prints median and spread of decode tok/s per prompt kind and server, and
appends rows to results/phase0-decode.jsonl.
"""
import json, os, statistics, sys, time, urllib.request
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8:27b-nvfp4"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
H = os.environ.get("PHASE0_HOST", "http://127.0.0.1")
SERVERS = {"mtp-on": H + ":11434", "mtp-off": H + ":11500"}
PROMPTS = {
    "prose": "Explain in about 300 words how ZFS snapshots work and why they are cheap.",
    "code": "Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.",
    "chat": "My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters.",
}
OUT = os.environ.get("PHASE0_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase0-decode.jsonl"))

def run(base, prompt):
    body = {"model": MODEL, "stream": False, "think": False, "keep_alive": -1,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": 400, "temperature": 0}}
    req = urllib.request.Request(base + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["eval_count"] / (r["eval_duration"] / 1e9), r["eval_count"]

res = {}
for name, base in SERVERS.items():   # warm both
    run(base, "Say OK.")
for i in range(RUNS):
    for kind, p in PROMPTS.items():
        for name, base in (SERVERS.items() if i % 2 == 0 else reversed(list(SERVERS.items()))):
            tps, n = run(base, p)
            res.setdefault((kind, name), []).append(tps)
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL, "kind": kind, "server": name, "tok_s": round(tps, 1), "tokens": n}) + "\n")
for (kind, name), v in sorted(res.items()):
    print(f"{kind:6} {name:8} median {statistics.median(v):6.1f}  min {min(v):6.1f}  max {max(v):6.1f}  n={len(v)}")
