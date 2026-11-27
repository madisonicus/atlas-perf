"""Foreground (owner) latency with N background streams on one test server.
    python3 qos.py /path/ollama port parallel model bgs reps
Foreground: streamed chat, 400 tokens; reports first word (s) and tok/s.
Background: endless 400-token chats while the foreground runs."""
import json, os, subprocess, sys, threading, time, statistics, urllib.request
binp, port, par, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BGS = [int(x) for x in sys.argv[5].split(",")]; REPS = int(sys.argv[6])
FG = ["Explain in about 300 words how ZFS snapshots work and why they are cheap.",
      "My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters."]
BG = "Write a detailed 400-word description of a wildlife photo of a red fox in snow, then list 20 keywords."
env = dict(os.environ, OLLAMA_KEEP_ALIVE="-1", OLLAMA_CONTEXT_LENGTH="65536", OLLAMA_NUM_PARALLEL=par,
           OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_KV_CACHE_SIZE="0", OLLAMA_MLX_TURN_SNAPSHOTS="1")
pr = subprocess.Popen([binp, "serve"], env=env, cwd=os.path.dirname(binp), stdout=open("/tmp/qos.log", "w"), stderr=subprocess.STDOUT)
time.sleep(6)
U = f"http://127.0.0.1:{port}/api/chat"
def req(p, stream):
    b = {"model": model, "stream": stream, "think": False, "messages": [{"role": "user", "content": p}], "options": {"num_predict": 400, "temperature": 0.7}}
    return urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode()), timeout=1800)
def fg(p):
    t0 = time.time(); first = None; n = 0; r = req(p, True)
    for line in r:
        d = json.loads(line)
        if d.get("message", {}).get("content") and first is None: first = time.time() - t0
        if d.get("done"): n = d["eval_count"]; ev = d["eval_duration"] / 1e9
    return first, n / ev
stop = False
def bg():
    while not stop:
        try: req(BG, False).read()
        except Exception: time.sleep(1)
try:
    req("Say OK.", False).read()
    for nbg in BGS:
        stop = False
        ths = [threading.Thread(target=bg, daemon=True) for _ in range(nbg)]
        for t in ths: t.start()
        time.sleep(8 if nbg else 0)
        fw, ts = [], []
        for i in range(REPS):
            f, s = fg(FG[i % 2]); fw.append(f); ts.append(s)
        stop = True
        for t in ths: t.join()
        print(f"parallel {par}, {nbg} background: first word {statistics.median(fw):.2f} s, writing {statistics.median(ts):.1f} tok/s", flush=True)
finally:
    pr.terminate(); pr.wait()
