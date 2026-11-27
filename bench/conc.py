"""Aggregate decode with N concurrent clients, arms interleaved.
    python3 conc.py '{"base":["/path/ollama",11590],"pr":["/path/ollama",11591]}' model 1,4,8 reps
"""
import json, os, subprocess, sys, time, statistics, urllib.request, concurrent.futures as cf
arms = json.loads(sys.argv[1]); MODEL = sys.argv[2]
NS = [int(x) for x in sys.argv[3].split(",")]; REPS = int(sys.argv[4])
P = ["Explain in about 300 words how ZFS snapshots work and why they are cheap.",
     "My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters.",
     "Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.",
     "Describe the history of the printing press in about 300 words.",
     "Give me ideas for a 10-year-old's birthday party at home.",
     "Write a bash script that rotates log files older than 7 days with gzip.",
     "Explain how a heat pump works in winter, about 300 words.",
     "Recommend five board games for two players and say why."]
def start(name):
    binp, port = arms[name]
    env = dict(os.environ, OLLAMA_KEEP_ALIVE="-1", OLLAMA_CONTEXT_LENGTH="65536", OLLAMA_NUM_PARALLEL="8", OLLAMA_HOST=f"127.0.0.1:{port}")
    p = subprocess.Popen([binp, "serve"], env=env, cwd=os.path.dirname(binp), stdout=open(f"/tmp/conc-{name}.log", "a"), stderr=subprocess.STDOUT)
    time.sleep(6); one(port, "Say OK.")
    return p
def one(port, p):
    b = {"model": MODEL, "stream": False, "think": False, "messages": [{"role": "user", "content": p}], "options": {"num_predict": 400, "temperature": 0}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", json.dumps(b).encode()), timeout=1800))
    return r["eval_count"]
res = {}
items = list(arms.items())
for rep in range(REPS):
    for name, (_, port) in (items if rep % 2 == 0 else items[::-1]):
        p = start(name)
        try:
            for n in NS:
                t = time.time()
                with cf.ThreadPoolExecutor(n) as ex:
                    toks = sum(ex.map(lambda i: one(port, P[i % len(P)]), range(n)))
                res.setdefault((n, name), []).append(toks / (time.time() - t))
                print(rep, name, n, round(res[(n, name)][-1], 1), flush=True)
        finally:
            p.terminate(); p.wait(); time.sleep(3)
base = list(arms)[0]
for n in NS:
    print(f"{n} clients: " + "  ".join(f"{a}={statistics.median(res[(n,a)]):.1f}({statistics.median(res[(n,a)])/statistics.median(res[(n,base)])-1:+.1%})" for a in arms))
