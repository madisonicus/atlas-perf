"""Generate the target's own greedy answers for MTP self-distillation."""
import json, random, sys, os, concurrent.futures, urllib.request
import pyarrow.parquet as pq
random.seed(7)
uc = pq.read_table(os.path.expanduser("~/lab-data/ultrachat-test.parquet")).column("messages").to_pylist()
prompts = [c[0]["content"] for c in uc if c and c[0]["role"] == "user" and 20 < len(c[0]["content"]) < 1500]
random.shuffle(prompts)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
train, held = prompts[:N], prompts[N:N+60]
json.dump(held, open("heldout_prompts.json", "w"))
def gen(p):
    b = {"model": "qwen3.8:27b-int4u", "stream": False, "think": False,
         "messages": [{"role": "user", "content": p}], "options": {"num_predict": 512, "temperature": 0}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11504/api/chat", json.dumps(b).encode()), timeout=1800))
    return {"prompt": p, "response": r["message"]["content"], "eval_count": r["eval_count"]}
done = set()
if os.path.exists("data.jsonl"):
    done = {json.loads(l)["prompt"] for l in open("data.jsonl")}
with open("data.jsonl", "a") as fh, concurrent.futures.ThreadPoolExecutor(4) as ex:
    for i, row in enumerate(ex.map(gen, [p for p in train if p not in done])):
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if i % 50 == 0: print(i, flush=True)
