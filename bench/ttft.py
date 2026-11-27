"""Phase 0 time-to-first-word test on a long chat, as Open WebUI sends it.

    python3 ttft.py [model]

Turn 1: a ~30k-token chat history plus a question, the date stamp at the end
of the newest user message (as atlas's filter does). Turn 2: the same history
plus turn 1's answer and a new question, with a new stamp; the stamp on turn
1's question is gone, as in the stored chat. Reports prompt tokens, tokens
served from cache, and seconds to first word, for a cold start and a follow-up.
"""
import json, random, sys, time, urllib.request
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8:27b-nvfp4"
URL = "http://127.0.0.1:11434/api/chat"

def chat(msgs):
    body = {"model": MODEL, "stream": False, "think": False, "messages": msgs, "options": {"num_predict": 60, "temperature": 0}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    ttft = (r.get("load_duration", 0) + r["prompt_eval_duration"]) / 1e9
    return r, ttft

random.seed(7)
hist = [{"role": "system", "content": "You talk with Ben."}]
for t in range(int(__import__("os").environ.get("TURNS","40"))):   # ~40 turns of ~750 tokens each
    q = f"Turn {t} [{random.getrandbits(40):x}]: " + " ".join(f"note{random.randrange(10**6)}" for _ in range(350))
    hist += [{"role": "user", "content": q}, {"role": "assistant", "content": "Noted. " + " ".join(f"ok{random.randrange(10**6)}" for _ in range(150))}]
stamp = lambda m: f"\n\n[Current date and time: Friday, 2026-11-27 03:{m:02d} EST (America/New_York)]"
q1 = "Summarize the last turn in one sentence."
r1, t1 = chat(hist + [{"role": "user", "content": q1 + stamp(1)}])
print(f"turn 1 (cold): prompt {r1['prompt_eval_count']} tokens, cached {r1.get('prompt_eval_cached_count', 0)}, first word {t1:.2f} s")
hist2 = hist + [{"role": "user", "content": q1}, {"role": "assistant", "content": r1["message"]["content"]}]
r2, t2 = chat(hist2 + [{"role": "user", "content": "And the turn before it?" + stamp(2)}])
print(f"turn 2 (follow-up): prompt {r2['prompt_eval_count']} tokens, cached {r2.get('prompt_eval_cached_count', 0)}, first word {t2:.2f} s")
