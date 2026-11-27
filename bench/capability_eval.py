"""Small capability check for an Ollama model: GSM8K, MMLU-Pro, HumanEval.

A quick screen for damage after abliteration, not a benchmark score.
Fixed subsets (first N of each pinned test split; MMLU-Pro takes the first
10 questions of each of its 14 categories). Thinking off, temperature 0.

  python capability_eval.py --ollama MODEL --name NAME --out out.jsonl

HumanEval runs the model's code in a subprocess with a timeout. Run it as an
unprivileged user.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import hf_hub_download

GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet")
MMLU = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/test-00000-of-00001.parquet")
HEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "openai_humaneval/test-00000-of-00001.parquet")


def load(spec):
    repo, rev, path = spec
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset", revision=rev))


def chat(host, model, prompt, max_tokens):
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["message"]["content"], d.get("eval_count", 0), d.get("eval_duration", 0)


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("####")[-1])
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_items(n):
    df = load(GSM8K).head(n)
    for q, a in zip(df.question, df.answer):
        prompt = f"{q}\n\nSolve step by step. End with a line of the form '#### <number>'."
        yield "gsm8k", prompt, a.split("####")[-1].strip().replace(",", "")


def mmlu_items(per_cat):
    df = load(MMLU)
    df = df.groupby("category", sort=True).head(per_cat)
    for _, row in df.iterrows():
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(row.options))
        prompt = (f"{row.question}\n\n{opts}\n\nThink briefly, then end with a line "
                  "'Answer: X' where X is the letter of the correct option.")
        yield "mmlu_pro", prompt, row.answer


def heval_items(n):
    df = load(HEVAL).head(n)
    for _, row in df.iterrows():
        prompt = ("Complete this Python function. Reply with the full function in one "
                  f"```python code block and nothing else.\n\n```python\n{row.prompt}```")
        yield "humaneval", prompt, {"prompt": row.prompt, "test": row.test, "entry": row.entry_point}


def run_code(resp, ref):
    m = re.findall(r"```(?:python)?\n(.*?)```", resp, re.S)
    code = m[0] if m else resp
    if f"def {ref['entry']}" not in code:
        code = ref["prompt"] + code
    prog = f"{ref['prompt']}\n{code}\n\n{ref['test']}\ncheck({ref['entry']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=15)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def score(task, resp, ref):
    if task == "gsm8k":
        got = last_number(resp)
        try:
            return got is not None and abs(float(got) - float(ref)) < 1e-6
        except ValueError:
            return False
    if task == "mmlu_pro":
        m = re.findall(r"Answer:\s*\**\s*\(?([A-J])\b", resp)
        return bool(m) and m[-1] == ref
    return run_code(resp, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm8k", type=int, default=100)
    ap.add_argument("--mmlu-per-cat", type=int, default=10)
    ap.add_argument("--humaneval", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    items = [*gsm8k_items(args.gsm8k), *mmlu_items(args.mmlu_per_cat), *heval_items(args.humaneval)]
    budget = {"gsm8k": 768, "mmlu_pro": 768, "humaneval": 768}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda it: chat(args.host, args.ollama, it[1], budget[it[0]]), items))
    wall = time.perf_counter() - t0

    rows, summary = [], {"name": args.name, "model": args.ollama, "wall_seconds": round(wall, 1)}
    for (task, prompt, ref), (resp, ntok, dur) in zip(items, outs):
        ok = score(task, resp, ref)
        rows.append({"task": task, "ok": ok, "tokens": ntok, "prompt": prompt[:200], "response": resp})
        s = summary.setdefault(task, {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(ok)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
