"""Blind bakeoff: answer every prompt with every model, then shuffle labels.

  python run_blind.py prompts.json OUT_DIR MODEL [MODEL ...]

Writes OUT_DIR/answers.md (each prompt with answers labeled A, B, C ...,
in a random order per prompt) and OUT_DIR/key.json (which label is which
model). Grade answers.md first, then open key.json. Thinking off and the
model's own sampling settings, as a chat in Open WebUI gets them.
Models whose name ends in '!' are unloaded after their answers.
"""
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

H = "http://127.0.0.1:11434"


def post(path, body):
    req = urllib.request.Request(H + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def main():
    prompts = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    specs = sys.argv[3:]
    answers = {p["id"]: {} for p in prompts}
    for spec in specs:
        model, unload = spec.rstrip("!"), spec.endswith("!")
        for p in prompts:
            r = post("/api/chat", {"model": model, "stream": False, "think": False,
                                   "messages": [{"role": "user", "content": p["prompt"]}],
                                   "options": {"num_ctx": 16384, "num_predict": 1500}})
            answers[p["id"]][model] = r["message"]["content"]
        if unload:
            post("/api/generate", {"model": model, "keep_alive": 0})
        print(f"{model}: done", flush=True)

    rng = random.Random(time.time_ns())
    key, lines = {}, [f"# Blind bakeoff answers\n\n{len(prompts)} prompts, {len(specs)} models. "
                      "Grade each answer 1–5 before you open key.json.\n"]
    for p in prompts:
        models = list(answers[p["id"]])
        rng.shuffle(models)
        key[p["id"]] = {chr(65 + i): m for i, m in enumerate(models)}
        lines.append(f"\n---\n\n## {p['id']} ({p['area']})\n\n> {p['prompt']}\n")
        for i, m in enumerate(models):
            lines.append(f"\n### Answer {chr(65 + i)}\n\n{answers[p['id']][m].strip()}\n")
    (out / "answers.md").write_text("".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
