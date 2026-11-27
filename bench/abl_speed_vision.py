"""Decode speed and a vision spot check for Ollama models, one at a time.

  python abl_speed_vision.py IMAGE.png MODEL [MODEL ...]

Per model: one warm-up, then 3 runs of each decode.py prompt (400 tokens,
thinking off, temperature 0), then one image question. Prints one JSON line
per model with its start and end time, so the server log's speculative
decode stats can be matched to it. Models whose name ends in '!' are
unloaded after the test.
"""
import base64
import json
import statistics
import sys
import time
import urllib.request

H = "http://127.0.0.1:11434"
PROMPTS = {
    "prose": "Explain in about 300 words how ZFS snapshots work and why they are cheap.",
    "code": "Write a Python function that parses an nginx access log line into a dict, with a docstring and 5 pytest tests.",
    "chat": "My D&D group meets Saturday. Suggest three short side quests in Vallaki for level 5 characters.",
}
VISION_Q = "List every shape in this image with its color, and copy any text exactly."


def post(path, body):
    req = urllib.request.Request(H + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def chat(model, content, images=None, n=400):
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    return post("/api/chat", {"model": model, "stream": False, "think": False, "messages": [msg],
                              "options": {"num_predict": n, "temperature": 0, "num_ctx": 16384}})


def main():
    img = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
    for spec in sys.argv[2:]:
        model, unload = spec.rstrip("!"), spec.endswith("!")
        chat(model, "Say hi.", n=8)
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        speeds = {}
        for kind, p in PROMPTS.items():
            runs = []
            for _ in range(3):
                r = chat(model, p)
                runs.append(r["eval_count"] / (r["eval_duration"] / 1e9))
            speeds[kind] = round(statistics.median(runs), 1)
        vis = chat(model, VISION_Q, images=[img], n=200)["message"]["content"]
        t1 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if unload:
            post("/api/generate", {"model": model, "keep_alive": 0})
        print(json.dumps({"model": model, "start": t0, "end": t1, "tok_s_median": speeds, "vision": vis}), flush=True)


if __name__ == "__main__":
    main()
