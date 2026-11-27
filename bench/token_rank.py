"""Rank Qwen3.8 token ids by frequency over public text (wikitext-103,
UltraChat test split, Go and Python source), then write the "mix" list the
draft head uses: low ids and ranked ids interleaved.

    uvx --with tokenizers --with pyarrow python token_rank.py

Inputs in ~/lab-data. Output: qwen3.8-token-rank.bin, qwen3.8-token-mix.bin
(little-endian int32 ids, best first).
"""
import glob, collections, struct, sys, os
import pyarrow.parquet as pq
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.expanduser("~/hf/qwen3.8-27b/tokenizer.json"))
texts = []
t = pq.read_table(os.path.expanduser("~/lab-data/wikitext-103-train-0.parquet")).column("text").to_pylist()
buf, n = [], 0
for line in t:
    buf.append(line); n += len(line)
    if n > 60_000_000: break
texts.append("".join(buf))
uc = pq.read_table(os.path.expanduser("~/lab-data/ultrachat-test.parquet")).column("messages").to_pylist()
texts.append("\n".join(m["content"] for conv in uc for m in conv)[:80_000_000])
code = []
for f in glob.glob(os.path.expanduser("~/src/ollama/**/*.go"), recursive=True)[:3000]:
    code.append(open(f, errors="ignore").read())
for f in glob.glob("/opt/homebrew/lib/python3*/**/*.py", recursive=True)[:3000]:
    code.append(open(f, errors="ignore").read())
texts.append("\n".join(code)[:40_000_000])
c = collections.Counter()
for t in texts:
    for i in range(0, len(t), 1_000_000):
        c.update(tok.encode(t[i:i+1_000_000], add_special_tokens=False).ids)
print("tokens", sum(c.values()), "distinct", len(c))
ids = [i for i, _ in c.most_common()]
# Always include the chat template specials (ids >= 248000 region) and ids < 256 (bytes).
vocab = tok.get_vocab_size()
special = [i for i in range(vocab) if tok.id_to_token(i) and tok.id_to_token(i).startswith("<|")]
order = list(dict.fromkeys(special + ids))
out = os.path.expanduser("~/lab-data/qwen3.8-token-rank.bin")
with open(out, "wb") as fh:
    fh.write(struct.pack(f"<{len(order)}i", *order))
tot = sum(c.values())
for n in (8192, 16384, 32768, 65536):
    print(n, "coverage", round(sum(cnt for _, cnt in c.most_common(n)) / tot, 4))
print("wrote", out, len(order), "vocab", vocab, "specials", len(special))

lo, seen, mix = 0, set(), []
for x in order:
    for y in (lo, x):
        if y not in seen:
            seen.add(y); mix.append(y)
    lo += 1
with open(os.path.expanduser("~/lab-data/qwen3.8-token-mix.bin"), "wb") as fh:
    fh.write(struct.pack(f"<{len(mix)}i", *mix))
