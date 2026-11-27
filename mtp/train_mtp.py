"""Self-distill the Qwen3.8 27B MTP head on the target's own greedy answers.

    uvx --from mlx-lm==0.32.2 --with pyarrow python train_mtp.py [epochs] [lr]

Target: ~/hf/qwen3.8-27b-mlx4 (frozen, 4-bit). MTP init: mtp.* from
~/hf/qwen3.8-27b (bf16). Data: data.jsonl (prompt, response). The last 60
rows are held out. Label at position t: the target's argmax after token t+1,
so the head learns what greedy verification accepts.
Writes mtp_trained.safetensors with HF names and HF norm convention.
"""
import json, os, sys, time, glob, random
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.qwen3_5 import DecoderLayer

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
HF = os.path.expanduser("~/hf/qwen3.8-27b")
MLX4 = os.path.expanduser("~/hf/qwen3.8-27b-mlx4")
MAXLEN = 1024
random.seed(0)

model, tok = load(MLX4)
lm = model.language_model
args = lm.args
D = args.hidden_size
model.freeze()

class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * D, D, bias=False)
        self.layer = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
        self.norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
    def __call__(self, h, emb):
        h, emb = h.astype(mx.float32), emb.astype(mx.float32)
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(h)], -1))
        x = self.layer(x, mask="causal")
        return self.norm(x)

# Load mtp.* weights (HF stores RMSNorm weights as w, applied as 1+w).
idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
raw = {}
for shard in sorted({v for k, v in idx.items() if k.startswith("mtp.")}):
    w = mx.load(f"{HF}/{shard}")
    raw.update({k: v for k, v in w.items() if k.startswith("mtp.")})
def to_local(k):
    return k.replace("mtp.layers.0.", "layer.").replace("mtp.", "")
mtp = MTP()
weights = []
for k, v in raw.items():
    v = v.astype(mx.bfloat16)
    if v.ndim == 1:
        v = v.astype(mx.float32) + 1.0
        v = v.astype(mx.bfloat16)
    weights.append((to_local(k), v))
mtp.load_weights(weights, strict=True)
mtp.set_dtype(mx.float32)  # fp32 master weights: bf16 cannot hold small Adam steps
mx.eval(mtp.parameters())
print("mtp params", sum(v.size for _, v in tree_flatten(mtp.parameters())) / 1e6, "M", flush=True)

rows = [json.loads(l) for l in open("data.jsonl")]
held, train = rows[-60:], rows[:-60]

def encode(r):
    msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
    ids = tok.apply_chat_template(msgs, enable_thinking=False, tokenize=True)
    pre = tok.apply_chat_template(msgs[:1], enable_thinking=False, tokenize=True, add_generation_prompt=True)
    return ids[:MAXLEN], min(len(pre), MAXLEN)

def features(r):
    ids, start = encode(r)
    x = mx.array(ids)[None]
    h = lm.model(x)                     # [1,T,D] final normed hidden
    tgt = mx.argmax(lm.lm_head(h), -1)  # target greedy next token at each t
    emb = lm.model.embed_tokens(x)
    mx.eval(h, tgt, emb)
    return h, emb, tgt, start, len(ids)

def lm_head(y):
    return lm.lm_head(y.astype(mx.bfloat16))

def loss_fn(mtp, h, emb, tgt, start):
    # position t: hidden h_t, embedding of token t+1, label tgt_{t+1}
    hh, ee, yy = h[:, :-1], emb[:, 1:], tgt[:, 1:]
    out = mtp(hh, ee)
    logits = lm_head(out).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, yy, reduction="none")
    m = (mx.arange(ce.shape[1]) >= max(start - 1, 0)).astype(mx.float32)[None]
    acc = ((mx.argmax(logits, -1) == yy).astype(mx.float32) * m).sum() / m.sum()
    return (ce * m).sum() / m.sum(), acc

def evaluate(feats):
    tot = n = ls = 0.0
    for h, emb, tgt, start, T in feats:
        loss, acc = loss_fn(mtp, h, emb, tgt, start)
        k = T - start
        tot += acc.item() * k; ls += loss.item() * k; n += k
    return f"{tot / n:.4f} (loss {ls / n:.3f})"

t0 = time.time()
held_f = [features(r) for r in held]
print(f"held-out features {time.time()-t0:.0f}s", flush=True)
print("stock head, held-out step-1 agreement:", evaluate(held_f), flush=True)

t0 = time.time()
train_f = [features(r) for r in train]
print(f"train features {len(train_f)} in {time.time()-t0:.0f}s", flush=True)

opt = optim.AdamW(learning_rate=LR, weight_decay=0.0)
lg = nn.value_and_grad(mtp, lambda m, *a: loss_fn(m, *a)[0])
step = 0
for ep in range(EPOCHS):
    random.shuffle(train_f)
    run = 0.0
    for h, emb, tgt, start, T in train_f:
        loss, g = lg(mtp, h, emb, tgt, start)
        opt.update(mtp, g)
        mx.eval(mtp.parameters(), opt.state, loss)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        step += 1
        if step % 100 == 0:
            print(f"ep {ep} step {step} loss {run:.3f} {time.time()-t0:.0f}s", flush=True)
    print(f"epoch {ep}: held-out step-1 agreement {evaluate(held_f)}", flush=True)

out = {}
for k, v in tree_flatten(mtp.parameters()):
    name = "mtp." + k.replace("layer.", "layers.0.", 1) if k.startswith("layer.") else "mtp." + k
    if v.ndim == 1:
        v = (v.astype(mx.float32) - 1.0).astype(mx.bfloat16)
    out[name] = v.astype(mx.bfloat16)
mx.save_safetensors("mtp_trained.safetensors", out)
print("saved", len(out), "tensors")
