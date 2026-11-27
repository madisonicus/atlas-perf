# MLX runner: chat follow-ups on hybrid models re-read up to 8k tokens

**Target:** ollama/ollama issues. Draft, not filed. Supersedes the earlier
"restore stops at 131,071 tokens" draft: that was the 16th 8192-token mark.

Ollama 0.40.18, MLX runner, Qwen3.8 27B (hybrid: Gated DeltaNet + attention),
M5 Ultra 96 GB.

Chat clients often change the previous user message on the next turn (Open
WebUI filters add and drop a date stamp; attachments and RAG context do the
same). The new prompt then diverges inside the previous turn. Attention
caches can trim to any offset, but recurrent state can only restore from a
snapshot. The prefix cache snapshots every 8192 tokens and at the prompt
end, so the restore point falls back to the last 8192 mark, and up to 8k
tokens are read again on every turn.

| chat tokens | first word, stock | with turn snapshots |
|---|---|---|
| 3.6k | 2.02 s | 0.11 s |
| 10k | 1.45 s | 0.12 s |
| 21k | 3.13 s | 0.15 s |
| 42k | 0.73 s | 0.19 s |
| 70k | 4.21 s | 0.25 s |
| 139k | 12.5 s | 0.46 s |

Fix: also schedule snapshots at the start of the last few turns (the
positions of the turn-start token, `<|im_start|>` for Qwen) in
`scheduleGenerationSnapshots`. About 40 lines; patch
`patches/ollama/0006-mlxrunner-turn-start-snapshots.patch`. Each extra
snapshot is one recurrent state (~150 MiB here). Repro:
`bench/turns_ab.py`.
