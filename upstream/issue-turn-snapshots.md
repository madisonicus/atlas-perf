Chat follow-ups on hybrid models (Qwen3.5+, Gated DeltaNet + attention) re-read up to 8k prompt tokens on every turn, because the MLX prefix cache can restore recurrent state only from a snapshot, and it takes snapshots only every 8192 tokens and near the end of the prompt.

**When it happens.** Chat clients often change the previous user message on the next turn: Open WebUI filters add a date stamp to the newest message only, and attachments or retrieved context do the same. The follow-up prompt then differs from the cached one inside the previous turn. Attention caches can trim to that point, but recurrent state falls back to the last 8192-token snapshot, and everything after it is read again. A new chat that reuses a long system prompt hits the same gap.

**Measured** (Apple M5 Ultra 96 GB, Ollama 0.40.18, MLX runner, Qwen3.8 27B int4, follow-up turn where only the previous message's stamp changes):

| chat length | first word, now | with turn-start snapshots |
|---|---|---|
| 3.6k tokens | 2.02 s | 0.11 s |
| 10k | 1.45 s | 0.12 s |
| 21k | 3.13 s | 0.15 s |
| 42k | 0.73 s | 0.19 s |
| 70k | 4.21 s | 0.25 s |
| 139k | 12.5 s | 0.46 s |

A new chat with a 6.4k-token system prompt (tools, skills): 3.7 s now, 0.2 s with the change.

**Repro.** Send a long chat to a Qwen3.5+ MLX model, then send the same history plus the model's answer and a new user message, with a few tokens removed from the end of the previous user message. The log shows `cache hit ... matched=N cached=M` with M at the last multiple of 8192 (or 0) below N.

**Proposed fix.** In `scheduleGenerationSnapshots`, also snapshot at the start of the last few turns (positions of the turn-start token, `<|im_start|>` for Qwen, `<start_of_turn>` for Gemma). About 40 lines, plus tests: `TestLastTurnStarts`, and `TestScenarioEditedPreviousTurn` on the hybrid-cache scenario harness (restore moves from the last periodic mark to the edited turn's start). Each extra snapshot is one recurrent state (about 150 MiB for this model). Branch, ready as a PR if you agree with the approach: https://github.com/madisonicus/ollama/tree/turn-start-snapshots

Disclosure: I found, wrote and measured this with Claude (Anthropic) on my own Mac Studio, and reviewed it before filing.
