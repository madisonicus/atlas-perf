# MLX runner: prefix cache restore stops at 131,071 tokens on long chats

**Target:** ollama/ollama issues. Draft, not filed.

Ollama 0.40.18, MLX runner, `qwen3.8:27b-nvfp4`, M5 Ultra 96 GB.

A follow-up turn on a long chat (same history, previous answer appended, one
new user message) reports these cached prompt tokens:

| history | prompt tokens | cached | first word |
|---|---|---|---|
| 31k | 31,363 | 31,258 | 0.18 s |
| 70k | 69,576 | 69,471 | 0.26 s |
| 139k | 139,134 | **131,071** | **13.1 s** |

Above 2^17 tokens the restore stops at 131,071, and about 8k tokens are read
again on every turn. Raising the paged-out snapshot budget
(`maxPagedOutBytes`) from 8 GiB to 32 GiB does not change it.
`OLLAMA_CONTEXT_LENGTH=262144`. Repro script: `bench/ttft.py` (TURNS=40).
