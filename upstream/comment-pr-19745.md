Tested this on an Apple M5 Ultra (96 GB), Qwen3.8 27B int4 + DFlash2, combined with #19769. The restore works well: a 21k-token chat answers in 1.4–1.5 s after a model unload or an Ollama restart, instead of 13.9 s.

Two things came up in daily use:

**1. Disk writes per turn.** With the default 100 ms write delay, each turn of a 15k-token chat wrote about 2.2 GB (iostat; 5 turns 10 s apart moved 11 GB). On a Mac the SSD is soldered, and batch workloads multiply this. Making the delay configurable fixed it for us: with `OLLAMA_KV_CACHE_WRITE_DELAY=10m`, the same 5 turns moved 7 MB, and the unload still wrote the conversation once (2.2 GB, 1.5 s) because unload, shutdown and pressure already bypass the delay. A longer default, or the env var, would make the feature safe to leave on. Our change is about 15 lines: https://github.com/madisonicus/ollama/commit/b60c7810

**2. A per-request opt-out.** Some chats must never reach disk (Open WebUI's Temporary Chats). We added a request option `no_cache_persist` that drops the slot's queued checkpoint; memory caching is unchanged. Verified: 5 flagged turns, empty cache directory. https://github.com/madisonicus/ollama/commit/527734d0

Happy to open these as PRs against this branch if useful. (Found and measured with Claude on my own machine.)
