Results on an Apple M5 Ultra (96 GB, 80-core GPU), Qwen3.8 27B uniform int4 + DFlash2, total output tok/s with N concurrent 400-token chats, median of 3, base 42ac2c0 vs this PR:

| clients | main | this PR |
|---:|---:|---:|
| 1 | 98.2 | 98.1 |
| 4 | 184.0 | 202.6 (+10%) |
| 8 | 167.7 | 264.5 (+58%) |

Without the PR, 8 clients gave less total throughput than 4. Single-stream speed is unchanged. We run it in production together with #19745; JSON output, tool calls, vision (Qwen3.6 35B-A3B) and embeddings checked fine. (Measured with Claude on my own machine.)
