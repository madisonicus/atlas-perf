# MLX runner: inkling-2-mini gains nothing from parallel requests

**Target:** ollama/ollama issues. Draft, not filed.

Ollama 0.40.18, M5 Ultra 96 GB. Total output tok/s, 1 request vs 4 at once
(wall clock, median of 3):

| model | 1 stream | 4 streams |
|---|---|---|
| qwen3.8:27b-nvfp4 | 98.5 | 212 |
| qwen3.6:35b-a3b | 198 | 322 |
| gemma4:31b | 80 | 188 |
| inkling-2-mini:26b-a5b-nvfp4 | 86 | 84 |

Every other model batches; Inkling 2 Mini does not. Possibly the looped
layers serialize requests.
