# Code review: heretic-mlx before first run (2026-11-27)

We read heretic-mlx before we ran it, because it is days old, has one
author, and has no outside review. The question was narrow: can it do
anything on our machine other than edit a model? This is not a review of
the math.

- Repo: https://github.com/scoltzero/heretic-mlx (AGPL-3.0-or-later)
- **Pinned commit: `e8ed39bbeb76db0ed0098bd631e50fc4fa439e81`** (2026-11-25,
  "perf(ara): replace CPU SciPy L-BFGS-B with compiled MLX GPU Strong-Wolfe
  L-BFGS"). The repo has two commits.
- Size: 31 files. About 170 KB of Python in `src/`, plus tests, configs, a
  logo and `uv.lock`.

## What we read

- Read in full: `cli.py`, `model.py`, `config.py`, `plugins.py`,
  `scorers.py`, `abliteration.py`, `reproduce.py`, `upload.py`,
  `pyproject.toml`, the two GitHub workflows and the configs.
- Searched, not read line by line: `ara.py` and `quant.py`. They import
  only `mlx`, `numpy`, `optuna`, `rich`, `tqdm` and the package itself, and
  they have no file, process or network calls.
- Searched all of `src/` for `subprocess`, `os.system`, `eval(`/`exec(`
  (only `mx.eval` found), `pickle`, `base64`, `marshal`, `ctypes`,
  `__import__`, `os.environ` and URLs.
- `uv.lock`: 293 packages, all from `https://pypi.org/simple`, no git or
  URL sources.

## Findings

No telemetry, no hidden network calls, no obfuscated code. Every outside
action is one that the tool documents:

| Action | Where | When it happens | Our control |
|---|---|---|---|
| Model download from Hugging Face | `model.py` `_resolve_model_dir` → `snapshot_download` | Only if the model argument is not a local path | We pass a local directory |
| Prompt datasets from Hugging Face | `config.py` → `datasets.load_dataset` | Default prompt sets (`mlabonne/harmless_alpaca`, `mlabonne/harmful_behaviors`) | Read only. Can pin with `commit` or use local files |
| Classifier model from Hugging Face | `scorers.py` `BERTClassifierScorer` | Only if that scorer is configured | Pin with `model_commit` |
| **Upload to Hugging Face** | `upload.py`, called from `cli.py` | **Only from the interactive menu**, after a manual "Upload" choice, and it needs a token | We run `--auto-select`, which exports and exits before the menu. The run user has no HF token |
| Shell-out | `reproduce.py` | `sysctl -n machdep.cpu.brand_string` for the manifest | Harmless |
| Load Python from a file | `plugins.py` `load_plugin_class` | Only for a `file.py:Class` scorer named in the config | We write the config |
| `trust_remote_code` | `model.py` | Default off. On only with `--trust-remote-code` | We do not pass it |
| File writes | `model.py`, `cli.py`, `reproduce.py` | Output dir, checkpoint journal (`checkpoints_mlx/` under the working directory), HF cache | All inside the run user's home |

Other notes:

- The `upstream-watch.yml` workflow runs on GitHub, not on our machine. It
  opens issues in the author's repo when upstream Heretic changes.
- The default refusal score is a keyword count (`KeywordRate`). An outside
  study found that keyword counts disagree with a judge model by about 7
  points. We judge the final models with a classifier, not this count.
- `--smoke` changes the prompt slices to 32 train and 16 test prompts. It
  is a speed and plumbing check, not a quality result.

## How we run it

- A separate macOS user, `heretic`, with no admin rights, no `sudo` and no
  ssh access. Home directory mode 700.
- Its own venv made with `uv`, Python 3.12, heretic-mlx installed editable
  from the pinned commit with the `classifier` extra. Note: `uv pip install`
  resolved current versions, not the author's `uv.lock`. Installed:
  mlx 0.32.5, mlx-lm 0.32.2, torch 2.15.0.
- No Hugging Face token for that user. `HF_HUB_DISABLE_TELEMETRY=1`.
- Always `--auto-select`, so the interactive upload menu never shows.
- The base model is a local directory downloaded at a pinned revision.
- It runs as its own MLX process, outside Ollama, so it competes with the
  models that Ollama serves. We run it only when the machine is quiet, and
  never 22:00–05:00 ET.
