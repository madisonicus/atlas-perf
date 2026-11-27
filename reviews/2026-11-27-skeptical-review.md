# Skeptical review against the +50% goal (2026-11-27)

An outside review by a second model (Fable 5.2), asked to judge the work
against the goals, not to confirm it. It read this README, `bench/` and
`patches/`, and re-ran nothing. Fixes made from it: the production check is
now in the results log, the stale "+17%" line is corrected, and gains under
2% are treated as unproven.

## Verdict

The current path will not reach +50% single-stream decode. A draft round
costs 26–29 ms and yields ~2.5 tokens on prose (1.5 accepted + 1). With code
at ~167 tok/s, prose and chat each need ~141 tok/s, so ~2.8 accepted tokens
per round. Even a verify that cost nothing gives only ~+25% on prose. It
needs a draft accepted about twice as long, a relaxed accept rule, or a
different model.

## Weak points named

- Quality gate too small (20 tool calls, six tasks).
- Every bench is thinking off, temperature 0, 400 tokens, short prompt.
  Decode at 30k–140k context is not measured.
- "~1.5 accepted" is not explained: 79% at position 1 would give ~2.3 for
  an independent depth-4 draft. Either the depth controller caps near 2, or
  later positions collapse.
- The goal "owner latency within 15% of idle during batch jobs" has no
  measurement.

## Ranked areas

1. Measure the real workload (prefill, thinking, answer time) from request
   logs, then cut tokens (thinking budget, routing easy turns).
2. Route routine turns to Qwen3.6 35B-A3B (200 tok/s), after a blind replay
   of 100 real prompts.
3. Owner priority and preemption of batch streams (the only route to the 15%
   goal).
4. Copy-from-context drafting (prompt lookup) for edits, RAG, tool JSON.
5. Per-position acceptance logging; then rollout-trained draft or relaxed
   (typical) acceptance.
6. Compare plain decode with mlx-lm; attack the non-matmul 25% of a step.
7. Draft on the CPU while the GPU verifies (speculative).
8. 3-bit or mixed 3/4-bit MLP layers (likely under 5% net).

Recommendation: 1 and 3 first; the logging in 5 decides whether +50% decode
is reachable at all.
