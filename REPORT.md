# MLOps HW3 Report

## Phase 0: Setup

The VM stack was brought up with vLLM, Prometheus, Grafana, Langfuse, and the agent reachable through forwarded ports (`8000`, `9090`, `3000`, `3001`, `8001`). The BIRD SQLite data was loaded under `data/bird/`, and the final measurements below use the required `Qwen/Qwen3-30B-A3B-Instruct-2507` backend on the H100.

## Phase 1: vLLM Serving

vLLM served the fixed model at `http://localhost:8000`; manual queries from `evals/eval_set.jsonl` returned SQL-shaped answers (`screenshots/vllm_manual_query.png`). The launch config is in `scripts/start_vllm.sh`.

| Flag | Value | Rationale |
|---|---:|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Required 30B MoE model for real pass-rate and SLO numbers. |
| `--host` / `--port` | `0.0.0.0` / `8000` | Exposes the OpenAI-compatible API through SSH forwarding for the agent. |
| `--max-model-len` | `8192` | The workload has 1.5-3K token schema prompts and short SQL outputs; this avoids wasting KV cache on unused long context. |
| `--gpu-memory-utilization` | `0.90` | Keeps most H100 memory available for model/KV cache while leaving runtime headroom. |

## Phase 2: Observability Dashboard

The Grafana dashboard is committed at `infra/grafana/provisioning/dashboards/serving.json` and evidence is in `screenshots/grafana_serving.png`. It covers the required serving-health views from Prometheus/vLLM metrics: request and token throughput, scheduler running/waiting queues, latency percentiles, time-to-first-token, request lifecycle latency, KV cache headroom, prefix-cache hit ratio, and preemption pressure. I also added agent panels for HTTP rate/status, in-flight pressure, graph duration, node latency, outcomes, and iterations so the dashboard answers both "is it slow?" and "is it vLLM or the agent?"

## Phase 3: Agent

The agent implementation is in `agent/graph.py` and `agent/prompts.py`. The graph runs `generate_sql -> execute -> verify`; failed verification or execution routes to `revise`, and the loop is bounded by an iteration cap. Prompts require SQLite-only `SELECT` statements, schema-grounded columns, exact values where available, and a real change on revision rather than returning the same SQL. The smoke and eval runs produced genuine revise cases, including fixes for invalid table patterns, wrong projection/order, missing `DISTINCT`, and wrong opaque BIRD columns.

## Phase 4: Langfuse Tracing

Langfuse captured agent traces after fixing the VM `.env` mismatch. The trace list was tagged with metadata such as `phase`, `run`, `question_index`, `db`, and `vm`; those same tags made it possible to separate Phase 4 smoke traces from Phase 5/6 runs. Evidence is saved in `screenshots/langfuse_trace.png` and `screenshots/langfuse_tags.png`; the selected trace shows the `generate_sql / verify / revise` waterfall for a verify-to-revise loop.

## Phase 5: Baseline Eval

`evals/run_eval.py` reads the 30-question eval set, calls `/answer`, executes final and gold SQL against the same SQLite DB, canonicalizes row sets, and records overall plus per-iteration execution accuracy. Results are saved in `results/eval_baseline.json`; Grafana evidence is `screenshots/grafana_eval_run.png`.

| Metric | Result |
|---|---:|
| Questions | 30 |
| Correct final answers | 17 |
| Overall execution accuracy | 56.7% |
| Agent errors / final SQL execution errors | 0 / 0 |
| Questions triggering revise | 12 |
| Wall-clock eval time | 63.0s |

Per-iteration pass rate:

| Attempt | Correct | Pass rate |
|---:|---:|---:|
| Iter 0 | 13 / 30 | 43.3% |
| Iter 1 | 16 / 30 | 53.3% |
| Iter 2 | 17 / 30 | 56.7% |

The loop earned its keep: stopping after the first SQL would have passed 13 questions, while the verify/revise loop passed 17. Four initially wrong answers became correct and no initially correct answer was broken.

## Phase 6: SLO Diagnosis

Target SLO: p95 end-to-end `/answer` latency under 5s at 10 scheduled RPS for 5 minutes. The load driver schedules 3000 requests over the 300s window; its achieved-RPS field includes drain time, so I use scheduled requests for the SLO load level and p95/quality for the verdict.

| Run | Scheduled | OK | Timeouts | HTTP errors | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline diagnostics | 3000 | 1138 | 976 | 67 | 48.2s | 92.4s | 105.1s |
| Async graph executor | 3000 | 2769 | 17 | 0 | 12.9s | 41.9s | 50.8s |
| Final fast path | 3000 | 2995 | 5 | 0 | 0.87s | 1.92s | 3.71s |

Iteration log:

1. Saw p95 `92.4s`, peak `/answer` in-flight near `928`, graph in-flight near `40`, and vLLM waiting queue near `0` -> hypothesized the bottleneck was agent backlog, not GPU scheduling -> added agent HTTP/graph/node metrics and Grafana panels -> result: the dashboard showed requests piling up before/inside the agent while vLLM was not queueing.
2. Saw graph in-flight capped around `40` and many timeout drains -> hypothesized FastAPI's sync endpoint threadpool was the first concurrency ceiling -> changed `/answer` to async with an explicit graph executor and queue metrics -> result: OK count rose to `2769` and HTTP errors disappeared, but p95 was still `41.9s`.
3. Saw p95 still far above SLO with vLLM waiting near `0` -> hypothesized sequential multi-step agent calls and prompt/response size were the remaining tail -> changed to a bounded fast path, capped generation at `256` tokens, trimmed schema context, bounded in-flight work to `96`, and added quick overload behavior -> result: p95 moved to `1.92s`, p99 to `3.71s`, with `2995/3000` OK.

The before/after evidence around the change that moved the metric is saved in `screenshots/grafana_before.png` and `screenshots/grafana_after.png`. The SLO was met on latency, but quality did not fully survive. The final tuned eval is in `results/eval_after_tuning.json`:

| Metric | Baseline full graph | After tuning |
|---|---:|---:|
| Correct final answers | 17 / 30 | 11 / 30 |
| Overall execution accuracy | 56.7% | 36.7% |
| Agent errors / final SQL execution errors | 0 / 0 | 2 / 2 |
| Questions triggering revise | 12 | 0 |
| Wall-clock eval time | 63.0s | 23.9s |

Verdict: the final config hits the high-load latency SLO by skipping the verifier/reviser path, but execution accuracy drops by 20 percentage points. The honest production answer would not be "always fast path"; it would route between quality mode and latency mode based on load and request importance.

## Phase 7: What I'd Do With More Time

- Use adaptive routing: full verify/revise for normal load, deterministic-only verifier under pressure, and fast path only when protecting the SLO.
- Use a smaller verifier model or deterministic verifier first, then rerun both Phase 5 eval and Phase 6 load to quantify the quality/latency tradeoff.
- Add request coalescing and SQL/result caching for repeated `(db, normalized question)` traffic.
- Improve schema retrieval with table/column ranking plus one-hop foreign-key neighborhoods instead of blunt character trimming.
- Build a focused prompt-tuning set from the remaining consistently wrong eval cases, especially `thrombosis_prediction`, `toxicology`, and harder `codebase_community` questions.

## Deliverables Checklist

| File | Status |
|---|---|
| `REPORT.md` | Complete writeup |
| `infra/grafana/provisioning/dashboards/serving.json` | Required Grafana dashboard |
| `agent/graph.py`, `agent/prompts.py` | Implemented agent |
| `evals/run_eval.py` | Eval runner |
| `results/eval_baseline.json` | Baseline eval results |
| `results/eval_after_tuning.json` | Post-tuning eval results |
| `screenshots/vllm_manual_query.png` | Phase 1 manual vLLM query |
| `screenshots/grafana_serving.png` | Phase 2 dashboard under load |
| `screenshots/langfuse_trace.png` | Phase 4 verify-to-revise trace |
| `screenshots/langfuse_tags.png` | Phase 4 metadata tags |
| `screenshots/grafana_eval_run.png` | Phase 5 eval dashboard |
| `screenshots/grafana_before.png`, `screenshots/grafana_after.png` | Phase 6 before/after tuning evidence |
