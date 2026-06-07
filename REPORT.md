# MLOps HW3 Report

## Serving Configuration

The final vLLM server runs `Qwen/Qwen3-30B-A3B-Instruct-2507` at `http://localhost:8000` using `scripts/start_vllm.sh`.

| Flag | Value | Justification |
|---|---:|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Required assignment model for final serving and evaluation. |
| `--host` | `0.0.0.0` | Allows the VM service to be reached from the laptop through SSH port forwarding. |
| `--port` | `8000` | Matches the assignment endpoint and Prometheus scrape target. |
| `--max-model-len` | `8192` | Avoids the default very large context, which overcommitted KV cache memory; still fits the tuned prompts. |
| `--gpu-memory-utilization` | `0.90` | Gives vLLM most of the H100 while leaving CUDA/runtime headroom. |

The final agent serving configuration uses the same vLLM endpoint, but the load path is optimized with:

| Setting | Value | Justification |
|---|---:|---|
| `AGENT_FAST_PATH` | `true` | Uses `generate_sql -> execute` for serving instead of the slower verify/revise loop. |
| `AGENT_MAX_WORKERS` | `96` | Runs blocking graph work in an explicit thread pool instead of FastAPI's implicit sync pool. |
| `AGENT_MAX_INFLIGHT` | `96` | Bounds admitted agent work so requests cannot pile up indefinitely. |
| `AGENT_QUEUE_TIMEOUT_SECONDS` | `0.25` | Fails overload quickly instead of letting requests wait until client timeout. |
| `AGENT_MAX_TOKENS` | `256` | SQL answers should be short; this caps decode tail latency. |
| `AGENT_SCHEMA_MAX_CHARS` | `12000` | Trims very large schemas to keep prompts under the serving context budget. |
| `AGENT_SQL_MAX_ROWS` | `100` | Prevents large accidental result sets from becoming huge HTTP responses. |

Manual vLLM verification is saved in `screenshots/vllm_manual_query.png`.

## Baseline Eval

Eval runner: `evals/run_eval.py`  
Baseline artifact: `results/eval_baseline.json`  
Post-tuning artifact: `results/eval_after_tuning.json`

The eval compares predicted SQL to gold SQL by executing both against the same BIRD SQLite database and comparing canonicalized row sets.

| Metric | Baseline | After tuning |
|---|---:|---:|
| Eval questions | 30 | 30 |
| Correct final answers | 10 | 10 |
| Overall execution accuracy | 33.3% | 33.3% |
| Agent errors | 0 | 0 |
| Final SQL execution errors | 0 | 0 |
| Questions triggering revise | 11 | 0 |
| Wall-clock eval time | 57.9s | 28.1s |

Baseline per-iteration pass rate:

| Attempt | Correct | Pass rate |
|---:|---:|---:|
| 1 / zero-based iter 0 | 10 / 30 | 33.3% |
| 2 / zero-based iter 1 | 10 / 30 | 33.3% |
| 3 / zero-based iter 2 | 10 / 30 | 33.3% |

The baseline verify/revise loop triggered on 11 questions, but none of those revisions changed an initially wrong answer into a correct one. It also did not break any initially correct answer. This matters for Phase 6: disabling revise on the fast load path removed latency without reducing measured execution accuracy on this eval set.

## Hitting The SLO

Target: p95 end-to-end agent latency under 5 seconds at 10 scheduled RPS over 5 minutes.

Load artifacts:

- `results/load_test_rps10_agent_diag.json`
- `results/load_test_rps10_async_threadpool.json`
- `results/load_test_rps10_fast_path.json`

Grafana artifacts:

- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

Load-test summary:

| Run | Scheduled | Total | OK | Timeouts | HTTP errors | Client errors | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline with diagnostics | 3000 | 2593 | 1138 | 976 | 67 | 412 | 48.2s | 92.4s | 105.1s |
| Async graph threadpool | 3000 | 2997 | 2769 | 17 | 0 | 211 | 12.9s | 41.9s | 50.8s |
| Final fast path | 3000 | 3000 | 2997 | 3 | 0 | 0 | 0.88s | 1.72s | 3.23s |

Pressure summary from Grafana/Prometheus:

| Run | Peak `/answer` in-flight | Peak graph in-flight | Peak executor queue | vLLM waiting | vLLM running peak |
|---|---:|---:|---:|---:|---:|
| Baseline with diagnostics | 928 | 40 | n/a | 0 | 35 |
| Async graph threadpool | 396 | 128 | 267 | 0 | 118 |
| Final fast path | 20 | 20 | 0 | 0 | 21 |

Iteration log:

1. saw `latency_p50=48.2s`, `latency_p95=92.4s`, `latency_p99=105.1s`, peak `/answer` in-flight `928`, graph in-flight `40`, and vLLM waiting `0` -> hypothesized the serving bottleneck was FastAPI/agent request backlog from unbounded client concurrency plus multi-step sequential LLM calls, not GPU/vLLM saturation -> changed Grafana/Prometheus instrumentation to expose `agent_http_requests_in_progress`, graph duration, node p95, outcomes, and agent-vs-vLLM latency -> result was Grafana made the root cause visible: requests piled up before/inside the agent while vLLM still had no scheduler queue.

2. saw graph in-flight capped at `40`, peak `/answer` in-flight `928`, `ok=1138`, `timeouts=976`, and `latency_p95=92.4s` -> hypothesized FastAPI's implicit sync endpoint threadpool was the first concurrency ceiling before vLLM -> changed `/answer` to an async endpoint backed by an explicit `AGENT_MAX_WORKERS=128` graph `ThreadPoolExecutor` and added executor queue metrics -> result was `ok=2769`, `timeouts=17`, HTTP 500s `0`, `latency_p50=12.9s`, `latency_p95=41.9s`, peak `/answer` in-flight `396`, graph in-flight `128`, executor queued `267`, and vLLM waiting still `0`.

3. saw the async-threadpool run still missed the SLO with `latency_p50=12.9s`, `latency_p95=41.9s`, `latency_p99=50.8s`, `ok=2769`, `timeouts=17`, peak `/answer` in-flight `396`, graph in-flight `128`, executor queued `267`, and vLLM waiting `0` -> hypothesized the remaining tail was sequential agent work plus occasional oversized schema/output payloads, not GPU queueing -> changed serving to `AGENT_FAST_PATH=true`, skipped verify/revise on the load path, capped generation at `AGENT_MAX_TOKENS=256`, trimmed schema context to `AGENT_SCHEMA_MAX_CHARS=12000`, bounded admission with `AGENT_MAX_INFLIGHT=96` and `AGENT_QUEUE_TIMEOUT_SECONDS=0.25`, and capped SQL response previews with `AGENT_SQL_MAX_ROWS=100` -> result was `ok=2997`, `timeouts=3`, `client_errors=0`, `latency_p50=0.88s`, `latency_p95=1.72s`, `latency_p99=3.23s`, peak `/answer` in-flight `20`, graph in-flight `20`, executor queued `0`, vLLM waiting `0`, meeting the p95 < 5s target.

The final run meets the latency SLO. The load driver scheduled all 3000 requests over the 300-second window; driver-reported achieved RPS was 8.46 because the wall clock includes tail/drain time, but the request scheduler did issue the target 10 RPS during the load window.

## Agent Value

The verify/revise loop did not add measurable quality on this eval set. The evidence is the per-iteration pass rate: attempt 1 was 10/30, attempt 2 was 10/30, and attempt 3 was 10/30. The loop did provide observability value because Langfuse traces clearly showed the `generate_sql -> execute -> verify -> revise` waterfall and explained why multi-step requests were slow, but the actual revision policy did not recover wrong SQL. For the final serving SLO, the right tradeoff was to keep the full graph available but use the fast `generate_sql -> execute` path under load.

## With More Time

I would improve quality without giving up the serving SLO by making the fast path smarter rather than restoring unconditional verify/revise:

- Add schema retrieval instead of crude schema trimming: rank tables/columns using question terms plus foreign-key neighborhoods, then include only the top schema slice.
- Add a cheap deterministic SQL guard before execution: reject `SELECT *`, missing `LIMIT` on broad queries, and joins that ignore available foreign keys.
- Run verifier only on suspicious cases: execution error, zero rows for count/list questions, too many rows, or SQL touching no question-relevant tables.
- Make revise produce an explicit alternative plan before SQL, then compare the new SQL to the previous SQL to avoid repeated identical revisions.
- Add a small offline prompt-tuning set from the 20 consistently wrong eval questions, especially `formula_1`, `thrombosis_prediction`, and `toxicology`.

## Final Deliverables

| File | Status |
|---|---|
| `REPORT.md` | Complete final writeup |
| `infra/grafana/provisioning/dashboards/serving.json` | Dashboard with vLLM and agent panels |
| `agent/graph.py`, `agent/prompts.py` | Implemented agent |
| `evals/run_eval.py` | Eval runner |
| `results/eval_baseline.json` | Baseline eval |
| `results/eval_after_tuning.json` | Post-tuning eval |
| `screenshots/vllm_manual_query.png` | Phase 1 screenshot |
| `screenshots/grafana_serving.png` | Phase 2 screenshot |
| `screenshots/langfuse_trace.png` | Phase 4 trace screenshot |
| `screenshots/langfuse_tags.png` | Phase 4 tags screenshot |
| `screenshots/grafana_eval_run.png` | Phase 5 screenshot |
| `screenshots/grafana_before.png`, `screenshots/grafana_after.png` | Phase 6 before/after screenshots |
