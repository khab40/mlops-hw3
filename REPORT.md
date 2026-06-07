# MLOps HW3 Report

## Setup And Serving

Phase 0 setup was completed on the VM with the five assignment ports forwarded to the laptop: Grafana `3000`, Prometheus `9090`, Langfuse `3001`, vLLM `8000`, and the agent server `8001`. The three required UIs were reachable from the laptop browser. `.env` was created from `.env.example` and later populated with Langfuse keys and serving settings. BIRD data was loaded under `data/bird/` with the SQLite databases and dev metadata used by the eval runner.

The vLLM endpoint served `Qwen/Qwen3-30B-A3B-Instruct-2507` at `http://localhost:8000`. Manual checks against questions from `evals/eval_set.jsonl` returned SQL-shaped answers, and the screenshot is saved as `screenshots/vllm_manual_query.png`.

| Flag | Value | Justification |
|---|---:|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Required assignment model for final serving and eval results. |
| `--host` | `0.0.0.0` | Makes the VM service reachable through SSH port forwarding. |
| `--port` | `8000` | Matches the assignment endpoint and Prometheus scrape target. |
| `--max-model-len` | `8192` | Avoids the default very large context that overcommitted KV cache on one H100, while still fitting the expected prompts and short SQL outputs. |
| `--gpu-memory-utilization` | `0.90` | Uses most of the H100 memory for serving while leaving runtime headroom. |

The final load-serving agent settings were:

| Setting | Value | Why |
|---|---:|---|
| `AGENT_FAST_PATH` | `true` | Uses `generate_sql -> execute` for load serving instead of unconditional verify/revise. |
| `AGENT_MAX_WORKERS` | `96` | Runs blocking graph work in an explicit pool instead of FastAPI's implicit sync pool. |
| `AGENT_MAX_INFLIGHT` | `96` | Bounds admitted work so requests do not queue indefinitely. |
| `AGENT_QUEUE_TIMEOUT_SECONDS` | `0.25` | Fails overload quickly instead of waiting until client timeout. |
| `AGENT_MAX_TOKENS` | `256` | SQL answers should be short; this caps decode tail latency. |
| `AGENT_SCHEMA_MAX_CHARS` | `12000` | Keeps large schemas from inflating prompts and latency. |
| `AGENT_SQL_MAX_ROWS` | `100` | Prevents accidental large result previews from becoming huge HTTP responses. |

## Observability And Agent

The Grafana dashboard is committed at `infra/grafana/provisioning/dashboards/serving.json`. It covers request volume, latency percentiles, lifecycle latency, throughput, scheduler queue, KV cache headroom, prefix cache hit ratio, vLLM running/waiting requests, and agent-side diagnostics. Prometheus scraped vLLM successfully, Grafana reported healthy, and the dashboard reacted during both request bursts and eval runs. The Phase 2 screenshot is `screenshots/grafana_serving.png`.

The agent is implemented in `agent/graph.py` and `agent/prompts.py`. The graph follows `generate_sql -> execute -> verify`, routes to `revise` when `verify.ok=false`, and caps the loop at three total generate/revise attempts. The Phase 3 smoke run used five eval questions against the real model: all five executed without runtime error, four were accepted by the verifier, and one question triggered the revise path. In that revise case the verifier rejected a no-data answer, but Qwen repeated the same SQL until the iteration cap stopped the loop.

Langfuse tracing works for the full graph after propagating the LangChain runnable config into nested LLM calls. The trace tags include `agent`, `phase:4`, `run:phase4-smoke`, `question_index:N`, and the VM tag. A selected trace showed the expected `generate_sql`, `execute`, `verify`, `revise`, and repeated execute/verify waterfall with nested model spans. The trace screenshot is `screenshots/langfuse_trace.png`, and the tagged trace list is `screenshots/langfuse_tags.png`.

## Baseline Eval

The eval runner is `evals/run_eval.py`. It calls the agent, extracts each generated/revised SQL attempt, executes predicted and gold SQL against the same BIRD SQLite database, canonicalizes row sets, and scores execution accuracy. Baseline results are saved in `results/eval_baseline.json`; Grafana during the run is saved in `screenshots/grafana_eval_run.png`.

| Metric | Result |
|---|---:|
| Eval questions | 30 |
| Correct final answers | 10 |
| Overall execution accuracy | 33.3% |
| Agent errors | 0 |
| Final SQL execution errors | 0 |
| Questions triggering revise | 11 |
| Wall-clock eval time | 57.9s |

Per-iteration pass rate:

| Attempt | Correct | Pass rate |
|---:|---:|---:|
| 1 / zero-based iter 0 | 10 / 30 | 33.3% |
| 2 / zero-based iter 1 | 10 / 30 | 33.3% |
| 3 / zero-based iter 2 | 10 / 30 | 33.3% |

The baseline loop did not improve accuracy. All 11 questions that triggered `revise` were incorrect on the first attempt and remained incorrect after revision. No initially correct answers were broken. The strongest datasets were `student_club` at `3/4`, `financial` at `2/3`, and `superhero` at `2/3`; the weakest were `formula_1`, `thrombosis_prediction`, and `toxicology`, all at `0` correct.

## Hitting The SLO

Target SLO: p95 end-to-end agent latency under 5 seconds at 10+ scheduled RPS over a 5-minute window.

| Run | Scheduled | Total | OK | Timeouts | HTTP errors | Client errors | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline diagnostics | 3000 | 2593 | 1138 | 976 | 67 | 412 | 48.2s | 92.4s | 105.1s |
| Async graph threadpool | 3000 | 2997 | 2769 | 17 | 0 | 211 | 12.9s | 41.9s | 50.8s |
| Final fast path | 3000 | 3000 | 2997 | 3 | 0 | 0 | 0.88s | 1.72s | 3.23s |

Pressure observed in Grafana/Prometheus:

| Run | Peak `/answer` in-flight | Peak graph in-flight | Peak executor queue | vLLM waiting | vLLM running peak |
|---|---:|---:|---:|---:|---:|
| Baseline diagnostics | 928 | 40 | n/a | 0 | 35 |
| Async graph threadpool | 396 | 128 | 267 | 0 | 118 |
| Final fast path | 20 | 20 | 0 | 0 | 21 |

Iteration log:

1. saw `latency_p50=48.2s`, `latency_p95=92.4s`, `latency_p99=105.1s`, peak `/answer` in-flight `928`, graph in-flight `40`, and vLLM waiting `0` -> hypothesized the bottleneck was FastAPI/agent backlog from unbounded client concurrency plus multi-step sequential LLM calls, not GPU saturation -> changed Grafana/Prometheus instrumentation to expose agent in-flight requests, graph duration, node p95, outcomes, and agent-vs-vLLM latency -> result was Grafana made the root cause visible: requests piled up before/inside the agent while vLLM had no scheduler queue.
2. saw graph in-flight capped at `40`, peak `/answer` in-flight `928`, `ok=1138`, `timeouts=976`, and `latency_p95=92.4s` -> hypothesized FastAPI's implicit sync endpoint threadpool was the first concurrency ceiling -> changed `/answer` to an async endpoint backed by `AGENT_MAX_WORKERS=128` and added executor queue metrics -> result was `ok=2769`, `timeouts=17`, HTTP 500s `0`, `latency_p50=12.9s`, `latency_p95=41.9s`, graph in-flight `128`, executor queued `267`, and vLLM waiting still `0`.
3. saw the async-threadpool run still missed the SLO with `latency_p95=41.9s`, executor queued `267`, and vLLM waiting `0` -> hypothesized the remaining tail was sequential agent work plus oversized schema/output payloads, not GPU queueing -> changed serving to `AGENT_FAST_PATH=true`, capped generation at `256` tokens, trimmed schema to `12000` chars, bounded admission with `AGENT_MAX_INFLIGHT=96` and `AGENT_QUEUE_TIMEOUT_SECONDS=0.25`, and capped SQL preview rows at `100` -> result was `ok=2997`, `timeouts=3`, `client_errors=0`, `latency_p50=0.88s`, `latency_p95=1.72s`, `latency_p99=3.23s`, executor queue `0`, and the p95 < 5s SLO was met.

The before/after Grafana evidence is saved as `screenshots/grafana_before.png` and `screenshots/grafana_after.png`. The final run scheduled all 3000 requests over the 300-second load window; driver-reported achieved RPS was 8.46 because wall-clock time included tail/drain, but the scheduler issued 10 RPS during the load window.

After tuning, I reran the eval set and saved it to `results/eval_after_tuning.json`.

| Metric | Baseline | After tuning |
|---|---:|---:|
| Correct final answers | 10 / 30 | 10 / 30 |
| Overall execution accuracy | 33.3% | 33.3% |
| Agent errors | 0 | 0 |
| Final SQL execution errors | 0 | 0 |
| Questions triggering revise | 11 | 0 |
| Wall-clock eval time | 57.9s | 28.1s |

Quality survived the serving changes on this eval set: there were 0 question-level regressions and 0 improvements. The same 10 questions remained correct.

## Agent Value

The verify/revise loop did not help execution accuracy on this eval set. The evidence is the per-iteration pass rate: attempt 1 was `10/30`, attempt 2 was `10/30`, and attempt 3 was `10/30`. The loop did help with observability and diagnosis because Langfuse exposed the full `generate_sql -> execute -> verify -> revise` waterfall and showed why multi-step requests were slower, but the revision policy did not recover wrong SQL. For the measured SLO, keeping the full graph available while using the fast `generate_sql -> execute` path under load was the right tradeoff.

## With More Time

- Replace crude schema trimming with schema retrieval: rank tables/columns by question terms plus foreign-key neighborhoods, then include only the top schema slice.
- Add deterministic SQL guards before execution: reject `SELECT *`, missing `LIMIT` on broad queries, and joins that ignore available foreign keys.
- Run verifier only on suspicious cases: execution error, zero rows for count/list questions, too many rows, or SQL touching no relevant tables.
- Make revise produce an explicit alternative plan before SQL, then reject identical revised SQL so the loop cannot waste attempts.
- Build a focused prompt-tuning set from the 20 consistently wrong eval questions, especially `formula_1`, `thrombosis_prediction`, and `toxicology`.

## Final Deliverables

| File | Status |
|---|---|
| `REPORT.md` | Complete writeup |
| `infra/grafana/provisioning/dashboards/serving.json` | Grafana dashboard with required panels |
| `agent/graph.py`, `agent/prompts.py` | Implemented agent |
| `evals/run_eval.py` | Eval runner |
| `results/eval_baseline.json` | Baseline eval results |
| `results/eval_after_tuning.json` | Post-tuning eval results |
| `screenshots/vllm_manual_query.png` | vLLM serving plus manual SQL query |
| `screenshots/grafana_serving.png` | Phase 2 dashboard reacting to load |
| `screenshots/langfuse_trace.png` | Langfuse verify/revise trace |
| `screenshots/langfuse_tags.png` | Langfuse trace list with metadata tags |
| `screenshots/grafana_eval_run.png` | Grafana during baseline eval |
| `screenshots/grafana_before.png`, `screenshots/grafana_after.png` | Phase 6 before/after dashboard screenshots |
