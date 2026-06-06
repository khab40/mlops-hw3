# Architecture

This repository is a self-hosted text-to-SQL MLOps assignment. It combines a vLLM OpenAI-compatible endpoint, a LangGraph agent, SQLite BIRD databases, Prometheus and Grafana serving observability, Langfuse tracing, offline evals, and a load driver for SLO testing.

Some files are intentionally scaffolded for the assignment. In particular, `agent/graph.py`, `agent/prompts.py`, and `evals/run_eval.py` contain implementation points for later phases.

The diagrams below intentionally use a conservative Mermaid subset for VS Code preview compatibility: only `flowchart`, quoted node labels, short edge labels, and no HTML line breaks.

## Main High-Level Diagram

```mermaid
flowchart LR
    Analyst["Analyst or eval client"] -->|POST answer| AgentAPI["FastAPI agent server"]
    AgentAPI --> Graph["LangGraph workflow"]
    Graph -->|schema lookup| Schema["Schema renderer"]
    Schema --> Bird["BIRD SQLite files"]
    Graph -->|chat completions| VLLM["vLLM API server"]
    Graph -->|read only SQL| Executor["SQL executor"]
    Executor --> Bird
    Graph -->|SQL rows history| AgentAPI

    VLLM -->|metrics endpoint| Prometheus["Prometheus"]
    Prometheus --> Grafana["Grafana dashboard"]

    AgentAPI -->|callbacks| Langfuse["Langfuse"]
    Langfuse --> Postgres["Postgres"]
    Langfuse --> ClickHouse["ClickHouse"]
    Langfuse --> Redis["Redis"]
    Langfuse --> MinIO["MinIO"]

    EvalRunner["Offline eval runner"] --> AgentAPI
    LoadDriver["Load driver"] --> AgentAPI
```

## Use Case: Load BIRD Data

`scripts/load_data.py` downloads the BIRD dev archive, extracts the databases, surfaces each SQLite file under `data/bird/`, and writes the curated eval and load-test inputs.

```mermaid
flowchart TD
    Start["Run load data script"] --> Download["Download BIRD dev archive"]
    Download --> ExtractOuter["Extract outer archive"]
    ExtractOuter --> ExtractInner["Extract database archive"]
    ExtractInner --> Consolidate["Copy SQLite files"]
    ExtractOuter --> DevJson["Read dev dataset"]
    DevJson --> SampleEval["Write eval set"]
    DevJson --> SamplePerf["Write load test pool"]
    Consolidate --> Ready["Data ready"]
    SampleEval --> Ready
    SamplePerf --> Ready
```

## Use Case: Serve the Model

`scripts/start_vllm.sh` starts vLLM on port `8000` with `Qwen/Qwen3-30B-A3B-Instruct-2507`. The agent talks to it through `langchain-openai` using `VLLM_BASE_URL`, `VLLM_MODEL`, and `OPENAI_API_KEY`.

```mermaid
flowchart LR
    Operator["Operator"] --> Script["start vLLM script"]
    Script --> VLLM["vLLM OpenAI API"]
    Agent["LangGraph agent"] -->|chat request| VLLM
    VLLM -->|model response| Agent
    Prometheus["Prometheus"] -->|scrape metrics| VLLM
```

## Use Case: Answer a Text-to-SQL Request

The runtime path starts at `POST /answer`. The server invokes the LangGraph workflow and returns the final SQL, rows, iteration count, status, and history.

```mermaid
flowchart TD
    Client["Client"] -->|question db tags| API["FastAPI answer endpoint"]
    API --> State["Create agent state"]
    State --> Schema["Render database schema"]
    Schema --> Generate["Generate SQL with vLLM"]
    Generate --> Execute["Execute SQL in SQLite"]
    Execute --> Verify["Verify result with vLLM"]
    Verify --> Response["Return SQL rows iterations history"]
    Response --> Client
```

## Use Case: Verify and Revise SQL

The graph is designed as a self-correction loop. `generate_sql` and each `revise` increment `iteration`; the router stops when verification succeeds or the max iteration cap is reached.

```mermaid
flowchart TD
    Start["Start"] --> AttachSchema["Attach schema"]
    AttachSchema --> GenerateSQL["Generate SQL"]
    GenerateSQL --> Execute["Execute SQL"]
    Execute --> Verify["Verify answer"]
    Verify -->|ok| End["End"]
    Verify -->|max iterations| End
    Verify -->|not ok| Revise["Revise SQL"]
    Revise --> Execute
```

## Use Case: Observe vLLM Serving Health

Prometheus scrapes vLLM's `/metrics` endpoint through `host.docker.internal:8000`. Grafana loads the Prometheus datasource and starter serving dashboard from `infra/grafana/provisioning`.

```mermaid
flowchart LR
    Requests["Agent eval load traffic"] --> VLLM["vLLM"]
    VLLM --> Metrics["Metrics endpoint"]
    Metrics --> Prometheus["Prometheus"]
    Prometheus --> Grafana["Grafana dashboard"]
    Grafana --> Operator["Operator reads serving health"]
```

## Use Case: Trace Agent Runs

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present, `agent/server.py` attaches the Langfuse callback handler to each graph invocation. Request `tags` are passed as metadata.

```mermaid
flowchart LR
    Client["Client"] -->|answer request with tags| API["FastAPI agent"]
    API -->|callbacks metadata| Graph["LangGraph nodes"]
    Graph -->|spans prompts timings| Langfuse["Langfuse"]
    Langfuse --> Store["Trace storage"]
    API -->|response payload| Client
    Client -->|inspect traces| Langfuse
```

## Use Case: Run Offline Evals

The eval runner calls the agent for every curated question, executes both predicted and gold SQL against the same SQLite database, canonicalizes row sets, and writes a JSON report under `results/`.

```mermaid
flowchart TD
    EvalSet["Eval set"] --> Runner["Eval runner"]
    Runner --> Agent["Agent answer endpoint"]
    Agent --> PredSQL["Predicted SQL"]
    Runner --> GoldExec["Run gold SQL"]
    PredSQL --> PredExec["Run predicted SQL"]
    GoldExec --> Compare["Compare row sets"]
    PredExec --> Compare
    Compare --> Summary["Accuracy summary"]
    Summary --> Results["Results JSON"]
```

## Use Case: Run Load and SLO Tests

The load driver samples questions from `load_test/perf_pool.jsonl`, sends them to the agent endpoint at a requested RPS, and writes latency and status summaries to `results/load_test.json`. The same traffic exercises vLLM metrics and Langfuse traces.

```mermaid
flowchart LR
    Pool["Load test question pool"] --> Driver["Load driver"]
    Driver -->|target RPS duration| Agent["Agent answer endpoint"]
    Agent --> VLLM["vLLM"]
    Agent --> DB["SQLite"]
    VLLM --> Prometheus["Prometheus metrics"]
    Prometheus --> Grafana["Grafana SLO dashboard"]
    Agent --> Langfuse["Langfuse traces"]
    Driver --> Results["Load test results"]
```
