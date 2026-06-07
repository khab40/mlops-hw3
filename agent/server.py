"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv
from prometheus_client import make_asgi_app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.requests import Request

load_dotenv()

from agent.graph import AgentState, graph, run_fast_path  # noqa: E402
from agent.metrics import (  # noqa: E402
    AGENT_HEALTH_UP,
    ANSWER_ITERATIONS,
    ANSWER_OUTCOMES_TOTAL,
    GRAPH_DURATION,
    GRAPH_ERRORS_TOTAL,
    GRAPH_EXECUTOR_MAX_WORKERS,
    GRAPH_EXECUTOR_QUEUE_DEPTH,
    GRAPH_IN_PROGRESS,
    HTTP_IN_PROGRESS,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    normalize_path,
)

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()

AGENT_MAX_WORKERS = int(os.environ.get("AGENT_MAX_WORKERS", "128"))
AGENT_FAST_PATH = os.environ.get("AGENT_FAST_PATH", "false").lower() in {"1", "true", "yes"}
AGENT_MAX_INFLIGHT = int(os.environ.get("AGENT_MAX_INFLIGHT", str(AGENT_MAX_WORKERS)))
AGENT_QUEUE_TIMEOUT_SECONDS = float(os.environ.get("AGENT_QUEUE_TIMEOUT_SECONDS", "0.25"))
_graph_executor = ThreadPoolExecutor(
    max_workers=AGENT_MAX_WORKERS,
    thread_name_prefix="agent-graph",
)
_admission = asyncio.Semaphore(AGENT_MAX_INFLIGHT)
atexit.register(_graph_executor.shutdown, wait=False, cancel_futures=True)

app = FastAPI()
app.mount("/metrics", make_asgi_app())
AGENT_HEALTH_UP.set(1)
GRAPH_EXECUTOR_MAX_WORKERS.set(AGENT_MAX_WORKERS)


def _executor_queue_depth() -> int:
    queue = getattr(_graph_executor, "_work_queue", None)
    if queue is None:
        return 0
    try:
        return int(queue.qsize())
    except NotImplementedError:
        return 0


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    path = normalize_path(request.url.path)
    if path == "/metrics":
        return await call_next(request)

    method = request.method
    start = time.perf_counter()
    status = "500"
    HTTP_IN_PROGRESS.labels(path=path).inc()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        HTTP_IN_PROGRESS.labels(path=path).dec()
        HTTP_REQUESTS_TOTAL.labels(path=path, method=method, status=status).inc()
        HTTP_REQUEST_DURATION.labels(path=path, method=method, status=status).observe(elapsed)


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


def _run_graph(state: AgentState, config: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    GRAPH_IN_PROGRESS.inc()
    try:
        final = run_fast_path(state, config) if AGENT_FAST_PATH else graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        GRAPH_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        GRAPH_DURATION.labels(status="error").observe(time.perf_counter() - start)
        raise
    finally:
        GRAPH_IN_PROGRESS.dec()
    GRAPH_DURATION.labels(status="ok").observe(time.perf_counter() - start)
    return final


@app.get("/health")
def health() -> dict[str, str]:
    AGENT_HEALTH_UP.set(1)
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    trace_tags = ["agent", *[f"{key}:{value}" for key, value in sorted(req.tags.items())]]
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": req.tags,
        "tags": trace_tags,
        "run_name": "text-to-sql-agent",
    }
    admitted = False
    try:
        await asyncio.wait_for(_admission.acquire(), timeout=AGENT_QUEUE_TIMEOUT_SECONDS)
        admitted = True
        GRAPH_EXECUTOR_QUEUE_DEPTH.set(_executor_queue_depth())
        loop = asyncio.get_running_loop()
        final = await loop.run_in_executor(_graph_executor, _run_graph, state, config)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="agent overloaded")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        if admitted:
            _admission.release()
        GRAPH_EXECUTOR_QUEUE_DEPTH.set(_executor_queue_depth())

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")
    ANSWER_ITERATIONS.observe(iteration)

    if execution is None:
        ANSWER_OUTCOMES_TOTAL.labels(outcome="no_execution").inc()
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        ANSWER_OUTCOMES_TOTAL.labels(outcome="execution_error").inc()
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    ANSWER_OUTCOMES_TOTAL.labels(outcome="ok").inc()
    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
