"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from prometheus_client import make_asgi_app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.requests import Request

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402
from agent.metrics import (  # noqa: E402
    AGENT_HEALTH_UP,
    ANSWER_ITERATIONS,
    ANSWER_OUTCOMES_TOTAL,
    GRAPH_DURATION,
    GRAPH_ERRORS_TOTAL,
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


app = FastAPI()
app.mount("/metrics", make_asgi_app())
AGENT_HEALTH_UP.set(1)


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


@app.get("/health")
def health() -> dict[str, str]:
    AGENT_HEALTH_UP.set(1)
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    trace_tags = ["agent", *[f"{key}:{value}" for key, value in sorted(req.tags.items())]]
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": req.tags,
        "tags": trace_tags,
        "run_name": "text-to-sql-agent",
    }
    start = time.perf_counter()
    GRAPH_IN_PROGRESS.inc()
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        GRAPH_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        GRAPH_DURATION.labels(status="error").observe(time.perf_counter() - start)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        GRAPH_IN_PROGRESS.dec()
    GRAPH_DURATION.labels(status="ok").observe(time.perf_counter() - start)

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
