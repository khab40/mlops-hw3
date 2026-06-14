"""Prometheus metrics for the FastAPI agent and LangGraph nodes."""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

from prometheus_client import Counter, Gauge, Histogram

F = TypeVar("F", bound=Callable)

LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    90.0,
    120.0,
    180.0,
)

AGENT_HEALTH_UP = Gauge(
    "agent_health_up",
    "Agent process health signal exposed by /health.",
)

HTTP_IN_PROGRESS = Gauge(
    "agent_http_requests_in_progress",
    "Agent HTTP requests currently being handled.",
    ["path"],
)
HTTP_REQUESTS_TOTAL = Counter(
    "agent_http_requests_total",
    "Agent HTTP requests completed.",
    ["path", "method", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "agent_http_request_duration_seconds",
    "Agent HTTP request latency including FastAPI dispatch and graph execution.",
    ["path", "method", "status"],
    buckets=LATENCY_BUCKETS,
)

GRAPH_IN_PROGRESS = Gauge(
    "agent_graph_invocations_in_progress",
    "LangGraph invocations currently running.",
)
GRAPH_EXECUTOR_MAX_WORKERS = Gauge(
    "agent_graph_executor_max_workers",
    "Maximum worker threads configured for graph invocation.",
)
GRAPH_EXECUTOR_QUEUE_DEPTH = Gauge(
    "agent_graph_executor_queue_depth",
    "Approximate number of graph invocations waiting in the executor queue.",
)
GRAPH_DURATION = Histogram(
    "agent_graph_invocation_duration_seconds",
    "End-to-end LangGraph invocation duration.",
    ["status"],
    buckets=LATENCY_BUCKETS,
)
GRAPH_ERRORS_TOTAL = Counter(
    "agent_graph_errors_total",
    "LangGraph invocation failures by exception class.",
    ["error_type"],
)

NODE_DURATION = Histogram(
    "agent_node_duration_seconds",
    "LangGraph node duration by node and status.",
    ["node", "status"],
    buckets=LATENCY_BUCKETS,
)
NODE_ERRORS_TOTAL = Counter(
    "agent_node_errors_total",
    "LangGraph node failures by node and exception class.",
    ["node", "error_type"],
)

ANSWER_ITERATIONS = Histogram(
    "agent_answer_iterations",
    "Generate/revise iteration count per completed answer request.",
    buckets=(1, 2, 3, 4, 5, 10),
)
ANSWER_OUTCOMES_TOTAL = Counter(
    "agent_answer_outcomes_total",
    "Completed answer outcomes.",
    ["outcome"],
)
ANSWER_CACHE_EVENTS_TOTAL = Counter(
    "agent_answer_cache_events_total",
    "Answer cache events.",
    ["event"],
)
ANSWER_CACHE_SIZE = Gauge(
    "agent_answer_cache_size",
    "Current number of entries in the in-process answer cache.",
)


def normalize_path(path: str) -> str:
    if path in {"/answer", "/health", "/metrics"}:
        return path
    return "other"


@contextmanager
def timed_node(node: str):
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        NODE_DURATION.labels(node=node, status="error").observe(time.perf_counter() - start)
        NODE_ERRORS_TOTAL.labels(node=node, error_type=type(exc).__name__).inc()
        raise
    else:
        NODE_DURATION.labels(node=node, status="ok").observe(time.perf_counter() - start)
