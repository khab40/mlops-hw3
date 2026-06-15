"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
MAX_ATTEMPTS = 3


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    gold_ok, gold_rows, gold_error = run_sql(question["db_id"], question["gold_sql"])
    started = time.monotonic()
    agent_error: str | None = None
    response: dict[str, Any] | None = None

    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                agent_url,
                json={
                    "question": question["question"],
                    "db": question["db_id"],
                    "tags": {
                        "phase": "5",
                        "run": "baseline-eval",
                        "db": question["db_id"],
                    },
                },
            )
            resp.raise_for_status()
            response = resp.json()
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    history = response.get("history", []) if response else []
    attempts: list[dict[str, Any]] = []
    for entry in history:
        if entry.get("node") not in {"generate_sql", "revise"}:
            continue
        sql = str(entry.get("sql") or "").strip()
        pred_ok, pred_rows, pred_error = run_sql(question["db_id"], sql) if sql else (False, None, "empty SQL")
        correct = matches(gold_rows, pred_rows) if gold_ok and pred_ok else False
        attempts.append({
            "attempt": len(attempts) + 1,
            "zero_based_iteration": len(attempts),
            "node": entry.get("node"),
            "sql": sql,
            "execution_ok": pred_ok,
            "execution_error": pred_error,
            "correct": correct,
        })

    # If the server returned SQL but history was missing, still score the served answer.
    if not attempts and response and response.get("sql"):
        sql = str(response["sql"]).strip()
        pred_ok, pred_rows, pred_error = run_sql(question["db_id"], sql)
        attempts.append({
            "attempt": 1,
            "zero_based_iteration": 0,
            "node": "final",
            "sql": sql,
            "execution_ok": pred_ok,
            "execution_error": pred_error,
            "correct": matches(gold_rows, pred_rows) if gold_ok and pred_ok else False,
        })

    final_attempt = attempts[-1] if attempts else None
    return {
        "db_id": question["db_id"],
        "question": question["question"],
        "gold_sql": question["gold_sql"],
        "gold_execution_ok": gold_ok,
        "gold_execution_error": gold_error,
        "agent_ok": bool(response and response.get("ok")),
        "agent_error": agent_error or (response.get("error") if response else None),
        "agent_iterations": response.get("iterations") if response else 0,
        "latency_seconds": time.monotonic() - started,
        "final_sql": response.get("sql") if response else "",
        "final_correct": bool(final_attempt and final_attempt["correct"]),
        "attempts": attempts,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("final_correct"))
    with_revise = sum(1 for r in results if int(r.get("agent_iterations") or 0) > 1)
    agent_errors = sum(1 for r in results if r.get("agent_error"))
    execution_errors = sum(
        1
        for r in results
        if r.get("attempts") and r["attempts"][-1].get("execution_ok") is False
    )

    def carried_correct(result: dict, attempt_number: int) -> bool:
        attempts = result.get("attempts") or []
        if not attempts:
            return False
        idx = min(attempt_number - 1, len(attempts) - 1)
        return bool(attempts[idx].get("correct"))

    pass_by_attempt: dict[str, dict[str, float | int]] = {}
    pass_by_zero_based_iteration: dict[str, dict[str, float | int]] = {}
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        n_correct = sum(1 for r in results if carried_correct(r, attempt_number))
        rate = (n_correct / total) if total else 0.0
        pass_by_attempt[str(attempt_number)] = {"correct": n_correct, "total": total, "pass_rate": rate}
        pass_by_zero_based_iteration[str(attempt_number - 1)] = {
            "correct": n_correct,
            "total": total,
            "pass_rate": rate,
        }

    return {
        "total": total,
        "correct": correct,
        "overall_pass_rate": (correct / total) if total else 0.0,
        "agent_errors": agent_errors,
        "final_execution_errors": execution_errors,
        "with_revise": with_revise,
        "max_attempts": MAX_ATTEMPTS,
        "pass_by_attempt": pass_by_attempt,
        "pass_by_zero_based_iteration": pass_by_zero_based_iteration,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
