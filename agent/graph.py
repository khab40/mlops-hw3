"""LangGraph agent: text-to-SQL with optional verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.metrics import timed_node
from agent.schema import db_path, retrieve_schema_for_question

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "3"))
MAX_REVISIONS = int(os.environ.get("AGENT_MAX_REVISIONS", "1"))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
VERIFIER_BASE_URL = os.environ.get("AGENT_VERIFIER_BASE_URL", VLLM_BASE_URL)
VERIFIER_MODEL = os.environ.get("AGENT_VERIFIER_MODEL", VLLM_MODEL)
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
VERIFIER_API_KEY = os.environ.get("AGENT_VERIFIER_API_KEY", LLM_API_KEY)
LLM_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "256"))
VERIFIER_MAX_TOKENS = int(os.environ.get("AGENT_VERIFIER_MAX_TOKENS", "96"))
CONDITIONAL_VERIFY = os.environ.get("AGENT_CONDITIONAL_VERIFY", "true").lower() in {"1", "true", "yes"}
SEMANTIC_LLM_VERIFY = os.environ.get("AGENT_SEMANTIC_LLM_VERIFY", "false").lower() in {"1", "true", "yes"}
VERIFY_SCHEMA_MAX_CHARS = int(os.environ.get("AGENT_VERIFY_SCHEMA_MAX_CHARS", "3000"))
VERIFY_MANY_ROWS_THRESHOLD = int(os.environ.get("AGENT_VERIFY_MANY_ROWS_THRESHOLD", "80"))
VALUE_HINT_MAX_COLUMNS = int(os.environ.get("AGENT_VALUE_HINT_MAX_COLUMNS", "12"))
VALUE_HINT_MAX_VALUES = int(os.environ.get("AGENT_VALUE_HINT_MAX_VALUES", "8"))

QUESTION_STOP_WORDS = {
    "about",
    "above",
    "after",
    "all",
    "among",
    "are",
    "before",
    "between",
    "calculate",
    "card",
    "cards",
    "could",
    "from",
    "give",
    "had",
    "has",
    "have",
    "highest",
    "how",
    "list",
    "lowest",
    "many",
    "more",
    "most",
    "name",
    "number",
    "please",
    "print",
    "prints",
    "provide",
    "received",
    "show",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "user",
    "users",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
    "was",
    "were",
    "year",
}

VALUE_HINT_SKIP_COLUMN_PARTS = (
    "about",
    "body",
    "email",
    "flavor",
    "image",
    "note",
    "profile",
    "text",
    "url",
    "website",
)

DOMAIN_ALIASES: dict[str, list[str]] = {
    "financial": [
        '"district"."A2" = district name.',
        '"district"."A3" = region.',
        '"district"."A4" = number of inhabitants.',
        '"district"."A11" = average salary.',
        '"district"."A12" = unemployment rate in 1995.',
        '"district"."A13" = unemployment rate in 1996.',
        '"district"."A14" = entrepreneurs per 1000 inhabitants.',
        '"district"."A15" = number of crimes committed in 1995.',
        '"district"."A16" = number of crimes committed in 1996.',
    ],
    "california_schools": [
        '"schools"."NCESDist" = NCES district identifier.',
        '"schools"."NCESSchool" = NCES school identifier.',
        '"schools"."CDSCode" = California school/district code used for joins.',
        '"frpm"."Enrollment (Ages 5-17)" = enrollment count for ages 5-17.',
        '"satscores"."NumGE1500" / "satscores"."NumTstTakr" = excellence rate.',
        '"satscores"."AvgScrRead" = average reading score.',
        'Complete school address columns should be "Street", "City", "State", "Zip".',
    ],
    "toxicology": [
        '"molecule"."label" = \'+\' means carcinogenic.',
        '"molecule"."label" = \'-\' means non carcinogenic.',
        '"atom"."element" stores lowercase chemical symbols such as \'cl\' for Chlorine and \'ca\' for Calcium.',
    ],
    "card_games": [
        '"cards"."id" is the printed card identifier.',
        '"cards"."name" is the card name.',
        '"cards"."rarity" uses lowercase values such as \'mythic\'.',
        '"legalities"."format" uses lowercase values such as \'gladiator\'.',
        '"legalities"."status" uses values such as \'Banned\'.',
    ],
    "formula_1": [
        '"status"."statusId" = 2 means Disqualified.',
        '"results"."time" being non-null indicates the driver finished with a recorded result time.',
        '"races"."name" stores Grand Prix names such as \'Australian Grand Prix\'.',
        'Lap times are stored as text and must be parsed numerically for fastest/average comparisons.',
    ],
}


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
    )


def verifier_llm() -> ChatOpenAI:
    """Separate verifier client so verification can use a smaller/faster model."""
    return ChatOpenAI(
        model=VERIFIER_MODEL,
        base_url=VERIFIER_BASE_URL,
        api_key=VERIFIER_API_KEY,
        temperature=0.0,
        max_tokens=VERIFIER_MAX_TOKENS,
    )


# ---- Nodes ------------------------------------------------------------

def _q(ident: str) -> str:
    """Double-quote a SQLite identifier."""
    return '"' + ident.replace('"', '""') + '"'


def _question_terms(question: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]+", question.lower()):
        term = term.strip("'")
        if len(term) >= 3 and term not in QUESTION_STOP_WORDS and term not in terms:
            terms.append(term)
    return terms[:16]


def _looks_like_value_column(table: str, column: str, ctype: str, terms: list[str]) -> bool:
    del table, ctype
    lowered_column = column.lower()
    if any(part in lowered_column for part in VALUE_HINT_SKIP_COLUMN_PARTS):
        return False
    if any(term in lowered_column for term in terms):
        return True
    if lowered_column in {
        "name",
        "type",
        "status",
        "format",
        "label",
        "element",
        "department",
        "district",
        "city",
        "state",
        "colour",
        "color",
    }:
        return True
    return any(part in lowered_column for part in ("date", "time", "name", "type", "status"))


def _value_match_score(value: str, terms: list[str]) -> int:
    lowered = value.lower()
    return sum(1 for term in terms if term in lowered)


def _matching_values(conn: sqlite3.Connection, table: str, column: str, terms: list[str]) -> list[str]:
    if not terms:
        return []
    where = " OR ".join(f"LOWER(CAST({_q(column)} AS TEXT)) LIKE ?" for _ in terms)
    params = [f"%{term}%" for term in terms]
    sql = (
        f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
        f"WHERE {_q(column)} IS NOT NULL AND ({where}) "
        "LIMIT 100"
    )
    rows = conn.execute(sql, params).fetchall()
    values = [str(row[0]) for row in rows if row[0] not in (None, "") and len(str(row[0])) <= 80]
    return sorted(values, key=lambda value: (_value_match_score(value, terms), -len(value)), reverse=True)[
        :VALUE_HINT_MAX_VALUES
    ]


def _sample_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    sql = (
        f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
        f"WHERE {_q(column)} IS NOT NULL AND CAST({_q(column)} AS TEXT) != '' "
        f"LIMIT {VALUE_HINT_MAX_VALUES}"
    )
    rows = conn.execute(sql).fetchall()
    return [str(row[0]) for row in rows if row[0] not in (None, "") and len(str(row[0])) <= 80]


def _render_domain_aliases(db_id: str) -> str:
    aliases = DOMAIN_ALIASES.get(db_id)
    if not aliases:
        return ""
    lines = ["", "-- Domain aliases for opaque columns. Use these meanings when choosing columns."]
    lines.extend(f"-- {alias}" for alias in aliases)
    return "\n".join(lines)


@lru_cache(maxsize=512)
def _render_value_hints(db_id: str, question: str) -> str:
    """Add compact exact-value hints for likely categorical/text columns."""
    terms = _question_terms(question)
    if not terms:
        return ""

    hints: list[tuple[int, str, str, list[str]]] = []
    try:
        with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True, timeout=2.0) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                )
            ]
            for table in tables:
                for _cid, column, ctype, *_rest in conn.execute(f"PRAGMA table_info({_q(table)})"):
                    if not _looks_like_value_column(table, column, ctype or "", terms):
                        continue
                    values = _matching_values(conn, table, column, terms)
                    column_hits = sum(1 for term in terms if term in column.lower())
                    table_hits = sum(1 for term in terms if term in table.lower())
                    if not values and column_hits:
                        values = _sample_values(conn, table, column)
                    if values:
                        value_hits = max(_value_match_score(value, terms) for value in values)
                        score = (10 * value_hits) + (3 * column_hits) + table_hits
                        hints.append((score, table, column, values))
    except Exception:  # noqa: BLE001
        return ""

    if not hints:
        return ""

    lines = [
        "",
        "-- Relevant exact values from the database. Prefer these spellings/casing in filters.",
    ]
    for _score, table, column, values in sorted(hints, reverse=True)[:VALUE_HINT_MAX_COLUMNS]:
        rendered = ", ".join(repr(value) for value in values[:VALUE_HINT_MAX_VALUES])
        lines.append(f"-- {_q(table)}.{_q(column)} values: {rendered}")
    return "\n".join(lines)


def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    with timed_node("attach_schema"):
        trimmed_schema = retrieve_schema_for_question(state.db_id, state.question)
        enriched_schema = (
            trimmed_schema
            + _render_domain_aliases(state.db_id)
            + _render_value_hints(state.db_id, state.question)
        )
        return {"schema": enriched_schema}


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else text).strip()
    start = re.search(r"\b(WITH|SELECT)\b", sql, re.IGNORECASE)
    if start:
        sql = sql[start.start():]
    sql = sql.strip().rstrip("`").strip()
    if ";" in sql:
        sql = sql[: sql.rfind(";") + 1]
    return sql


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM reply."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _is_aggregate_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in (
            "average",
            "avg",
            "count",
            "difference",
            "how many",
            "number of",
            "percentage",
            "sum",
            "total",
        )
    )


def _is_aggregate_sql(sql: str) -> bool:
    return bool(re.search(r"\b(avg|count|sum|min|max)\s*\(", sql, flags=re.IGNORECASE))


def _expects_single_aggregate(question: str) -> bool:
    lowered = question.lower()
    if not _is_aggregate_question(question):
        return False
    grouped_cues = (" by each ", " for each ", " per ", " grouped by ", " group by ", " list ")
    return not any(cue in f" {lowered} " for cue in grouped_cues)


def _first_select_expression(sql: str) -> str:
    match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).lower() if match else ""


def _schema_chunks_for_verify(schema: str) -> dict[str, str]:
    chunks = re.split(r"\n(?=CREATE TABLE )", schema)
    tables: dict[str, str] = {}
    for chunk in chunks:
        match = re.match(r'CREATE TABLE "((?:[^"]|"")+)".*?\(', chunk, flags=re.DOTALL)
        if match:
            tables[match.group(1).replace('""', '"').lower()] = chunk
    return tables


def _sql_table_names(sql: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(
        r'\b(?:from|join)\s+(?:"((?:[^"]|"")+)"|([A-Za-z_][A-Za-z0-9_]*))',
        sql,
        flags=re.IGNORECASE,
    ):
        quoted, bare = match.groups()
        names.add((quoted or bare or "").replace('""', '"').lower())
    return {name for name in names if name}


def _compact_schema_for_verifier(state: AgentState) -> str:
    if VERIFY_SCHEMA_MAX_CHARS <= 0 or len(state.schema) <= VERIFY_SCHEMA_MAX_CHARS:
        return state.schema

    table_chunks = _schema_chunks_for_verify(state.schema)
    sql_tables = _sql_table_names(state.sql)
    selected = [chunk for table, chunk in table_chunks.items() if table in sql_tables]
    aliases = _render_domain_aliases(state.db_id).strip()
    parts = ["-- Compact verifier schema: only SQL-touched tables plus domain aliases."]
    if aliases:
        parts.append(aliases)
    parts.extend(selected)
    rendered = "\n".join(parts)
    if len(rendered) > VERIFY_SCHEMA_MAX_CHARS:
        return rendered[:VERIFY_SCHEMA_MAX_CHARS] + "\n-- verifier schema truncated"
    return rendered


def _execution_metadata(execution: ExecutionResult | None) -> str:
    if execution is None:
        return "status=not_executed"
    if not execution.ok:
        return f"status=error\nerror={execution.error}"
    columns = ", ".join(execution.columns or [])
    first_row = ""
    if execution.rows:
        first_row = " | ".join("" if cell is None else str(cell) for cell in execution.rows[0])
    return (
        "status=ok\n"
        f"row_count={execution.row_count}\n"
        f"columns={columns}\n"
        f"first_row={first_row}"
    )


def _adaptive_mode(config: RunnableConfig) -> str:
    metadata = config.get("metadata") if isinstance(config, dict) else None
    if not isinstance(metadata, dict):
        return "normal"
    return str(metadata.get("agent_adaptive_mode") or "normal")


def _needs_llm_verify(state: AgentState, config: RunnableConfig) -> tuple[bool, str]:
    if not CONDITIONAL_VERIFY:
        return True, "conditional verifier disabled"
    if _adaptive_mode(config) == "deterministic_only":
        return False, "adaptive pressure mode uses deterministic verifier only"
    execution = state.execution
    if execution is None or not execution.ok:
        return False, "deterministic rejection already handles missing/error execution"

    rows = execution.rows or []
    question = state.question.lower()
    sql = state.sql.lower()
    if not SEMANTIC_LLM_VERIFY:
        return False, "semantic LLM verifier disabled for SLO mode"
    if not rows and not _is_aggregate_sql(state.sql):
        return True, "zero rows for a non-aggregate answer"
    if re.search(r"\bselect\s+\*", sql):
        return True, "SELECT * needs semantic verifier review"
    if execution.row_count >= VERIFY_MANY_ROWS_THRESHOLD and not any(
        cue in question for cue in ("all", "list", "show", "which", "what are")
    ):
        return True, "many rows returned without an obvious list request"
    if "limit" not in sql and any(cue in question for cue in ("top ", "highest", "lowest", "first", "largest", "smallest")):
        return True, "ranking/superlative question without LIMIT"
    if "order by" not in sql and any(cue in question for cue in ("top ", "highest", "lowest", "largest", "smallest")):
        return True, "ranking/superlative question without ORDER BY"
    return False, "deterministic checks passed"


def _deterministic_verify_issue(state: AgentState) -> str | None:
    execution = state.execution
    if execution is None:
        return "SQL was not executed; revise the query before verification."
    if not execution.ok:
        return f"SQL execution returned an error: {execution.error}"

    rows = execution.rows or []
    columns = execution.columns or []
    if (
        rows
        and len(rows) == 1
        and len(rows[0]) == 1
        and (rows[0][0] is None or rows[0][0] == "")
        and (_is_aggregate_sql(state.sql) or _is_aggregate_question(state.question))
    ):
        return (
            "Aggregate result is NULL/empty. Reconsider the measure column and filters. "
            "For financial crime questions, use \"district\".\"A15\" for crimes in 1995 "
            "and \"district\".\"A16\" for crimes in 1996."
        )

    if not rows and not _is_aggregate_sql(state.sql) and not _is_aggregate_question(state.question):
        return (
            "Zero-row result for a non-aggregate question. Reconsider exact literals, "
            "date formats, joins, and filter columns."
        )

    if (
        rows
        and len(rows) > 1
        and _is_aggregate_sql(state.sql)
        and _expects_single_aggregate(state.question)
    ):
        return (
            "Aggregate shape mismatch: the question asks for one overall aggregate value, "
            "but the SQL returned multiple rows. Remove GROUP BY unless the question asks for "
            "a per-group result."
        )

    if _expects_single_aggregate(state.question) and re.search(r"\bgroup\s+by\b", state.sql, re.IGNORECASE):
        return (
            "Unrequested GROUP BY: the question asks for one overall aggregate value. "
            "Remove GROUP BY and compute the aggregate over the filtered rows."
        )

    if rows and len(rows) > 1 and not _is_aggregate_sql(state.sql):
        canonical_rows = [tuple("" if cell is None else str(cell) for cell in row) for row in rows]
        if len(set(canonical_rows)) < len(canonical_rows) and "distinct" not in state.sql.lower():
            return (
                "Duplicate rows detected. If the question asks for entities, coordinates, IDs, names, "
                "or a distinct list, add DISTINCT or fix the join cardinality."
            )

    question = state.question.lower()
    sql = state.sql.lower()
    select_expr = _first_select_expression(state.sql)

    if state.db_id == "california_schools":
        asks_school_id = (
            "nces school" in question
            or ("school" in question and "identification" in question)
            or ("school" in question and "id" in question)
        )
        if asks_school_id and "ncesdist" in select_expr and "ncesschool" not in select_expr:
            return (
                "Projection mismatch: the question asks for the NCES school identification number. "
                "Select \"schools\".\"NCESSchool\", not \"schools\".\"NCESDist\"."
            )
        asks_complete_address = "complete address" in question
        if asks_complete_address and columns and [c.lower() for c in columns[:4]] == [
            "street",
            "city",
            "zip",
            "state",
        ]:
            return (
                "Projection order mismatch: complete school addresses should be returned as "
                "\"Street\", \"City\", \"State\", \"Zip\"."
            )

    if state.db_id == "card_games" and "print card" in question:
        if re.search(r"\bname\b", select_expr) and not re.search(r"\bid\b", select_expr):
            return (
                "Projection mismatch: the question asks for print cards. "
                "Select DISTINCT \"cards\".\"id\" as the printed card identifier, not only the card name."
            )

    if state.db_id == "toxicology":
        if "carcinogenic" in question and "'carcinogenic'" in sql:
            return (
                "Wrong label literal: toxicology uses \"molecule\".\"label\" = '+' for carcinogenic "
                "and '-' for non carcinogenic, not the string 'carcinogenic'."
            )

    return None


def generate_sql_node(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the first SQL candidate for the question.

    Build messages from the tuned prompts, call the shared llm(), extract the
    SQL, and bump `iteration` so route_after_verify can enforce MAX_ITERATIONS.
    """
    with timed_node("generate_sql"):
        response = llm().invoke(
            [
                ("system", prompts.GENERATE_SQL_SYSTEM),
                ("user", prompts.GENERATE_SQL_USER.format(
                    schema=state.schema,
                    question=state.question,
                )),
            ],
            config=config,
        )
        sql = _extract_sql(response.content)
        return {
            "sql": sql,
            "iteration": state.iteration + 1,
            "history": state.history + [{"node": "generate_sql", "sql": sql}],
        }


def execute_node(state: AgentState) -> dict:
    """Run the current SQL and store the SQLite execution result."""
    with timed_node("execute"):
        return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState, config: RunnableConfig) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Deterministic checks run first. Clean, normal-shaped executions can be
    accepted without an LLM verifier; ambiguous cases use a compact verifier
    prompt and the separate verifier_llm() client.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    """
    with timed_node("verify"):
        deterministic_issue = _deterministic_verify_issue(state)
        if deterministic_issue:
            return {
                "verify_ok": False,
                "verify_issue": deterministic_issue,
                "history": state.history + [{
                    "node": "verify",
                    "ok": False,
                    "issue": deterministic_issue,
                    "source": "deterministic",
                }],
            }
        needs_llm, reason = _needs_llm_verify(state, config)
        if not needs_llm:
            issue = f"Accepted by deterministic verifier: {reason}."
            return {
                "verify_ok": True,
                "verify_issue": issue,
                "history": state.history + [{
                    "node": "verify",
                    "ok": True,
                    "issue": issue,
                    "source": "deterministic",
                }],
            }

        compact_schema = _compact_schema_for_verifier(state)
        compact_execution = _execution_metadata(state.execution)
        response = verifier_llm().invoke(
            [
                ("system", prompts.VERIFY_SYSTEM),
                ("user", prompts.VERIFY_USER.format(
                    schema=compact_schema,
                    question=state.question,
                    sql=state.sql,
                    execution=compact_execution,
                )),
            ],
            config=config,
        )
        parsed = _extract_json_object(response.content)
        ok = bool(parsed.get("ok", False))
        issue = str(parsed.get("issue") or "").strip()
        if not issue:
            issue = "Verifier accepted the result." if ok else "Verifier rejected the result without a reason."
        return {
            "verify_ok": ok,
            "verify_issue": issue,
            "history": state.history + [{
                "node": "verify",
                "ok": ok,
                "issue": issue,
                "source": "llm",
                "reason": reason,
                "raw": response.content,
            }],
        }


def revise_node(state: AgentState, config: RunnableConfig) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    with timed_node("revise"):
        execution = state.execution.render() if state.execution else "ERROR: SQL was not executed."
        revision_issue = state.verify_issue
        response = llm().invoke(
            [
                ("system", prompts.REVISE_SYSTEM),
                ("user", prompts.REVISE_USER.format(
                    schema=state.schema,
                    question=state.question,
                    sql=state.sql,
                    execution=execution,
                    issue=revision_issue,
                )),
            ],
            config=config,
        )
        sql = _extract_sql(response.content)
        repeated = _normalize_sql(sql) == _normalize_sql(state.sql)
        if repeated:
            revision_issue = (
                f"{state.verify_issue}\n"
                "The attempted revision repeated the previous SQL. Produce a materially different "
                "query: change the literal, join path, projection, DISTINCT, aggregation, or filter "
                "that is most likely wrong."
            )
            response = llm().invoke(
                [
                    ("system", prompts.REVISE_SYSTEM),
                    ("user", prompts.REVISE_USER.format(
                        schema=state.schema,
                        question=state.question,
                        sql=state.sql,
                        execution=execution,
                        issue=revision_issue,
                    )),
                ],
                config=config,
            )
            sql = _extract_sql(response.content)
        return {
            "sql": sql,
            "iteration": state.iteration + 1,
            "history": state.history + [{
                "node": "revise",
                "issue": revision_issue,
                "repeated_previous_sql": repeated,
                "sql": sql,
            }],
        }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    revision_count = max(0, state.iteration - 1)
    if state.verify_ok or state.iteration >= MAX_ITERATIONS or revision_count >= MAX_REVISIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()


def run_fast_path(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Latency-optimized serving path: generate once, execute once, skip verify/revise."""
    with timed_node("attach_schema"):
        state.schema = retrieve_schema_for_question(state.db_id, state.question) + _render_domain_aliases(state.db_id)

    generated = generate_sql_node(state, config)
    state.sql = generated["sql"]
    state.iteration = generated["iteration"]
    state.history = generated["history"]

    executed = execute_node(state)
    state.execution = executed["execution"]
    return {
        "schema": state.schema,
        "sql": state.sql,
        "iteration": state.iteration,
        "history": state.history,
        "execution": state.execution,
        "verify_ok": True,
        "verify_issue": "fast path skipped verifier",
    }
