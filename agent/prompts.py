"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a careful text-to-SQL generator for SQLite.

Rules:
- Return exactly one read-only SQLite SELECT query.
- Use only tables and columns present in the provided schema.
- Quote identifiers with double quotes when they contain spaces, punctuation, or mixed case.
- Do not invent columns, tables, CTE inputs, or values that are not implied by the question.
- Do not explain the query. Do not use markdown.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

Write the SQL query that answers the question."""


VERIFY_SYSTEM = """You verify whether a SQLite query result plausibly answers a user question.

Return only a compact JSON object with this schema:
{{"ok": true_or_false, "issue": "short reason"}}

Mark ok=false when:
- the SQL execution returned an error,
- the SQL does not address the question,
- the selected columns do not contain the requested information,
- zero rows look implausible for the question,
- the query uses an obviously wrong table/column/value,
- aggregation, sorting, filtering, or limiting is missing when the question asks for it.

Mark ok=true when the SQL executed and the result shape is a plausible answer, even if you cannot prove it is perfect.
"""

VERIFY_USER = """Question:
{question}

SQL:
{sql}

Execution result:
{execution}

Should this answer be accepted? Return JSON only."""


REVISE_SYSTEM = """You repair SQLite SELECT queries.

Rules:
- Return exactly one corrected read-only SQLite SELECT query.
- Use only tables and columns present in the schema.
- Preserve the user's intent, but fix the verifier issue and execution error/result mismatch.
- Quote identifiers with double quotes when needed.
- Do not explain the query. Do not use markdown.
"""

REVISE_USER = """Schema:
{schema}

Question:
{question}

Previous SQL:
{sql}

Execution result:
{execution}

Verifier issue:
{issue}

Return the revised SQL query."""
