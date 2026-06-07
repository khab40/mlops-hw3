"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
SCHEMA_MAX_CHARS = int(os.environ.get("AGENT_SCHEMA_MAX_CHARS", "12000"))


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                ref = f" REFERENCES {_q(fk[2])}"
                if fk[4] is not None:
                    ref += f"({_q(fk[4])})"
                col_lines.append(f"  FOREIGN KEY ({_q(fk[3])}){ref}")
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def _question_terms(question: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", question.lower()) if len(t) > 2}


def trim_schema_for_question(schema: str, question: str, max_chars: int = SCHEMA_MAX_CHARS) -> str:
    """Keep prompt context bounded by prioritizing tables that match question terms."""
    if max_chars <= 0 or len(schema) <= max_chars:
        return schema

    chunks = re.split(r"\n(?=CREATE TABLE )", schema)
    header = chunks[0].split("\nCREATE TABLE ", 1)[0].strip()
    table_chunks = [c for c in chunks if c.startswith("CREATE TABLE ")]
    terms = _question_terms(question)

    def score(chunk: str) -> tuple[int, int]:
        lowered = chunk.lower()
        hits = sum(1 for term in terms if term in lowered)
        return hits, -len(chunk)

    ordered = sorted(table_chunks, key=score, reverse=True)
    parts = [header, "-- Schema trimmed for serving latency; tables most relevant to the question are shown first."]
    size = sum(len(p) + 1 for p in parts)
    for chunk in ordered:
        added = len(chunk) + 1
        if size + added > max_chars:
            continue
        parts.append(chunk)
        size += added
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
