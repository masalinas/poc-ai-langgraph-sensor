"""
Persistent memory for the agent's Reflect step, backed by SQLite.

Swap `DB_PATH` / the connection logic for Postgres, MinIO+DuckDB, etc. if you
want this to line up with the rest of your VeraDoc storage stack -- the
interface (record / recent) is what the graph nodes depend on, not the engine.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "database/agent_memory.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                topic      TEXT NOT NULL,
                reading    TEXT NOT NULL,   -- JSON blob of the sensed payload
                decision   TEXT NOT NULL,
                reasoning  TEXT NOT NULL,
                engine     TEXT NOT NULL DEFAULT 'rules',  -- 'rules' | 'llm' | 'rules_fallback'
                created_at TEXT NOT NULL
            )
            """
        )
        # Backfill for DBs created before `engine` existed.
        cols = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
        if "engine" not in cols:
            conn.execute("ALTER TABLE memory ADD COLUMN engine TEXT NOT NULL DEFAULT 'rules'")


def record(topic: str, reading: dict, decision: str, reasoning: str, engine: str = "rules") -> int:
    """Called by the Reflect node after every cycle. `engine` records whether
    the rule layer decided outright ('rules'), the LLM was consulted for an
    ambiguous case ('llm'), or the LLM failed and we fell back ('rules_fallback').
    Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memory (topic, reading, decision, reasoning, engine, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (topic, json.dumps(reading), decision, reasoning, engine, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def recent(topic: str, limit: int = 3) -> list[dict]:
    """Called by the Reason node to give the LLM short-term history for this
    specific sensor/topic, most recent last."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, reading, decision, reasoning, engine, created_at FROM memory "
            "WHERE topic = ? ORDER BY id DESC LIMIT ?",
            (topic, limit),
        ).fetchall()
 
    rows.reverse()  # oldest -> newest, easier for the LLM to read as a timeline
    return [
        {
            "cycle": r[0],
            "reading": json.loads(r[1]),
            "decision": r[2],
            "reasoning": r[3],
            "engine": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
