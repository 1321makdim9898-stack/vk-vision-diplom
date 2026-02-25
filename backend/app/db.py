from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/app.db")

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            source_ref TEXT,
            thumb_path TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def insert_analysis(created_at: str, source: str, source_ref: Optional[str], thumb_path: str, result_json: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (created_at, source, source_ref, thumb_path, result_json) VALUES (?,?,?,?,?)",
        (created_at, source, source_ref, thumb_path, result_json),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid

def list_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, source, source_ref, thumb_path FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_analysis(analysis_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
