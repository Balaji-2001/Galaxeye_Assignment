from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tiles.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    filename            TEXT NOT NULL,
    ingested_at         TEXT NOT NULL,          -- ISO8601 UTC
    label               TEXT NOT NULL,
    confidence          REAL NOT NULL,
    low_confidence      INTEGER NOT NULL,       -- 0/1
    class_probabilities TEXT NOT NULL,          -- JSON dict, full distribution
    model_version       TEXT NOT NULL,
    -- room to grow without a migration: tile geolocation, capture time, etc.
    -- are not required by this exercise but the schema anticipates them.
    lat                 REAL,
    lon                 REAL
);
CREATE INDEX IF NOT EXISTS idx_tiles_label ON tiles(label);
CREATE INDEX IF NOT EXISTS idx_tiles_low_confidence ON tiles(low_confidence);
CREATE INDEX IF NOT EXISTS idx_tiles_ingested_at ON tiles(ingested_at);
"""


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_result(filename: str, ingested_at: str, result: dict, model_version: str,
                   lat: float | None = None, lon: float | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tiles
               (filename, ingested_at, label, confidence, low_confidence,
                class_probabilities, model_version, lat, lon)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename, ingested_at, result["label"], result["confidence"],
                int(result["low_confidence"]),
                json.dumps(result["class_probabilities"]),
                model_version, lat, lon,
            ),
        )
        return cur.lastrowid


def get_by_id(tile_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,)).fetchone()
        return _row_to_dict(row) if row else None


def query(label: str | None = None, min_confidence: float | None = None,
          low_confidence_only: bool = False, limit: int = 50, offset: int = 0) -> list[dict]:
    clauses, params = [], []
    if label:
        clauses.append("label = ?")
        params.append(label)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    if low_confidence_only:
        clauses.append("low_confidence = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM tiles {where} ORDER BY ingested_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM tiles").fetchone()["n"]
        by_label = conn.execute(
            "SELECT label, COUNT(*) AS n, AVG(confidence) AS avg_conf FROM tiles GROUP BY label"
        ).fetchall()
        low_conf = conn.execute(
            "SELECT COUNT(*) AS n FROM tiles WHERE low_confidence = 1"
        ).fetchone()["n"]
    return {
        "total_tiles": total,
        "low_confidence_tiles": low_conf,
        "low_confidence_rate": round(low_conf / total, 4) if total else 0.0,
        "by_label": {r["label"]: {"count": r["n"], "avg_confidence": round(r["avg_conf"], 4)}
                     for r in by_label},
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["low_confidence"] = bool(d["low_confidence"])
    d["class_probabilities"] = json.loads(d["class_probabilities"])
    return d
