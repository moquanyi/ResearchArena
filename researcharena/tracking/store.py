"""SQLite-backed store for experiment tracking."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path.home() / ".researcharena" / "runs.db"


def _db_path() -> Path:
    import os
    p = Path(os.environ.get("RESEARCHARENA_DB", "") or DEFAULT_DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class TrackingStore:
    def __init__(self, db_path: Path | str | None = None):
        self.path = Path(db_path) if db_path else _db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT UNIQUE NOT NULL,
                seed        TEXT,
                agent       TEXT,
                model       TEXT,
                platform    TEXT,
                status      TEXT DEFAULT 'running',
                workspace   TEXT,
                config_json TEXT,
                started_at  REAL,
                finished_at REAL,
                wall_time_s REAL,
                tokens_in   INTEGER DEFAULT 0,
                tokens_out  INTEGER DEFAULT 0,
                cost_usd    REAL DEFAULT 0,
                best_score  REAL,
                decision    TEXT
            );

            CREATE TABLE IF NOT EXISTS stages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT REFERENCES runs(run_id),
                stage            TEXT,
                attempt          INTEGER DEFAULT 1,
                status           TEXT,
                started_at       REAL,
                finished_at      REAL,
                elapsed_s        REAL,
                tokens_in        INTEGER DEFAULT 0,
                tokens_out       INTEGER DEFAULT 0,
                cost_usd         REAL DEFAULT 0,
                outcome          TEXT,
                failure_category TEXT,
                details          TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT REFERENCES runs(run_id),
                stage      TEXT,
                step       INTEGER DEFAULT 0,
                key        TEXT,
                value      REAL,
                logged_at  REAL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT REFERENCES runs(run_id),
                name       TEXT,
                type       TEXT,
                path       TEXT,
                size_bytes INTEGER,
                created_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_runs_agent     ON runs(agent);
            CREATE INDEX IF NOT EXISTS idx_metrics_run    ON metrics(run_id, key);
            CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts(run_id, type);
            CREATE INDEX IF NOT EXISTS idx_stages_run     ON stages(run_id);
        """)
        self._conn.commit()

    # ── runs ─────────────────────────────────────────────────────────────────

    def insert_run(self, run_id: str, seed: str, agent: str, model: str,
                   platform: str, workspace: str, config_json: str = ""):
        with self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO runs
                   (run_id, seed, agent, model, platform, workspace, config_json, started_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, seed, agent, model, platform, workspace, config_json, time.time()),
            )

    def finish_run(self, run_id: str, status: str,
                   tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0,
                   best_score: float | None = None, decision: str | None = None,
                   wall_time_s: float | None = None):
        now = time.time()
        with self._conn:
            self._conn.execute(
                """UPDATE runs SET status=?, finished_at=?, wall_time_s=?,
                   tokens_in=?, tokens_out=?, cost_usd=?, best_score=?, decision=?
                   WHERE run_id=?""",
                (status, now, wall_time_s, tokens_in, tokens_out, cost_usd,
                 best_score, decision, run_id),
            )

    def list_runs(self, limit: int = 50, agent: str | None = None,
                  status: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM runs"
        conds, params = [], []
        if agent:
            conds.append("agent = ?"); params.append(agent)
        if status:
            conds.append("status = ?"); params.append(status)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(q, params).fetchall()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM runs WHERE run_id=? OR run_id LIKE ?",
            (run_id, f"{run_id}%"),
        ).fetchone()

    # ── stages ───────────────────────────────────────────────────────────────

    def insert_stage(self, run_id: str, stage: str, attempt: int, status: str,
                     started_at: float, finished_at: float, elapsed_s: float,
                     tokens_in: int, tokens_out: int, cost_usd: float,
                     outcome: str, failure_category: str | None, details: str):
        with self._conn:
            self._conn.execute(
                """INSERT INTO stages
                   (run_id, stage, attempt, status, started_at, finished_at,
                    elapsed_s, tokens_in, tokens_out, cost_usd,
                    outcome, failure_category, details)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, stage, attempt, status, started_at, finished_at,
                 elapsed_s, tokens_in, tokens_out, cost_usd,
                 outcome, failure_category, details),
            )

    def get_stages(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM stages WHERE run_id=? ORDER BY started_at",
            (run_id,),
        ).fetchall()

    # ── metrics ──────────────────────────────────────────────────────────────

    def log_metric(self, run_id: str, key: str, value: float,
                   stage: str = "", step: int = 0):
        with self._conn:
            self._conn.execute(
                "INSERT INTO metrics (run_id, stage, step, key, value, logged_at) VALUES (?,?,?,?,?,?)",
                (run_id, stage, step, key, value, time.time()),
            )

    def get_metrics(self, run_id: str, key: str | None = None) -> list[sqlite3.Row]:
        if key:
            return self._conn.execute(
                "SELECT * FROM metrics WHERE run_id=? AND key=? ORDER BY logged_at",
                (run_id, key),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM metrics WHERE run_id=? ORDER BY logged_at",
            (run_id,),
        ).fetchall()

    def metric_keys(self, run_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT key FROM metrics WHERE run_id=?", (run_id,)
        ).fetchall()
        return [r["key"] for r in rows]

    # ── artifacts ────────────────────────────────────────────────────────────

    def log_artifact(self, run_id: str, name: str, type: str,
                     path: str, size_bytes: int):
        with self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO artifacts
                   (run_id, name, type, path, size_bytes, created_at) VALUES (?,?,?,?,?,?)""",
                (run_id, name, type, path, size_bytes, time.time()),
            )

    def get_artifacts(self, run_id: str, type: str | None = None) -> list[sqlite3.Row]:
        if type:
            return self._conn.execute(
                "SELECT * FROM artifacts WHERE run_id=? AND type=? ORDER BY created_at",
                (run_id, type),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()

    def close(self):
        self._conn.close()
