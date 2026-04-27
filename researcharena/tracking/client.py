"""Pipeline-facing tracking client — mirrors wandb's init/log/finish API."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from researcharena.tracking.store import TrackingStore


class TrackingClient:
    def __init__(self, db_path: Path | str | None = None):
        self.store = TrackingStore(db_path)
        self.run_id: str | None = None
        self._stage_start: float = 0.0
        self._current_stage: str = ""
        self._current_attempt: int = 1
        self._step: int = 0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._total_cost: float = 0.0

    def init_run(self, cfg: dict) -> str:
        self.run_id = uuid.uuid4().hex[:16]
        self.store.insert_run(
            run_id=self.run_id,
            seed=cfg.get("seed_topic", ""),
            agent=cfg.get("agent", {}).get("type", ""),
            model=cfg.get("agent", {}).get("model", ""),
            platform=cfg.get("seed_platform", "gpu"),
            workspace=str(Path(cfg.get("experiment", {}).get("workspace", "")).absolute()),
            config_json=json.dumps(cfg),
        )
        return self.run_id

    def begin_stage(self, stage: str, attempt: int = 1):
        self._stage_start = time.time()
        self._current_stage = stage
        self._current_attempt = attempt

    def end_stage(self, outcome: str, tokens_in: int = 0, tokens_out: int = 0,
                  cost_usd: float = 0.0, details: str = "",
                  failure_category: str | None = None):
        if not self.run_id:
            return
        now = time.time()
        elapsed = now - self._stage_start
        status = "success" if outcome == "success" else outcome

        self._total_tokens_in += tokens_in
        self._total_tokens_out += tokens_out
        self._total_cost += cost_usd
        self._step += 1

        self.store.insert_stage(
            run_id=self.run_id,
            stage=self._current_stage,
            attempt=self._current_attempt,
            status=status,
            started_at=self._stage_start,
            finished_at=now,
            elapsed_s=elapsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            outcome=outcome,
            failure_category=failure_category,
            details=details,
        )

        # Auto-log token metric
        if tokens_in + tokens_out > 0:
            self.store.log_metric(
                self.run_id, "tokens_total", tokens_in + tokens_out,
                stage=self._current_stage, step=self._step,
            )

        # Extract score from details like "score=7.5, passed"
        m = re.search(r"score=([\d.]+)", details)
        if m:
            self.store.log_metric(
                self.run_id, f"{self._current_stage}_score", float(m.group(1)),
                stage=self._current_stage, step=self._step,
            )

    def log_metric(self, key: str, value: float, stage: str = "", step: int | None = None):
        if not self.run_id:
            return
        self.store.log_metric(
            self.run_id, key, value,
            stage=stage or self._current_stage,
            step=step if step is not None else self._step,
        )

    def log_artifact(self, path: Path, type: str):
        if not self.run_id or not path.exists():
            return
        self.store.log_artifact(
            run_id=self.run_id,
            name=path.name,
            type=type,
            path=str(path.absolute()),
            size_bytes=path.stat().st_size,
        )

    def finish_run(self, status: str, best_score: float | None = None,
                   decision: str | None = None, wall_time_s: float | None = None):
        if not self.run_id:
            return
        self.store.finish_run(
            run_id=self.run_id,
            status=status,
            tokens_in=self._total_tokens_in,
            tokens_out=self._total_tokens_out,
            cost_usd=self._total_cost,
            best_score=best_score,
            decision=decision,
            wall_time_s=wall_time_s,
        )
        if best_score is not None:
            self.store.log_metric(self.run_id, "best_score", best_score, step=self._step)
