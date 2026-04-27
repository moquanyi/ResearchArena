"""Batch import existing JSON outputs into the tracking database."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from researcharena.tracking.store import TrackingStore

# Map stage names → artifact types for known output files
_ARTIFACT_MAP = {
    "idea.json":    "idea",
    "plan.json":    "plan",
    "results.json": "results",
    "paper.tex":    "paper",
    "paper.pdf":    "paper",
    "reviews.json": "review",
}

# Map log file suffixes → artifact types
_LOG_SUFFIXES = ("_events.jsonl", "_stdout.txt", "_stderr.txt", "_command.txt")


def _stable_run_id(workspace_dir: Path) -> str:
    return hashlib.md5(str(workspace_dir.absolute()).encode()).hexdigest()[:16]


def ingest_workspace(workspace_dir: Path, store: TrackingStore) -> str | None:
    """Import one workspace directory (e.g. outputs/rtx3090/) into the DB.

    Returns the run_id if ingested, None if skipped (no data found).
    """
    workspace_dir = workspace_dir.resolve()
    idea_dirs = sorted(workspace_dir.glob("idea_*"))
    if not idea_dirs:
        return None

    run_id = _stable_run_id(workspace_dir)

    # Load workspace-level files
    summary: dict = {}
    tracker_data: dict = {}
    summary_path = workspace_dir / "summary.json"
    tracker_path = workspace_dir / "tracker.json"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass
    if tracker_path.exists():
        try:
            tracker_data = json.loads(tracker_path.read_text())
        except Exception:
            pass

    agent = summary.get("agent", tracker_data.get("agent_type", ""))
    model = summary.get("agent_model", "")
    status = summary.get("status", "running")
    wall_time = summary.get("wall_time_seconds")
    best_paper = summary.get("best_paper") or {}
    best_score = best_paper.get("score")
    decision = None

    # Derive started_at from oldest events file or tracker
    started_at = time.time()
    for idea_dir in idea_dirs:
        for ef in sorted((idea_dir / "logs").glob("*_events.jsonl") if (idea_dir / "logs").exists() else []):
            started_at = min(started_at, ef.stat().st_mtime)

    # Try to read seed from summary or workspace name
    seed = summary.get("seed_topic", workspace_dir.name)

    store.insert_run(
        run_id=run_id,
        seed=seed,
        agent=agent,
        model=model,
        platform=summary.get("platform", "gpu"),
        workspace=str(workspace_dir),
        config_json=json.dumps(summary),
    )

    # Ingest stage actions from tracker.json
    actions = tracker_data.get("actions", [])
    step = 0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    for action in actions:
        step += 1
        stage = action.get("stage", "")
        attempt = action.get("attempt") or 1
        outcome = action.get("outcome", "")
        elapsed = action.get("elapsed_seconds", 0.0)
        tok = action.get("tokens", {})
        tin = tok.get("input_tokens", 0)
        tout = tok.get("output_tokens", 0)
        cost = action.get("cost_usd", 0.0)
        details = action.get("details", "")
        fail_cat = action.get("failure_category")
        start = action.get("start_time", started_at)
        end = action.get("end_time", start + elapsed)

        total_tokens_in += tin
        total_tokens_out += tout
        total_cost += cost

        store.insert_stage(
            run_id=run_id, stage=stage, attempt=attempt,
            status=outcome, started_at=start, finished_at=end,
            elapsed_s=elapsed, tokens_in=tin, tokens_out=tout,
            cost_usd=cost, outcome=outcome,
            failure_category=fail_cat, details=details,
        )

        if tin + tout > 0:
            store.log_metric(run_id, "tokens_total", tin + tout, stage=stage, step=step)

        # Extract numeric score from details string
        import re
        m = re.search(r"score=([\d.]+)", details)
        if m:
            store.log_metric(run_id, f"{stage}_score", float(m.group(1)), stage=stage, step=step)

    # Ingest per-idea artifacts and review scores
    for idea_dir in idea_dirs:
        for fname, atype in _ARTIFACT_MAP.items():
            p = idea_dir / fname
            if p.exists():
                store.log_artifact(run_id, fname, atype, str(p.absolute()), p.stat().st_size)

        # Reviews
        reviews_path = idea_dir / "reviews.json"
        if reviews_path.exists():
            try:
                reviews = json.loads(reviews_path.read_text())
                avg = reviews.get("avg_score")
                if avg is not None:
                    store.log_metric(run_id, "review_score", float(avg), stage="review", step=step)
                    best_score = best_score or float(avg)
                    decision = reviews.get("decision")
            except Exception:
                pass

        # Log files
        logs_dir = idea_dir / "logs"
        if logs_dir.exists():
            for f in logs_dir.iterdir():
                if any(f.name.endswith(s) for s in _LOG_SUFFIXES):
                    store.log_artifact(run_id, f.name, "log", str(f.absolute()), f.stat().st_size)

    store.finish_run(
        run_id=run_id, status=status,
        tokens_in=total_tokens_in, tokens_out=total_tokens_out,
        cost_usd=total_cost, best_score=best_score,
        decision=decision, wall_time_s=wall_time,
    )
    return run_id


def ingest_all(outputs_dir: Path, store: TrackingStore) -> list[str]:
    """Scan outputs_dir for all workspace dirs and ingest each one."""
    run_ids = []
    for candidate in sorted(outputs_dir.iterdir()):
        if candidate.is_dir() and any(candidate.glob("idea_*")):
            rid = ingest_workspace(candidate, store)
            if rid:
                run_ids.append(rid)
    return run_ids
