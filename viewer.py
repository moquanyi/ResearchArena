"""ResearchArena experiment viewer — stream events, metrics, and artifacts."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

OUTPUTS_DIR = Path("outputs")
REFRESH_INTERVAL = 3  # seconds


# ── Data loading ─────────────────────────────────────────────────────────────

def find_runs() -> list[dict]:
    if not OUTPUTS_DIR.exists():
        return []

    # Group all events files by idea_dir so multi-stage runs appear once
    idea_dirs: dict[Path, list[Path]] = {}
    for events_file in OUTPUTS_DIR.rglob("*_events.jsonl"):
        idea_dir = events_file.parent.parent  # logs/ → idea_XX/
        idea_dirs.setdefault(idea_dir, []).append(events_file)

    runs = []
    for idea_dir, events_files in idea_dirs.items():
        workspace_dir = idea_dir.parent
        # All events files in sorted order — show them all, latest is current stage
        events_files_sorted = sorted(events_files, key=lambda f: f.stat().st_mtime)
        latest_events = events_files_sorted[-1]

        run: dict = {
            "workspace": workspace_dir,
            "idea_dir": idea_dir,
            "events_files": events_files_sorted,   # all stages
            "events_file": latest_events,           # current/latest stage
            "label": f"{workspace_dir.name} / {idea_dir.name}",
            "last_modified": latest_events.stat().st_mtime,
        }

        for fname, key in [("summary.json", "summary"), ("tracker.json", "tracker")]:
            p = workspace_dir / fname
            if p.exists():
                try:
                    run[key] = json.loads(p.read_text())
                except Exception:
                    run[key] = {}

        for fname, key in [("idea.json", "idea"), ("reviews.json", "reviews"), ("results.json", "results")]:
            p = idea_dir / fname
            if p.exists():
                try:
                    run[key] = json.loads(p.read_text())
                except Exception:
                    run[key] = {}

        run["title"] = (run.get("idea") or {}).get("title", "")
        run["score"] = (run.get("reviews") or {}).get("avg_score")
        run["decision"] = (run.get("reviews") or {}).get("decision", "")
        run["seed"] = (run.get("summary") or {}).get("seed_topic", workspace_dir.name)
        run["agent"] = (run.get("summary") or {}).get("agent", "claude")
        run["status"] = (run.get("summary") or {}).get("status", "running")
        run["is_running"] = (time.time() - run["last_modified"]) < 120

        runs.append(run)

    runs.sort(key=lambda r: r["last_modified"], reverse=True)
    return runs


def parse_events(events_file: Path) -> list[dict]:
    events = []
    try:
        for line in events_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return events


# ── Event rendering ───────────────────────────────────────────────────────────

def _fmt_tool_input(name: str, inp: dict) -> str:
    if name == "Bash":
        cmd = inp.get("command", "")
        return f"`{cmd[:300]}`"
    if name in ("Read", "Write", "Edit"):
        return f"`{inp.get('file_path', '')}`"
    if name == "WebSearch":
        return f"`{inp.get('query', '')}`"
    if name == "WebFetch":
        return f"`{inp.get('url', '')}`"
    s = json.dumps(inp)
    return (s[:200] + "…") if len(s) > 200 else s


def render_events(events: list[dict]) -> None:
    if not events:
        st.info("No events yet.")
        return

    for evt in events:
        ts = evt.get("ts", 0)
        event = evt.get("event", {})
        etype = event.get("type", "")
        t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        if etype == "system" and event.get("subtype") == "init":
            model = event.get("model", "")
            session = event.get("session_id", "")[:8]
            st.caption(f"🚀 `{t}` — session `{session}` · model `{model}`")
            continue

        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type", "")
                if btype == "text":
                    txt = block.get("text", "").strip()
                    if txt:
                        st.markdown(f"💬 `{t}` {txt}")
                elif btype == "tool_use":
                    name = block.get("name", "")
                    detail = _fmt_tool_input(name, block.get("input", {}))
                    st.markdown(f"🔧 `{t}` **{name}** {detail}")
                elif btype == "thinking":
                    thinking = block.get("thinking", "").strip()
                    if thinking:
                        with st.expander(f"💭 `{t}` thinking…", expanded=False):
                            st.text(thinking[:3000])
            continue

        if etype == "user":
            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                display = content[:600] + ("…" if len(content) > 600 else "")
                # Only show non-trivial results
                if len(content.strip()) > 5:
                    st.caption(f"↩️ `{t}` {display}")


# ── Metrics tab ───────────────────────────────────────────────────────────────

def render_metrics(run: dict) -> None:
    tracker = run.get("tracker") or {}
    reviews = run.get("reviews") or {}
    summary = run.get("summary") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost", f"${summary.get('total_cost_usd', tracker.get('total_cost_usd', 0)):.3f}")
    col2.metric("Wall time", f"{summary.get('wall_time_seconds', tracker.get('total_elapsed_seconds', 0))/60:.1f} min")
    col3.metric("Review score", f"{run['score']:.1f}/10" if run.get("score") else "—")

    stages = tracker.get("stages") or {}
    if stages:
        st.subheader("Stage breakdown")
        rows = []
        for stage, data in stages.items():
            tok = data.get("tokens", {})
            outcomes = data.get("outcomes", {})
            outcome_str = next(
                (k for k, v in outcomes.items() if v > 0), "—"
            )
            rows.append({
                "Stage": stage,
                "Time (s)": round(data.get("elapsed_seconds", 0), 1),
                "Tokens in": tok.get("input_tokens", 0),
                "Tokens out": tok.get("output_tokens", 0),
                "Cost $": round(data.get("cost_usd", 0), 4),
                "Outcome": outcome_str,
            })
        st.dataframe(rows, use_container_width=True)

    if reviews.get("reviews"):
        st.subheader("Peer reviews")
        for rev in reviews["reviews"]:
            src = rev.get("source", "reviewer")
            score = rev.get("overall_score", "—")
            decision = rev.get("decision", "")
            icon = {"accept": "✅", "revision": "🔄", "reject": "❌"}.get(decision, "❓")
            with st.expander(f"{icon} {src} — {score}/10"):
                scores = rev.get("scores", {})
                if scores:
                    st.dataframe(
                        [{"Criterion": k, "Score": v} for k, v in scores.items()],
                        use_container_width=True,
                    )
                for field in ("summary", "strengths", "weaknesses", "detailed_feedback"):
                    val = rev.get(field)
                    if val:
                        st.markdown(f"**{field.replace('_', ' ').title()}**")
                        if isinstance(val, list):
                            for item in val:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(val)


# ── Artifacts tab ─────────────────────────────────────────────────────────────

def render_artifacts(run: dict) -> None:
    idea_dir: Path = run["idea_dir"]

    st.subheader("Key files")
    for fname, label, lang in [
        ("idea.json",    "💡 Idea",          "json"),
        ("plan.json",    "📋 Plan",           "json"),
        ("results.json", "📊 Results",        "json"),
        ("paper.tex",    "📄 Paper (LaTeX)",  "latex"),
        ("reviews.json", "🔍 Reviews",        "json"),
    ]:
        fpath = idea_dir / fname
        if not fpath.exists():
            continue
        size = fpath.stat().st_size
        with st.expander(f"{label} · `{fname}` · {size:,} B"):
            text = fpath.read_text()
            if lang == "json":
                try:
                    st.json(json.loads(text))
                except Exception:
                    st.code(text[:4000])
            else:
                st.code(text[:4000], language=lang)

    st.subheader("Experiment outputs")
    exp_dir = idea_dir / "exp"
    if exp_dir.exists():
        for f in sorted(exp_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(idea_dir)
                st.caption(f"📂 `{rel}` · {f.stat().st_size:,} B")
    else:
        st.caption("No exp/ directory yet.")

    st.subheader("Logs")
    logs_dir = idea_dir / "logs"
    if logs_dir.exists():
        for f in sorted(logs_dir.iterdir()):
            st.caption(f"📝 `{f.name}` · {f.stat().st_size:,} B")


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ResearchArena Viewer", page_icon="🔬", layout="wide")
st.title("🔬 ResearchArena Experiment Viewer")

with st.sidebar:
    st.header("Runs")
    auto_refresh = st.toggle("Auto-refresh (3 s)", value=False)
    runs = find_runs()

    if not runs:
        st.warning(f"No runs found under `{OUTPUTS_DIR}/`")
        st.stop()

    labels = []
    for r in runs:
        icon = "🟢" if r["is_running"] else ("✅" if r["decision"] == "accept" else "⬜")
        score = f" · {r['score']:.1f}" if r.get("score") is not None else ""
        labels.append(f"{icon} {r['label']}{score}")

    selected_idx = st.radio("Select run", range(len(runs)), format_func=lambda i: labels[i])

run = runs[selected_idx]

# Header
col_a, col_b, col_c = st.columns([3, 1, 1])
with col_a:
    title = run.get("title") or run.get("seed") or run["label"]
    st.subheader(title)
with col_b:
    if run["is_running"]:
        st.success("🟢 Running")
    else:
        st.info(f"Status: {run['status']}")
with col_c:
    last = datetime.fromtimestamp(run["last_modified"]).strftime("%H:%M:%S")
    st.caption(f"Last event: {last}")

st.caption(f"`{run['events_file']}`")

tab_events, tab_metrics, tab_artifacts = st.tabs(["📡 Events", "📊 Metrics", "📁 Artifacts"])

with tab_events:
    events_files: list[Path] = run["events_files"]

    for ef in events_files:
        events = parse_events(ef)
        st.markdown(f"**`{ef.name}`** · {len(events)} events")
        render_events(events)
        st.divider()

with tab_metrics:
    render_metrics(run)

with tab_artifacts:
    render_artifacts(run)

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
