"""CLI entry point for researcharena."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from researcharena.utils.config import load_config, merge_configs

console = Console()


def _normalize_seeds(raw_seeds: list) -> list[dict]:
    """Normalize seeds to list of dicts.

    Supports both old format (list of strings) and new format (list of dicts
    with name/conferences/platform fields).
    """
    normalized = []
    for s in raw_seeds:
        if isinstance(s, str):
            normalized.append({"name": s, "conferences": [], "platform": "gpu", "domain": "ml"})
        elif isinstance(s, dict):
            normalized.append({
                "name": s.get("name", ""),
                "conferences": s.get("conferences", []),
                "platform": s.get("platform", "gpu"),
                "domain": s.get("domain", "ml"),
            })
    return normalized


def _resolve_platform_config(cfg: dict, platform: str) -> dict:
    """Merge platform-specific resources and docker_image into the config.

    Always returns a new dict — never mutates the input.
    """
    platforms = cfg.get("platforms", {})
    plat_cfg = platforms.get(platform, {})

    overrides = {"seed_platform": platform}

    if plat_cfg:
        if "resources" in plat_cfg:
            overrides["resources"] = plat_cfg["resources"]
        if "docker_image" in plat_cfg:
            overrides.setdefault("agent", {})["docker_image"] = plat_cfg["docker_image"]

    return merge_configs(cfg, overrides)


@click.group()
def main():
    """ResearchArena: benchmark CLI agents on autonomous research."""
    pass


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seed", "-s", default=None, help="Override seed topic")
@click.option("--agent", default=None, help="Agent type (claude/codex/kimi/minimax/custom)")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Platform (gpu/cpu)")
@click.option("--domain", default=None, type=click.Choice(["ml", "systems", "databases", "pl", "theory", "security"]), help="Domain (selects guideline templates)")
@click.option("--workspace", "-w", default=None, help="Override output workspace directory")
@click.option("--resume", "-r", default=None, type=click.Path(exists=True), help="Resume from existing idea workspace (e.g., outputs/kimi/run_3/computer_vision/idea_01)")
def run(config, seed, agent, model, max_ideas, platform, domain, workspace, resume):
    """Run the full research pipeline with a CLI agent."""
    cfg = load_config(config)

    overrides = {}
    if seed:
        overrides["seed_topic"] = seed
    if agent:
        overrides.setdefault("agent", {})["type"] = agent
    if model:
        overrides.setdefault("agent", {})["model"] = model
    if max_ideas:
        overrides.setdefault("pipeline", {})["max_ideas_per_seed"] = max_ideas
    if domain:
        overrides["seed_domain"] = domain
    if workspace:
        overrides.setdefault("experiment", {})["workspace"] = workspace
    elif seed:
        # Auto-derive workspace from seed so each seed gets its own directory
        slug = seed.replace(" ", "_").replace("/", "_").lower()[:60]
        overrides.setdefault("experiment", {})["workspace"] = f"outputs/runs/{slug}"

    if overrides:
        cfg = merge_configs(cfg, overrides)

    # Resolve platform (CLI flag > config > default "gpu")
    resolved_platform = platform or cfg.get("seed_platform", "gpu")
    cfg = _resolve_platform_config(cfg, resolved_platform)

    from researcharena.pipeline import Pipeline

    pipeline = Pipeline(cfg)

    if resume:
        result = pipeline.resume(resume)
    else:
        result = pipeline.run()

    # Save final summary
    summary_path = Path(cfg["experiment"]["workspace"]) / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2))

    console.print(f"\n[bold]Summary saved to {summary_path}[/]")
    console.print(json.dumps(result, indent=2))


@main.command()
@click.option("--config", "-c", default="configs/default.yaml")
@click.option("--domain", default=None, type=click.Choice(["ml", "systems", "databases", "pl", "theory", "security"]), help="Domain (selects reviewer guidelines)")
@click.argument("workspace", type=click.Path(exists=True))
def review_only(config, domain, workspace):
    """Run review on an existing paper (skip research stages)."""
    cfg = load_config(config)
    workspace = Path(workspace)

    from researcharena.stages.review import review_paper, save_reviews

    paper_tex = workspace / "paper.tex"
    paper_pdf = workspace / "paper.pdf"

    if not paper_tex.exists():
        console.print(f"[red]No paper.tex found in {workspace}[/]")
        return

    resolved_domain = domain or cfg.get("seed_domain", "ml")

    # Use all agents as reviewers for standalone review
    result = review_paper(
        paper_latex=paper_tex.read_text(),
        paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
        reviewer_agents=cfg["review"].get("agents", []),
        paperreview_config=cfg["review"].get("paperreview", {}),
        venue=cfg.get("seed_conferences", [None])[0] or cfg["paper"].get("template") or {"ml": "neurips", "systems": "osdi", "databases": "sigmod", "pl": "pldi", "theory": "stoc", "security": "ccs"}.get(resolved_domain, "neurips"),
        accept_threshold=cfg["review"]["accept_threshold"],
        workspace=workspace,
        docker_image=cfg["agent"].get("docker_image", "researcharena/agent:latest"),
        runtime=cfg["agent"].get("runtime", "docker"),
        domain=resolved_domain,
    )
    save_reviews(result, workspace)
    console.print(f"Score: {result.avg_score:.1f}/10, Decision: {result.decision}")


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seeds-file", default="configs/seeds.yaml", help="Seeds YAML file")
@click.option("--field", "-f", default=None, help="Run only this field (e.g., 'computer vision')")
@click.option("--agent", default=None, help="Agent type override")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
@click.option("--conference", default=None, help="Filter seeds by conference (e.g., sigmod, iclr)")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Filter seeds by platform")
def bench(config, seeds_file, field, agent, model, max_ideas, conference, platform):
    """Run the benchmark across seed fields.

    Each seed is a field name with a platform (gpu/cpu) and conference tags.
    The agent decides what specific problem to research — that's part of what
    we're testing.

    Examples:
      researcharena bench --agent claude                      # all seeds
      researcharena bench --agent claude --platform gpu       # GPU seeds only
      researcharena bench --agent claude --conference sigmod  # SIGMOD seeds
      researcharena bench --agent claude --field "query optimization"
    """
    cfg = load_config(config)
    seeds_cfg = load_config(seeds_file)

    all_seeds = _normalize_seeds(seeds_cfg.get("seeds", []))

    # Apply filters
    seeds = all_seeds
    if field:
        seeds = [s for s in seeds if s["name"] == field]
        if not seeds:
            all_names = [s["name"] for s in all_seeds]
            console.print(f"[red]Unknown field: {field}.[/]")
            # Show close matches
            close = [n for n in all_names if field.lower() in n.lower()]
            if close:
                console.print(f"[yellow]Did you mean: {close}[/]")
            return

    if conference:
        conf_lower = conference.lower()
        seeds = [s for s in seeds if conf_lower in [c.lower() for c in s["conferences"]]]
        if not seeds:
            console.print(f"[red]No seeds found for conference: {conference}[/]")
            return

    if platform:
        seeds = [s for s in seeds if s["platform"] == platform]
        if not seeds:
            console.print(f"[red]No seeds found for platform: {platform}[/]")
            return

    console.print(f"[bold]Running {len(seeds)} seed(s)[/]")
    if conference:
        console.print(f"  Conference filter: {conference}")
    if platform:
        console.print(f"  Platform filter: {platform}")

    overrides = {}
    if agent:
        overrides.setdefault("agent", {})["type"] = agent
    if model:
        overrides.setdefault("agent", {})["model"] = model
    if max_ideas:
        overrides.setdefault("pipeline", {})["max_ideas_per_seed"] = max_ideas

    from researcharena.pipeline import Pipeline
    from rich.table import Table

    results = []
    for seed in seeds:
        seed_name = seed["name"]
        seed_platform = seed["platform"]

        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(
            f"[bold magenta]Seed: {seed_name}  |  "
            f"Platform: {seed_platform}  |  "
            f"Conferences: {', '.join(seed['conferences'])}[/]"
        )
        console.print(f"[bold magenta]{'='*60}[/]")

        run_cfg = merge_configs(cfg, {
            **overrides,
            "seed_topic": seed_name,
            "seed_domain": seed.get("domain", "ml"),
            "seed_conferences": seed.get("conferences", []),
        })
        run_cfg = _resolve_platform_config(run_cfg, seed_platform)

        slug = seed_name.replace(" ", "_").replace("/", "_").lower()
        run_cfg["experiment"]["workspace"] = f"outputs/runs/{slug}"

        pipeline = Pipeline(run_cfg)
        result = pipeline.run()
        result["seed_topic"] = seed_name
        result["platform"] = seed_platform
        result["conferences"] = seed["conferences"]
        results.append(result)

        # Save per-seed summary
        summary_path = Path(run_cfg["experiment"]["workspace"]) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2))

    # Print leaderboard
    console.print(f"\n[bold]{'='*60}[/]")
    table = Table(title="Benchmark Results")
    table.add_column("Seed")
    table.add_column("Platform")
    table.add_column("Status")
    table.add_column("Best Score", justify="center")
    table.add_column("Best Title")
    table.add_column("Ideas", justify="center")

    for r in results:
        best = r.get("best_paper")
        score = f"{best['score']:.1f}" if best else "-"
        title = best.get("description", best.get("title", ""))[:40] if best else "-"
        table.add_row(
            r.get("seed_topic", ""),
            r.get("platform", ""),
            r.get("status", ""),
            score,
            title,
            str(r.get("ideas_tried", 0)),
        )

    console.print(table)

    combined_path = Path("outputs/runs/benchmark_results.json")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(results, indent=2))
    console.print(f"\nResults saved to {combined_path}")


@main.command()
@click.option("--seeds-file", default="configs/seeds.yaml", help="Seeds YAML file")
@click.option("--conference", default=None, help="Filter by conference")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Filter by platform")
def list_seeds(seeds_file, conference, platform):
    """List all available seed fields."""
    cfg = load_config(seeds_file)
    all_seeds = _normalize_seeds(cfg.get("seeds", []))

    seeds = all_seeds
    if conference:
        conf_lower = conference.lower()
        seeds = [s for s in seeds if conf_lower in [c.lower() for c in s["conferences"]]]
    if platform:
        seeds = [s for s in seeds if s["platform"] == platform]

    gpu_seeds = [s for s in seeds if s["platform"] == "gpu"]
    cpu_seeds = [s for s in seeds if s["platform"] == "cpu"]

    console.print(f"[bold]Available seeds ({len(seeds)} total, {len(gpu_seeds)} GPU, {len(cpu_seeds)} CPU):[/]")

    if gpu_seeds:
        console.print(f"\n[bold green]GPU platform ({len(gpu_seeds)}):[/]")
        for s in gpu_seeds:
            confs = ", ".join(s["conferences"]) if s["conferences"] else "—"
            console.print(f"  - {s['name']}  [dim]({confs})[/]")

    if cpu_seeds:
        console.print(f"\n[bold blue]CPU platform ({len(cpu_seeds)}):[/]")
        for s in cpu_seeds:
            confs = ", ".join(s["conferences"]) if s["conferences"] else "—"
            console.print(f"  - {s['name']}  [dim]({confs})[/]")



# ── Tracking commands ─────────────────────────────────────────────────────────

def _get_store():
    from researcharena.tracking.store import TrackingStore
    return TrackingStore()


def _short_id(run_id: str) -> str:
    return run_id[:8]


def _status_icon(status: str) -> str:
    return {"accepted": "✅", "running": "🟢", "failed": "❌", "rejected": "🔴"}.get(status, "⬜")


@main.group()
def runs():
    """List and inspect experiment runs."""
    pass


@runs.command("list")
@click.option("--limit", "-n", default=20, help="Max rows to show")
@click.option("--agent", default=None, help="Filter by agent type")
@click.option("--status", default=None, help="Filter by status")
def runs_list(limit, agent, status):
    """List recent experiment runs."""
    store = _get_store()
    rows = store.list_runs(limit=limit, agent=agent, status=status)
    if not rows:
        console.print("[yellow]No runs found. Run `researcharena ingest --all` to import existing outputs.[/]")
        return

    from rich.table import Table
    t = Table(title=f"Runs ({len(rows)} shown)")
    t.add_column("ID")
    t.add_column("Seed")
    t.add_column("Agent")
    t.add_column("Status")
    t.add_column("Score", justify="right")
    t.add_column("Time", justify="right")
    t.add_column("Cost $", justify="right")
    t.add_column("Started")

    import datetime
    for r in rows:
        icon = _status_icon(r["status"])
        started = datetime.datetime.fromtimestamp(r["started_at"]).strftime("%m-%d %H:%M") if r["started_at"] else "—"
        t.add_row(
            _short_id(r["run_id"]),
            (r["seed"] or "")[:40],
            r["agent"] or "—",
            f"{icon} {r['status']}",
            f"{r['best_score']:.1f}" if r["best_score"] is not None else "—",
            f"{r['wall_time_s']/60:.1f}m" if r["wall_time_s"] else "—",
            f"{r['cost_usd']:.3f}" if r["cost_usd"] else "—",
            started,
        )
    console.print(t)


@runs.command("show")
@click.argument("run_id")
def runs_show(run_id):
    """Show detailed info for a run."""
    store = _get_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        return

    from rich.table import Table
    import datetime

    info = Table(title=f"Run {_short_id(run['run_id'])}")
    info.add_column("Field"); info.add_column("Value")
    for k in ("run_id", "seed", "agent", "model", "platform", "status", "workspace", "decision"):
        info.add_row(k, str(run[k] or "—"))
    if run["best_score"] is not None:
        info.add_row("best_score", f"{run['best_score']:.1f}/10")
    if run["wall_time_s"]:
        info.add_row("wall_time", f"{run['wall_time_s']/60:.1f} min")
    console.print(info)

    stages = store.get_stages(run["run_id"])
    if stages:
        st = Table(title="Stages")
        st.add_column("Stage"); st.add_column("Attempt"); st.add_column("Outcome")
        st.add_column("Time (s)", justify="right"); st.add_column("Tokens", justify="right")
        st.add_column("Details")
        for s in stages:
            st.add_row(
                s["stage"], str(s["attempt"]), s["outcome"] or "—",
                f"{s['elapsed_s']:.0f}" if s["elapsed_s"] else "—",
                str(s["tokens_in"] + s["tokens_out"]),
                (s["details"] or "")[:60],
            )
        console.print(st)

    metrics = store.get_metrics(run["run_id"])
    if metrics:
        mt = Table(title="Metrics")
        mt.add_column("Key"); mt.add_column("Value", justify="right"); mt.add_column("Stage")
        for m in metrics:
            mt.add_row(m["key"], f"{m['value']:.3f}", m["stage"] or "—")
        console.print(mt)

    artifacts = store.get_artifacts(run["run_id"])
    if artifacts:
        at = Table(title="Artifacts")
        at.add_column("Name"); at.add_column("Type"); at.add_column("Size", justify="right")
        for a in artifacts:
            at.add_row(a["name"], a["type"], f"{a['size_bytes']:,} B" if a["size_bytes"] else "—")
        console.print(at)


@main.command()
@click.argument("run_id")
@click.option("--key", "-k", default=None, help="Filter to specific metric key")
def metrics(run_id, key):
    """Plot metric curves for a run."""
    store = _get_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        return

    keys = [key] if key else store.metric_keys(run["run_id"])
    if not keys:
        console.print("[yellow]No metrics recorded for this run.[/]")
        return

    try:
        import plotext as plt
        for k in keys:
            rows = store.get_metrics(run["run_id"], key=k)
            if not rows:
                continue
            xs = list(range(len(rows)))
            ys = [r["value"] for r in rows]
            plt.clf()
            plt.plot(xs, ys, marker="braille")
            plt.title(f"{k}  (run {_short_id(run['run_id'])})")
            plt.xlabel("step"); plt.ylabel(k)
            plt.show()
    except ImportError:
        from rich.table import Table
        for k in keys:
            rows = store.get_metrics(run["run_id"], key=k)
            if not rows:
                continue
            t = Table(title=k)
            t.add_column("Step"); t.add_column("Value", justify="right"); t.add_column("Stage")
            for r in rows:
                t.add_row(str(r["step"]), f"{r['value']:.4f}", r["stage"] or "—")
            console.print(t)


@main.command()
@click.argument("run_id")
@click.option("--type", "-t", "atype", default=None, help="Filter by type (idea/plan/results/paper/review/log)")
def artifacts(run_id, atype):
    """List artifacts for a run."""
    store = _get_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        return

    rows = store.get_artifacts(run["run_id"], type=atype)
    if not rows:
        console.print("[yellow]No artifacts recorded.[/]")
        return

    from rich.table import Table
    t = Table(title=f"Artifacts — {_short_id(run['run_id'])}")
    t.add_column("Name"); t.add_column("Type"); t.add_column("Size", justify="right"); t.add_column("Path")
    for a in rows:
        t.add_row(a["name"], a["type"], f"{a['size_bytes']:,} B" if a["size_bytes"] else "—", a["path"])
    console.print(t)


@main.command()
@click.argument("run_id")
@click.option("--stage", "-s", default=None, help="Filter events to this stage log filename substring")
@click.option("--tail", "-n", default=30, help="Last N events to show")
def logs(run_id, stage, tail):
    """Stream the event log for a run."""
    store = _get_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        return

    all_artifacts = store.get_artifacts(run["run_id"], type="log")
    event_files = [a for a in all_artifacts if a["name"].endswith("_events.jsonl")]
    if stage:
        event_files = [a for a in event_files if stage in a["name"]]

    if not event_files:
        console.print("[yellow]No event log files found.[/]")
        return

    import datetime
    for a in event_files:
        console.print(f"\n[bold]{a['name']}[/]")
        p = Path(a["path"])
        if not p.exists():
            console.print(f"  [red]File not found: {p}[/]")
            continue
        lines = p.read_text().splitlines()
        for line in lines[-tail:]:
            try:
                evt = json.loads(line)
                ts = evt.get("ts", 0)
                event = evt.get("event", {})
                etype = event.get("type", "")
                t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                if etype == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            console.print(f"  💬 {t} {block['text'][:120]}")
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            detail = next(iter(inp.values()), "") if inp else ""
                            console.print(f"  🔧 {t} [cyan]{name}[/] {str(detail)[:100]}")
            except Exception:
                pass


@main.command()
@click.argument("run_id_1")
@click.argument("run_id_2")
def compare(run_id_1, run_id_2):
    """Side-by-side comparison of two runs."""
    store = _get_store()
    r1 = store.get_run(run_id_1)
    r2 = store.get_run(run_id_2)
    if not r1 or not r2:
        console.print("[red]One or both runs not found.[/]")
        return

    from rich.table import Table
    t = Table(title="Comparison")
    t.add_column("Field")
    t.add_column(_short_id(r1["run_id"]))
    t.add_column(_short_id(r2["run_id"]))

    for k in ("seed", "agent", "model", "status", "decision"):
        t.add_row(k, str(r1[k] or "—"), str(r2[k] or "—"))
    t.add_row("best_score",
              f"{r1['best_score']:.1f}" if r1["best_score"] is not None else "—",
              f"{r2['best_score']:.1f}" if r2["best_score"] is not None else "—")
    t.add_row("wall_time",
              f"{r1['wall_time_s']/60:.1f}m" if r1["wall_time_s"] else "—",
              f"{r2['wall_time_s']/60:.1f}m" if r2["wall_time_s"] else "—")
    t.add_row("tokens",
              f"{r1['tokens_in']+r1['tokens_out']:,}",
              f"{r2['tokens_in']+r2['tokens_out']:,}")
    t.add_row("cost $",
              f"{r1['cost_usd']:.3f}" if r1["cost_usd"] else "—",
              f"{r2['cost_usd']:.3f}" if r2["cost_usd"] else "—")
    console.print(t)

    # Metric comparison
    keys1 = set(store.metric_keys(r1["run_id"]))
    keys2 = set(store.metric_keys(r2["run_id"]))
    all_keys = sorted(keys1 | keys2)
    if all_keys:
        mt = Table(title="Metrics")
        mt.add_column("Key")
        mt.add_column(_short_id(r1["run_id"]), justify="right")
        mt.add_column(_short_id(r2["run_id"]), justify="right")
        for k in all_keys:
            def _last(rid):
                rows = store.get_metrics(rid, key=k)
                return f"{rows[-1]['value']:.3f}" if rows else "—"
            mt.add_row(k, _last(r1["run_id"]), _last(r2["run_id"]))
        console.print(mt)


@main.command()
@click.option("--workspace", "-w", default=None, type=click.Path(exists=True),
              help="Ingest a specific workspace directory")
@click.option("--all", "ingest_all", is_flag=True, default=False,
              help="Scan all of outputs/ and ingest every workspace found")
@click.option("--outputs-dir", default="outputs", help="Root outputs directory (default: outputs/)")
def ingest(workspace, ingest_all, outputs_dir):
    """Import existing run outputs into the tracking database."""
    from researcharena.tracking.ingest import ingest_workspace as _ingest_one
    from researcharena.tracking.ingest import ingest_all as _ingest_all
    from researcharena.tracking.store import TrackingStore

    store = TrackingStore()

    if workspace:
        rid = _ingest_one(Path(workspace), store)
        if rid:
            console.print(f"[green]Ingested: {rid}  ({workspace})[/]")
        else:
            console.print(f"[yellow]No idea_XX/ dirs found in {workspace}[/]")
    elif ingest_all:
        rids = _ingest_all(Path(outputs_dir), store)
        console.print(f"[green]Ingested {len(rids)} run(s).[/]")
        for rid in rids:
            console.print(f"  {rid}")
    else:
        console.print("Specify --workspace PATH or --all")


if __name__ == "__main__":
    main()
