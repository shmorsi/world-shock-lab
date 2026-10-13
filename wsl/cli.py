"""Command line: ``uv run wsl --help``."""

from __future__ import annotations

from collections import Counter
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Data layer: fetch, build, report.")
events_app = typer.Typer(no_args_is_help=True, help="Event lists we construct ourselves.")
validate_app = typer.Typer(no_args_is_help=True, help="Method validation reports.")
export_app = typer.Typer(no_args_is_help=True, help="Pages and files built from results.")
live_app = typer.Typer(no_args_is_help=True,
                       help="Live pre-registered forecasts (config/live.yaml, protocol §8).")
app.add_typer(data_app, name="data")
app.add_typer(events_app, name="events")
app.add_typer(validate_app, name="validate")
app.add_typer(export_app, name="export")
app.add_typer(live_app, name="live")


@data_app.command("fetch")
def data_fetch(
    source: Annotated[list[str] | None, typer.Option(help="Limit to these groups.")] = None,
    repin: Annotated[bool, typer.Option(help="Adopt new bytes when a source changed.")] = False,
) -> None:
    """Download every artifact into data/raw and record hashes in data/manifest.json."""
    from wsl.data.fetch import fetch

    outcomes = fetch(only=set(source) if source else None, repin=repin)
    for o in outcomes:
        if o.status not in ("unchanged", "pinned_new"):
            typer.echo(f"{o.status:<20} {o.key}  {o.detail}")
    typer.echo(f"\n{dict(Counter(o.status for o in outcomes))}")
    if any(o.status == "failed" for o in outcomes):
        raise typer.Exit(code=1)


@data_app.command("prune")
def data_prune() -> None:
    """Remove manifest entries the registry no longer refers to (raw files are kept)."""
    from wsl.data.fetch import prune

    for key in prune():
        typer.echo(f"dropped {key}")


@data_app.command("build")
def data_build() -> None:
    """Parse pinned raw files -> validated parquet in data/clean -> DuckDB."""
    from wsl.data.build import build

    summary = build()
    for line in summary:
        typer.echo(line)


@data_app.command("report")
def data_report() -> None:
    """Write docs/DATA_QUALITY.md from the built data."""
    from wsl.data.quality import write_report

    path = write_report()
    typer.echo(f"wrote {path}")


@events_app.command("boc-fad")
def events_boc_fad(
    crawl: Annotated[
        bool, typer.Option(help="Re-download all pages from bankofcanada.ca (~1,100 requests).")
    ] = False,
) -> None:
    """Build data/events/boc_fad.csv from BoC press releases and validate it."""
    from wsl.shocks.boc_fad import build_fad_list

    for line in build_fad_list(crawl=crawl):
        typer.echo(line)


@app.command("replicate")
def replicate() -> None:
    """M3: replication table, placebo tests, BoC proxy validation -> docs/REPLICATION.md."""
    from wsl.analysis.report import write_report

    typer.echo(f"wrote {write_report()}")


@app.command("analyze")
def analyze() -> None:
    """M4: proxy BVAR vs LP, state dependence, Canada lens, exporters, network, robustness
    -> docs/RESULTS.md and results/m4/ (several minutes)."""
    from wsl.analysis.m4_report import write_report

    typer.echo(f"wrote {write_report()}")


@app.command("evaluate")
def evaluate(
    validation: Annotated[bool, typer.Option(help="Model selection on 2015-2018.")] = False,
    final: Annotated[bool, typer.Option(help="The single run on the 2019-2026 test set.")] = False,
    report: Annotated[bool, typer.Option(help="Write docs/ML_BENCHMARK.md.")] = False,
    rerun_reason: Annotated[str | None, typer.Option(help="Required for any second final run.")]
    = None,
) -> None:
    """M5 ML benchmark under the pre-registered protocol."""
    from wsl.ml import evaluate as ev
    from wsl.ml import report as rep

    if validation:
        out = ev.run_validation()
        typer.echo(f"validation done; candidates: {out['candidates']}")
    if final:
        out = ev.run_final(rerun_reason)
        typer.echo(f"final run done: {out['verdict']}")
    if report or validation or final:
        typer.echo(f"wrote {rep.write()}")


@validate_app.command("simulation")
def validate_simulation() -> None:
    """Monte Carlo recovery of known impulse responses -> docs/SIMULATION.md (~1 minute)."""
    from wsl.estimation.validation_report import write

    typer.echo(f"wrote {write()}")


@export_app.command("site")
def export_site(
    local: Annotated[bool, typer.Option(help="Unfiltered export for your own use only.")] = False,
) -> None:
    """Web app data -> web/public/data/ (public licence filter) or data-local/."""
    from wsl.export.site import export

    typer.echo(f"wrote {export(public=not local)}")


@export_app.command("preview")
def export_preview() -> None:
    """Milestone preview page -> web/preview/index.html (~1 minute: reruns the simulations)."""
    from wsl.export.preview import write

    typer.echo(f"wrote {write()}")


@export_app.command("live")
def export_live() -> None:
    """Only web/public/data/live.json (no research data needed; used by the live workflow)."""
    from wsl.live.export import export_live_json

    typer.echo(f"wrote {export_live_json()}")


def _live_data(download: bool):  # type: ignore[no-untyped-def]
    from wsl.live.data import config, live_cache, load
    from wsl.live.data import download as fetch_live
    from wsl.live.run import _now

    cfg, cache, now = config(), live_cache(), _now()
    if download:
        for o in fetch_live(cfg, cache):
            if o.status in ("failed", "skipped"):
                typer.echo(f"  {o.status:8} {o.key}: {o.detail}", err=True)
    return cfg, load(cfg, cache, now), now


@live_app.command("run")
def live_run(
    download: Annotated[bool, typer.Option(help="Fetch fresh data first.")] = True,
    days: Annotated[int | None, typer.Option(help="Publish forecasts this many days ahead "
                                             "(default: config).")] = None,
) -> None:
    """Fetch fresh data, score finished decisions, publish due forecasts, update live.json."""
    from wsl.live.export import export_live_json
    from wsl.live.run import forecast_due, score_due, status

    cfg, data, now = _live_data(download)
    for p in score_due(cfg, data, now) + forecast_due(cfg, data, now, days):
        typer.echo(f"wrote {p}")
    export_live_json()
    for line in status(cfg, now):
        typer.echo(line)


@live_app.command("status")
def live_status() -> None:
    """Which decisions have forecasts and scores."""
    from wsl.live.run import status

    for line in status():
        typer.echo(line)


if __name__ == "__main__":
    app()
