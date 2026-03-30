from __future__ import annotations

import click
import httpx
from rich.console import Console


@click.command()
@click.option("--port", type=int, default=8000, help="Server port")
def stop(port: int) -> None:
    """Stop the running server."""
    console = Console()
    try:
        httpx.post(f"http://127.0.0.1:{port}/shutdown", timeout=3)
        console.print("[green]Server shutting down[/green]")
    except httpx.ConnectError:
        console.print(f"[yellow]No server running on port {port}[/yellow]")


@click.command()
@click.option("--port", type=int, default=8000, help="Server port")
def status(port: int) -> None:
    """Show server status and TQ metrics."""
    console = Console()
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/tq/status", timeout=3)
        resp.raise_for_status()
        data = resp.json()

        console.print(f"[bold]Model:[/bold] {data.get('model', 'unknown')}")
        tq = data.get("turboquant", {})
        if tq.get("enabled"):
            console.print(
                f"[bold]TurboQuant:[/bold]"
                f" keys={tq.get('key_bits')}-bit,"
                f" values={tq.get('value_bits')}-bit"
            )
            console.print(f"[bold]Compression:[/bold] {tq.get('compression_ratio')}x")
        else:
            console.print("[bold]TurboQuant:[/bold] disabled")

        perf = data.get("performance", {})
        console.print(f"[bold]Tokens generated:[/bold] {perf.get('tokens_generated', 0)}")
        console.print(f"[bold]Uptime:[/bold] {perf.get('uptime_seconds', 0)}s")

    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to server on port {port}. Is it running?[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
