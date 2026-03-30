from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from tq.core.download import list_models, pull_model, remove_model


@click.command()
@click.argument("model")
@click.option("--quant", default="Q4_K_M", help="Quantization format")
def pull(model: str, quant: str) -> None:
    """Download a model."""
    pull_model(model, quant)


@click.command(name="list")
def list_cmd() -> None:
    """List installed models."""
    console = Console()
    models = list_models()

    if not models:
        console.print(
            "[yellow]No models installed."
            " Use [bold]tq pull <model>[/bold] to download one.[/yellow]"
        )
        return

    table = Table(title="Installed Models")
    table.add_column("Model ID", style="bold")
    table.add_column("Shortname")
    table.add_column("Quant")
    table.add_column("Size")
    table.add_column("Path")

    for m in models:
        size_gb = m.size_bytes / (1024**3)
        table.add_row(m.model_id, m.shortname, m.quant, f"{size_gb:.1f} GB", m.gguf_path)

    console.print(table)


@click.command()
@click.argument("model")
@click.confirmation_option(prompt="Are you sure you want to remove this model?")
def remove(model: str) -> None:
    """Remove an installed model."""
    console = Console()
    if remove_model(model):
        console.print(f"[green]Removed {model}[/green]")
    else:
        console.print(f"[red]Model {model} not found in installed models[/red]")
