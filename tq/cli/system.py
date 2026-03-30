from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from tq.core.hardware import detect_hardware


@click.command()
def system() -> None:
    """Show detected hardware profile."""
    console = Console()
    hw = detect_hardware()

    table = Table(title="Hardware Profile", show_header=False)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    table.add_row("GPU", hw.gpu_name)
    table.add_row("VRAM", f"{hw.gpu_vram_mb:,} MB ({hw.available_vram_mb:,} MB available)")
    table.add_row("System RAM", f"{hw.system_ram_mb:,} MB")
    table.add_row("CPU", f"{hw.cpu_name} ({hw.cpu_cores} cores)")
    table.add_row("OS", hw.os)
    table.add_row("Arch", hw.arch)
    table.add_row("Backend", hw.backend.upper())
    if hw.gpu_compute_cap:
        table.add_row("Compute Cap", hw.gpu_compute_cap)

    console.print(table)
