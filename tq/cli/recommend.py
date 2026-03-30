from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from tq.core.hardware import detect_hardware
from tq.core.recommend import recommend


@click.command()
@click.option("--coding", is_flag=True, help="Recommend for coding use case")
@click.option("--chat", is_flag=True, help="Recommend for chat use case")
@click.option("--context", type=int, default=16384, help="Minimum desired context length")
def recommend_cmd(coding: bool, chat: bool, context: int) -> None:
    """Recommend models for your hardware."""
    console = Console()
    use_case = "coding" if coding else ("chat" if chat else "general")

    hw = detect_hardware()
    rec = recommend(hardware=hw, use_case=use_case, min_context=context)

    table = Table(title=f"Recommended Model ({use_case})")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", rec.model_id)
    table.add_row("Quant", rec.quant)
    table.add_row("Size", f"{rec.model_size_mb:,} MB")
    table.add_row("Max Context (vanilla)", f"{rec.max_context_vanilla:,} tokens")
    table.add_row("Max Context (TurboQuant)", f"{rec.max_context_tq:,} tokens")
    table.add_row(
        "TQ Config", f"keys={rec.tq_config.key_bits}-bit, values={rec.tq_config.value_bits}-bit"
    )
    table.add_row("Compression", f"{rec.tq_config.compression_ratio}x")
    table.add_row("Quality", rec.tq_config.estimated_quality)
    table.add_row("Est. Speed", f"{rec.estimated_tok_s:.0f} tok/s")
    table.add_row("Fit Score", f"{rec.fit_score:.2f}")

    console.print(table)

    if rec.alternatives:
        console.print("\n[bold]Alternatives:[/bold]")
        for alt in rec.alternatives:
            console.print(
                f"  {alt.model_id} — {alt.max_context_tq:,} ctx, "
                f"{alt.tq_config.compression_ratio}x, score {alt.fit_score:.2f}"
            )
