from __future__ import annotations

import click
import uvicorn
from rich.console import Console

from tq.core.config import ensure_dirs, load_global_config
from tq.core.download import pull_model
from tq.core.hardware import detect_hardware
from tq.core.recommend import recommend
from tq.server.app import set_engine
from tq.server.inference import InferenceEngine


@click.command()
@click.argument("model", required=False)
@click.option("--coding", is_flag=True, help="Auto-select best coding model")
@click.option("--chat", is_flag=True, help="Auto-select best chat model")
@click.option("--context", type=int, default=None, help="Target context length")
@click.option("--bits", type=int, default=None, help="Force TQ bit-width (3 or 4)")
@click.option("--port", type=int, default=None, help="Server port")
@click.option("--host", default=None, help="Server host")
@click.option("--verbose", is_flag=True, help="Show detailed TQ configuration")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--cpu", is_flag=True, help="Force CPU-only mode")
def start(
    model: str | None,
    coding: bool,
    chat: bool,
    context: int | None,
    bits: int | None,
    port: int | None,
    host: str | None,
    verbose: bool,
    json_output: bool,
    cpu: bool,
) -> None:
    """Start serving a model with TurboQuant."""
    console = Console()
    config = load_global_config()
    ensure_dirs()

    # Step 1: Detect hardware
    console.print("[bold blue]Step 1/5[/bold blue] Detecting hardware...")
    hw = detect_hardware()
    if cpu:
        from tq.core.hardware import HardwareProfile

        hw = HardwareProfile(
            gpu_name="No GPU (forced CPU)",
            gpu_vram_mb=0,
            gpu_compute_cap="",
            system_ram_mb=hw.system_ram_mb,
            cpu_name=hw.cpu_name,
            cpu_cores=hw.cpu_cores,
            os=hw.os,
            arch=hw.arch,
            backend="cpu",
            available_vram_mb=0,
        )
    console.print(f"  GPU: {hw.gpu_name} ({hw.gpu_vram_mb:,} MB VRAM)")

    # Step 2: Recommend model
    use_case = "coding" if coding else ("chat" if chat else "general")
    console.print(f"[bold blue]Step 2/5[/bold blue] Recommending model (use_case={use_case})...")
    min_ctx = context or 16384
    rec = recommend(hardware=hw, use_case=use_case, min_context=min_ctx)

    model_id = model or rec.model_id
    console.print(f"  Selected: [green]{model_id}[/green]")
    if verbose:
        console.print(
            f"  TQ: keys={rec.tq_config.key_bits}-bit,"
            f" values={rec.tq_config.value_bits}-bit"
            f" ({rec.tq_config.compression_ratio}x)"
        )
        console.print(f"  Max context: {rec.max_context_tq:,} tokens")

    # Step 3: Download model if needed
    console.print("[bold blue]Step 3/5[/bold blue] Ensuring model is downloaded...")
    pull_model(model_id.split("/")[-1].lower().replace("-", "-"))

    # Step 4: Load model
    console.print("[bold blue]Step 4/5[/bold blue] Loading model...")
    tq_config = rec.tq_config
    if bits:
        from tq.core.turbo import TQConfig

        tq_config = TQConfig(
            key_bits=bits,
            value_bits=bits,
            key_method=tq_config.key_method,
            value_method=tq_config.value_method,
            outlier_channels=tq_config.outlier_channels,
            outlier_bits=tq_config.outlier_bits,
            compression_ratio=16.0 / bits,
            estimated_quality="lossless" if bits == 4 else "near-lossless",
        )

    device = "cpu" if cpu else "auto"
    engine = InferenceEngine(
        model_name=model_id,
        tq_config=tq_config,
        device=device,
    )
    engine.load()
    set_engine(engine)
    console.print("  Model loaded successfully")

    # Step 5: Start server
    serve_host = host or config.host
    serve_port = port or config.port
    console.print(
        f"[bold blue]Step 5/5[/bold blue] Starting server at http://{serve_host}:{serve_port}"
    )
    console.print("  OpenAI-compatible API ready")
    console.print("  [dim]Press Ctrl+C to stop[/dim]")

    try:
        uvicorn.run(
            "tq.server.app:app",
            host=serve_host,
            port=serve_port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
