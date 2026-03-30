from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from tq.core.config import (
    MODELS_DIR,
    InstalledModel,
    ensure_dirs,
    find_installed_model,
    load_installed_models,
    register_installed_model,
    unregister_installed_model,
)
from tq.core.models import resolve_model


def pull_model(shortname: str, quant: str = "Q4_K_M") -> Path:
    ensure_dirs()
    console = Console()

    model_cfg = resolve_model(shortname)
    repo_id = model_cfg.model_id
    filename = _gguf_filename(repo_id, quant)

    existing = find_installed_model(model_cfg.model_id)
    if existing and Path(existing.gguf_path).exists():
        console.print(
            f"[green]Model {model_cfg.model_id} already installed at {existing.gguf_path}[/green]"
        )
        return Path(existing.gguf_path)

    console.print(f"[bold]Downloading[/bold] {model_cfg.model_id} ({quant})...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=None)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            resume_download=True,
        )
        progress.update(task, completed=1, total=1)

    path = Path(path)
    size_bytes = path.stat().st_size

    from datetime import datetime

    register_installed_model(
        InstalledModel(
            model_id=model_cfg.model_id,
            shortname=shortname,
            gguf_path=str(path),
            quant=quant,
            size_bytes=size_bytes,
            downloaded_at=datetime.now().isoformat(),
        )
    )

    console.print(f"[green]Downloaded[/green] {model_cfg.model_id} → {path}")
    return path


def list_models() -> list[InstalledModel]:
    return load_installed_models()


def remove_model(shortname: str) -> bool:
    model_cfg = resolve_model(shortname)
    removed = unregister_installed_model(model_cfg.model_id)
    if not removed:
        return False

    existing = find_installed_model(model_cfg.model_id)
    if existing:
        path = Path(existing.gguf_path)
        if path.exists():
            path.unlink()

    return True


def _gguf_filename(repo_id: str, quant: str) -> str:
    name = repo_id.split("/")[-1]
    return f"{name}-{quant}.gguf"
