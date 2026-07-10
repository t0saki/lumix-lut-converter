from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .converter import convert_collection


app = typer.Typer(
    no_args_is_help=True,
    help="Rebase Panasonic V-Log creative LUTs for LUMIX Like709 in-camera use.",
)
console = Console()


@app.callback()
def main() -> None:
    """Panasonic LUMIX LUT base conversion tools."""


@app.command()
def convert(
    source: Path = typer.Option(..., exists=True, file_okay=False, resolve_path=True),
    reference_zip: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
    output: Path = typer.Option(..., file_okay=False, resolve_path=True),
    grid_size: int = typer.Option(33, min=2, max=65),
    std_folder: str = typer.Option("5_STD-base"),
    resume: bool = typer.Option(
        False,
        help="Resume a project-generated partial output directory.",
    ),
) -> None:
    """Convert a tiered LUT collection without modifying the source files."""
    summary = convert_collection(
        source,
        reference_zip,
        output,
        output_size=grid_size,
        std_folder=std_folder,
        resume=resume,
    )

    table = Table(title="Like709 conversion validation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in summary.metrics.to_dict().items():
        formatted = str(value) if isinstance(value, int) else f"{value:.10g}"
        table.add_row(name, formatted)
    console.print(table)
    console.print(
        f"Converted [bold]{summary.converted_count}[/bold] V-Log LUTs; "
        f"tagged [bold]{summary.copied_std_count}[/bold] Standard LUTs."
    )
    console.print(f"Output: {summary.output_root}")
    console.print(f"Manifest: {summary.manifest_path}")
