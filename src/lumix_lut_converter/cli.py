from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .calibration import generate_calibration_targets
from .capture import analyze_calibration_captures
from .converter import convert_collection
from .empirical import convert_collection_with_empirical_adapter, fit_empirical_std_to_vlog


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


@app.command("generate-targets")
def generate_targets(
    output: Path = typer.Option(..., file_okay=False, resolve_path=True),
    reference_zip: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Panasonic VLog_to_V709 ZIP; also creates a camera reference LUT.",
    ),
    width: int = typer.Option(3840, min=1280),
    height: int = typer.Option(2160, min=720),
    cube_levels: int = typer.Option(9, min=3, max=17),
) -> None:
    """Generate SDR display targets and a machine-readable calibration manifest."""
    manifest = generate_calibration_targets(
        output,
        width=width,
        height=height,
        cube_levels=cube_levels,
        reference_zip=reference_zip,
    )
    console.print(f"Calibration package: {manifest.parent}")
    console.print(f"Manifest: {manifest}")


@app.command("analyze-captures")
def analyze_captures(
    captures: Path = typer.Option(..., exists=True, file_okay=False, resolve_path=True),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
    output: Path = typer.Option(..., file_okay=False, resolve_path=True),
) -> None:
    """Locate calibration targets and extract paired camera RGB samples."""
    summary = analyze_calibration_captures(captures, manifest, output)
    console.print(
        f"Located [bold]{summary.frame_count}[/bold] frames across "
        f"[bold]{summary.page_count}[/bold] pages."
    )
    console.print(f"Extracted samples: [bold]{summary.sample_count}[/bold]")
    console.print(f"Report: {summary.report_path}")
    console.print(f"Samples: {summary.samples_path}")


@app.command("fit-captures")
def fit_captures(
    samples: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
    output: Path = typer.Option(..., file_okay=False, resolve_path=True),
    grid_size: int = typer.Option(33, min=2, max=65),
    smoothing_sigma: float = typer.Option(0.15, min=0.0, max=2.0),
) -> None:
    """Fit a temporary Standard-to-V-Log adapter from extracted camera samples."""
    summary = fit_empirical_std_to_vlog(
        samples,
        manifest,
        output,
        output_size=grid_size,
        smoothing_sigma=smoothing_sigma,
    )
    console.print(f"Adapter: {summary.adapter_path}")
    console.print(f"Fit report: {summary.report_path}")
    console.print(
        f"Samples kept: [bold]{summary.training_samples}[/bold] training, "
        f"[bold]{summary.validation_samples}[/bold] validation"
    )


@app.command("convert-empirical")
def convert_empirical(
    source: Path = typer.Option(..., exists=True, file_okay=False, resolve_path=True),
    adapter: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
    output: Path = typer.Option(..., file_okay=False, resolve_path=True),
    grid_size: int = typer.Option(33, min=2, max=65),
    std_folder: str = typer.Option("5_STD-base"),
) -> None:
    """Rebase a V-Log LUT collection to Standard with a measured camera adapter."""
    summary = convert_collection_with_empirical_adapter(
        source,
        adapter,
        output,
        output_size=grid_size,
        std_folder=std_folder,
    )
    console.print(
        f"Converted [bold]{summary.converted_count}[/bold] V-Log LUTs; "
        f"tagged [bold]{summary.copied_std_count}[/bold] existing Standard LUTs."
    )
    console.print(f"Output: {summary.output_root}")
    console.print(f"Manifest: {summary.manifest_path}")
