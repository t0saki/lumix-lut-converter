from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

from .io import read_reference_zip, write_cube
from .lut import LUT3D


RGB8 = tuple[int, int, int]


@dataclass(frozen=True)
class Patch:
    patch_id: str
    rgb: RGB8
    rect: tuple[int, int, int, int]
    sample_rect: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.patch_id,
            "rgb8": list(self.rgb),
            "rect": list(self.rect),
            "sample_rect": list(self.sample_rect),
        }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _sample_rect(rect: tuple[int, int, int, int], fraction: float = 0.55) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    sample_width = max(2, int(width * fraction))
    sample_height = max(2, int(height * fraction))
    return (
        x + (width - sample_width) // 2,
        y + (height - sample_height) // 2,
        sample_width,
        sample_height,
    )


def _grid_rectangles(
    width: int,
    height: int,
    rows: int,
    columns: int,
) -> list[tuple[int, int, int, int]]:
    margin_x = max(60, int(width * 0.045))
    header = max(90, int(height * 0.085))
    footer = max(110, int(height * 0.075))
    gap = max(8, int(min(width, height) * 0.007))
    available_width = width - 2 * margin_x
    available_height = height - header - footer
    patch_size = min(
        (available_width - gap * (columns - 1)) // columns,
        (available_height - gap * (rows - 1)) // rows,
    )
    grid_width = patch_size * columns + gap * (columns - 1)
    grid_height = patch_size * rows + gap * (rows - 1)
    origin_x = (width - grid_width) // 2
    origin_y = header + (available_height - grid_height) // 2
    return [
        (
            origin_x + column * (patch_size + gap),
            origin_y + row * (patch_size + gap),
            patch_size,
            patch_size,
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _anchors(width: int, height: int) -> list[Patch]:
    values = (0, 64, 128, 192, 255)
    size = max(42, int(height * 0.033))
    gap = max(14, int(size * 0.35))
    total = len(values) * size + (len(values) - 1) * gap
    start_x = (width - total) // 2
    y = height - size - max(20, int(height * 0.018))
    return [
        Patch(
            patch_id=f"anchor_{value:03d}",
            rgb=(value, value, value),
            rect=(start_x + index * (size + gap), y, size, size),
            sample_rect=_sample_rect((start_x + index * (size + gap), y, size, size)),
        )
        for index, value in enumerate(values)
    ]


def _draw_fiducials(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    size = max(30, int(min(width, height) * 0.026))
    inset = max(20, size // 2)
    for x, y in (
        (inset, inset),
        (width - inset - size, inset),
        (inset, height - inset - size),
        (width - inset - size, height - inset - size),
    ):
        draw.rectangle((x, y, x + size, y + size), fill=(245, 245, 245))
        inner = size // 3
        draw.rectangle(
            (x + inner, y + inner, x + size - inner, y + size - inner),
            fill=(8, 8, 8),
        )


def _save_page(
    path: Path,
    *,
    width: int,
    height: int,
    title: str,
    patch_values: list[tuple[str, RGB8]],
    rows: int,
    columns: int,
) -> dict[str, object]:
    image = Image.new("RGB", (width, height), (48, 48, 48))
    draw = ImageDraw.Draw(image)
    _draw_fiducials(draw, width, height)
    title_font = _font(max(26, int(height * 0.024)))
    draw.text(
        (width // 2, max(20, int(height * 0.022))),
        title,
        fill=(210, 210, 210),
        font=title_font,
        anchor="ma",
    )

    rectangles = _grid_rectangles(width, height, rows, columns)
    patches: list[Patch] = []
    for (patch_id, rgb), rect in zip(patch_values, rectangles, strict=True):
        x, y, patch_width, patch_height = rect
        draw.rectangle(
            (x, y, x + patch_width - 1, y + patch_height - 1),
            fill=rgb,
        )
        patches.append(Patch(patch_id, rgb, rect, _sample_rect(rect)))

    anchors = _anchors(width, height)
    for patch in anchors:
        x, y, patch_width, patch_height = patch.rect
        draw.rectangle(
            (x, y, x + patch_width - 1, y + patch_height - 1),
            fill=patch.rgb,
        )

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(path, format="PNG", compress_level=6, icc_profile=profile)
    return {
        "filename": path.name,
        "title": title,
        "rows": rows,
        "columns": columns,
        "patches": [patch.to_dict() for patch in patches],
        "anchors": [patch.to_dict() for patch in anchors],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _viewer_html(pages: list[dict[str, object]]) -> str:
    filenames = [f'targets/{page["filename"]}' for page in pages]
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>LUMIX calibration targets</title>
<style>html,body{{margin:0;background:#000;width:100%;height:100%;overflow:hidden}}
img{{width:100vw;height:100vh;object-fit:contain;display:block}}</style></head>
<body><img id=\"target\"><script>
const pages={json.dumps(filenames)}; let index=0;
const image=document.getElementById('target');
function show(i){{index=(i+pages.length)%pages.length;image.src=pages[index];}}
addEventListener('keydown',e=>{{if(['ArrowRight',' ','PageDown'].includes(e.key))show(index+1);
if(['ArrowLeft','PageUp'].includes(e.key))show(index-1);
if(e.key==='f')document.documentElement.requestFullscreen();}});show(0);
</script></body></html>"""


def _serve_command() -> str:
    return """#!/bin/zsh
set -e
cd "$(dirname "$0")"
PORT=8765
python3 -m http.server "$PORT" --bind 127.0.0.1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 1
open "http://127.0.0.1:${PORT}/viewer.html"
wait "$SERVER_PID"
"""


def _shooting_plan() -> str:
    return """# LUMIX S9 校准拍摄清单

## 屏幕

- 使用 SDR sRGB/Rec.709 模式，关闭 HDR、动态对比度、节能和自动亮度。
- 固定亮度并预热约 20 分钟；双击 `serve.command` 启动本地网页，按 `f` 全屏，方向键换页。
- 相机垂直正对屏幕并稍微虚焦，避免 QD-OLED 子像素产生摩尔纹。

## 相机固定设置

- 三脚架；M档；手动对焦；固定构图。
- ISO 640；固定光圈；快门建议 1/10 秒或更慢，并让最亮色块不过曝。
- 白平衡固定 6500 K，不能使用 AWB。
- sRGB JPEG Fine + RAW；所有画质微调保持默认/归零。
- 基础组的全部页面使用同一曝光；不得按页面单独调快门。
- 三种 Photo Style 之间不得改变曝光、白平衡、焦点和构图。

## 三组照片

按 `manifest.json` 的页面顺序，每页拍一张，共 13 张/组。

### A_NATIVE_VLOG

- Base Photo Style：V-Log
- 不套 LUT；直接保留原生 V-Log 灰片。

### B_LIKE709

- Photo Style：Like709
- 不套 LUT
- Knee：Off（不能使用 Auto）

### C_STANDARD

- Photo Style：Standard
- 不套 LUT

建议每组放到单独文件夹，或记录每组第一张和最后一张文件名。

## 可选阴影加强组

完成基础组后，把曝光统一提高 +2 EV，按相同顺序重新拍完整三组；不要逐页调曝光。
过曝色块会在拟合时自动丢弃，剩余色块用于增强暗部精度。不要更改白平衡、
焦点、构图和 Photo Style 参数。

## HDR/自然照片

现在先不用拍。收到校准图后会先生成 v2 LUT；随后再用 3–5 张 HDR 照片以及
日光、暖色室内、混合光真实场景比较“原 V-Log LUT”和“v2 低 ISO LUT”。
"""


def generate_calibration_targets(
    output_root: Path,
    *,
    width: int = 3840,
    height: int = 2160,
    cube_levels: int = 9,
    reference_zip: Path | None = None,
) -> Path:
    if cube_levels < 3 or cube_levels > 17:
        raise ValueError("cube_levels must be between 3 and 17")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    targets_root = output_root / "targets"
    targets_root.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, object]] = []
    full_gray = np.rint(np.linspace(0, 255, 64)).astype(int)
    shadow_gray = np.rint(np.linspace(0, 96, 64)).astype(int)
    highlight_gray = np.rint(np.linspace(160, 255, 64)).astype(int)
    for filename, title, values in (
        ("00_gray_full.png", "64-step grayscale 0-255", full_gray),
        ("01_gray_shadows.png", "64-step shadow grayscale 0-96", shadow_gray),
        ("02_gray_highlights.png", "64-step highlight grayscale 160-255", highlight_gray),
    ):
        patches = [
            (f"gray_{index:02d}_{value:03d}", (int(value),) * 3)
            for index, value in enumerate(values)
        ]
        pages.append(
            _save_page(
                targets_root / filename,
                width=width,
                height=height,
                title=title,
                patch_values=patches,
                rows=4,
                columns=16,
            )
        )

    ramp_values = np.rint(np.linspace(0, 255, 33)).astype(int)
    ramp_bases: tuple[tuple[str, RGB8], ...] = (
        ("R", (1, 0, 0)),
        ("G", (0, 1, 0)),
        ("B", (0, 0, 1)),
        ("C", (0, 1, 1)),
        ("M", (1, 0, 1)),
        ("Y", (1, 1, 0)),
    )
    ramp_patches: list[tuple[str, RGB8]] = []
    for name, base in ramp_bases:
        for index, value in enumerate(ramp_values):
            rgb = tuple(int(channel * value) for channel in base)
            ramp_patches.append((f"{name}_{index:02d}_{value:03d}", rgb))
    pages.append(
        _save_page(
            targets_root / "03_rgb_cmy_ramps.png",
            width=width,
            height=height,
            title="RGB and CMY ramps - 33 levels",
            patch_values=ramp_patches,
            rows=6,
            columns=33,
        )
    )

    cube_values = np.rint(np.linspace(0, 255, cube_levels)).astype(int)
    for blue_index, blue in enumerate(cube_values):
        cube_patches: list[tuple[str, RGB8]] = []
        for green_index, green in enumerate(cube_values):
            for red_index, red in enumerate(cube_values):
                cube_patches.append(
                    (
                        f"cube_b{blue_index:02d}_g{green_index:02d}_r{red_index:02d}",
                        (int(red), int(green), int(blue)),
                    )
                )
        pages.append(
            _save_page(
                targets_root / f"{10 + blue_index:02d}_cube_b{int(blue):03d}.png",
                width=width,
                height=height,
                title=f"RGB cube {cube_levels}^3 - blue={int(blue)}",
                patch_values=cube_patches,
                rows=cube_levels,
                columns=cube_levels,
            )
        )

    reference: dict[str, object] | None = None
    if reference_zip is not None:
        camera_luts = output_root / "camera_luts"
        camera_luts.mkdir(parents=True, exist_ok=True)
        reference_lut, entry = read_reference_zip(reference_zip.resolve())
        tagged_reference = LUT3D(reference_lut.table, title="CAL_REF_V709")
        reference_path = camera_luts / "CAL_REF_V709.cube"
        write_cube(
            reference_path,
            tagged_reference,
            photo_style="VLOG",
            comments=(
                "# Calibration-only Panasonic official VLog_to_V709 reference",
                f"# ZIP entry: {entry}",
            ),
        )
        reference = {
            "source_zip": str(reference_zip.resolve()),
            "source_entry": entry,
            "camera_lut": str(reference_path.relative_to(output_root)),
            "sha256": _sha256(reference_path),
        }

    manifest = {
        "format_version": 1,
        "purpose": "Paired LUMIX S9 native V-Log, Like709 and Standard pipeline calibration",
        "encoding": "8-bit sRGB PNG with embedded sRGB ICC profile; SDR only",
        "canvas": {"width": width, "height": height, "background_rgb8": [48, 48, 48]},
        "cube_levels": [int(value) for value in cube_values],
        "capture_groups": [
            "A_NATIVE_VLOG: V-Log, no LUT",
            "B_LIKE709: Like709, no LUT, Knee Off",
            "C_STANDARD: Standard, no LUT",
        ],
        "reference": reference,
        "pages": pages,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "viewer.html").write_text(_viewer_html(pages), encoding="utf-8")
    serve_path = output_root / "serve.command"
    serve_path.write_text(_serve_command(), encoding="utf-8")
    serve_path.chmod(0o755)
    (output_root / "SHOOTING_PLAN.md").write_text(_shooting_plan(), encoding="utf-8")
    checksums = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", ".DS_Store"}:
            checksums.append(f"{_sha256(path)}  {path.relative_to(output_root)}")
    (output_root / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return manifest_path
