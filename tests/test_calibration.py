import json
from pathlib import Path

from PIL import Image

from lumix_lut_converter.calibration import generate_calibration_targets


def test_generate_calibration_targets(tmp_path: Path) -> None:
    output = tmp_path / "targets"
    manifest_path = generate_calibration_targets(
        output,
        width=1280,
        height=720,
        cube_levels=5,
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["pages"]) == 9
    assert [len(page["patches"]) for page in manifest["pages"][:4]] == [64, 64, 64, 198]
    assert all(len(page["patches"]) == 25 for page in manifest["pages"][4:])

    page = manifest["pages"][4]
    patch = page["patches"][12]
    image = Image.open(output / "targets" / page["filename"]).convert("RGB")
    x, y, width, height = patch["sample_rect"]
    assert image.getpixel((x + width // 2, y + height // 2)) == tuple(patch["rgb8"])
    assert (output / "viewer.html").exists()
    assert (output / "SHOOTING_PLAN.md").exists()
    assert (output / "checksums.sha256").exists()
