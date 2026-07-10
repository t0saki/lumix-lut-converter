from pathlib import Path

import numpy as np
from PIL import Image

from lumix_lut_converter.calibration import generate_calibration_targets
from lumix_lut_converter.capture import (
    homography_from_points,
    locate_fiducials,
    sample_page,
    target_fiducial_centres,
    transform_points,
)


def test_locate_and_sample_synthetic_capture(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest_path = generate_calibration_targets(
        package,
        width=1280,
        height=720,
        cube_levels=3,
    )
    import json

    manifest = json.loads(manifest_path.read_text())
    target = Image.open(package / "targets" / manifest["pages"][0]["filename"]).convert("RGB")
    resized = target.resize((1400, 788), Image.Resampling.BILINEAR)
    camera = Image.new("RGB", (1600, 1100), (3, 3, 3))
    camera.paste(resized, (100, 150))

    fiducials = locate_fiducials(camera)
    destination = np.asarray([(item.x, item.y) for item in fiducials])
    source = target_fiducial_centres(1280, 720)
    expected = source * np.asarray([1400 / 1280, 788 / 720]) + np.asarray([100, 150])
    np.testing.assert_allclose(destination, expected, atol=4.0)

    homography = homography_from_points(source, destination)
    np.testing.assert_allclose(transform_points(homography, source), destination, atol=1.0e-8)
    samples = sample_page(camera, homography, manifest["pages"][0], samples_per_axis=9)
    patch = next(sample for sample in samples if sample["id"] == "gray_32_130")
    np.testing.assert_allclose(patch["median_rgb8"], [130, 130, 130], atol=2.0)
