from pathlib import Path

import numpy as np

from lumix_lut_converter.io import parse_lut_text, read_lut, write_cube
from lumix_lut_converter.lut import identity_lut


def test_vlt_12_bit_normalisation() -> None:
    text = "LUT_3D_SIZE 2\n" + "\n".join(
        ["0 2048 4095"] * 8
    )
    lut = parse_lut_text(text, ".vlt")
    np.testing.assert_allclose(lut.table[0, 0, 0], [0.0, 2048 / 4095, 1.0])


def test_cube_round_trip_and_lumix_tag(tmp_path: Path) -> None:
    path = tmp_path / "identity.cube"
    write_cube(path, identity_lut(5), photo_style="709L")
    text = path.read_text()
    assert "#LUMIXPHOTOSTYLE 709L" in text
    loaded = read_lut(path)
    np.testing.assert_allclose(loaded.table, identity_lut(5).table, atol=5.1e-11)
