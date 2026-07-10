from pathlib import Path

import numpy as np

from lumix_lut_converter.cst import (
    BT709_TO_V_GAMUT,
    V_GAMUT_TO_BT709,
    build_srgb_to_vlog_adapter,
    convert_collection_srgb_cst,
    srgb_oetf,
    srgb_to_vlog,
    vlog_decode,
    vlog_encode,
)
from lumix_lut_converter.io import read_lut, write_cube
from lumix_lut_converter.lut import LUT3D, cube_grid, identity_lut


def test_vlog_reference_code_values() -> None:
    reflection = np.asarray([0.0, 0.18, 0.90])
    expected_10_bit = np.asarray([128.0, 433.0, 602.0]) / 1023.0
    np.testing.assert_allclose(vlog_encode(reflection), expected_10_bit, atol=1.1 / 1023.0)


def test_vlog_round_trip() -> None:
    linear = np.geomspace(1.0e-5, 4.0, 1000)
    np.testing.assert_allclose(vlog_decode(vlog_encode(linear)), linear, atol=2.0e-13)


def test_srgb_round_trip_and_gamut_matrix() -> None:
    encoded = np.linspace(0.0, 1.0, 1000)
    from lumix_lut_converter.cst import srgb_eotf

    np.testing.assert_allclose(srgb_oetf(srgb_eotf(encoded)), encoded, atol=2.0e-15)
    np.testing.assert_allclose(
        BT709_TO_V_GAMUT @ V_GAMUT_TO_BT709,
        np.identity(3),
        atol=2.0e-15,
    )


def test_neutral_srgb_maps_to_neutral_vlog() -> None:
    encoded = np.repeat(np.linspace(0.0, 1.0, 101)[:, None], 3, axis=1)
    actual = srgb_to_vlog(encoded)
    np.testing.assert_allclose(actual[:, 0], actual[:, 1], atol=3.0e-7)
    np.testing.assert_allclose(actual[:, 1], actual[:, 2], atol=3.0e-7)


def test_cst_collection_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "1_Everyday").mkdir(parents=True)
    (source / "5_STD-base").mkdir(parents=True)
    write_cube(source / "1_Everyday" / "Look.cube", identity_lut(5), photo_style="VLOG")
    write_cube(source / "5_STD-base" / "Std.cube", identity_lut(5), photo_style="STD")

    output = tmp_path / "output"
    summary = convert_collection_srgb_cst(source, output, output_size=5)
    assert summary.converted_count == 1
    assert summary.copied_std_count == 1
    expected = build_srgb_to_vlog_adapter(5)
    converted = read_lut(output / "1_Everyday" / "Look.cube")
    np.testing.assert_allclose(converted.table, expected.table, atol=5.1e-11)
    assert "#LUMIXPHOTOSTYLE STD" in (
        output / "1_Everyday" / "Look.cube"
    ).read_text()
