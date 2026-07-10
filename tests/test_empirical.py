from pathlib import Path

import numpy as np

from lumix_lut_converter.empirical import convert_collection_with_empirical_adapter
from lumix_lut_converter.io import read_lut, write_cube
from lumix_lut_converter.lut import LUT3D, cube_grid, identity_lut


def test_empirical_collection_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "1_Everyday").mkdir(parents=True)
    (source / "5_STD-base").mkdir(parents=True)
    write_cube(source / "1_Everyday" / "Look.cube", identity_lut(5), photo_style="VLOG")
    write_cube(source / "5_STD-base" / "Std.cube", identity_lut(5), photo_style="STD")

    grid = cube_grid(5)
    adapter = LUT3D(np.power(grid, 0.9).reshape(5, 5, 5, 3))
    adapter_path = tmp_path / "adapter.cube"
    write_cube(adapter_path, adapter, photo_style="STD")

    output = tmp_path / "output"
    summary = convert_collection_with_empirical_adapter(
        source, adapter_path, output, output_size=5
    )
    assert summary.converted_count == 1
    assert summary.copied_std_count == 1
    converted = read_lut(output / "1_Everyday" / "Look.cube")
    np.testing.assert_allclose(converted.table, adapter.table, atol=5.1e-11)
    assert "#LUMIXPHOTOSTYLE STD" in (
        output / "1_Everyday" / "Look.cube"
    ).read_text()
