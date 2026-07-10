from pathlib import Path
from zipfile import ZipFile

from lumix_lut_converter.converter import convert_collection
from lumix_lut_converter.inverse import legalise_rgb
from lumix_lut_converter.io import write_cube
from lumix_lut_converter.lut import LUT3D, cube_grid, identity_lut


def test_collection_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "1_Everyday").mkdir(parents=True)
    (source / "5_STD-base").mkdir(parents=True)
    write_cube(source / "1_Everyday" / "Look.cube", identity_lut(5), photo_style="VLOG")
    write_cube(source / "5_STD-base" / "Std.cube", identity_lut(5), photo_style="STD")

    reference_cube = tmp_path / "reference.cube"
    grid = cube_grid(9)
    reference = LUT3D(legalise_rgb(grid).reshape(9, 9, 9, 3))
    write_cube(reference_cube, reference, photo_style="VLOG")
    reference_zip = tmp_path / "reference.zip"
    with ZipFile(reference_zip, "w") as archive:
        archive.write(reference_cube, "reference/reference.cube")

    output = tmp_path / "output"
    summary = convert_collection(source, reference_zip, output, output_size=5)
    assert summary.converted_count == 1
    assert summary.copied_std_count == 1
    assert "#LUMIXPHOTOSTYLE 709L" in (
        output / "1_Everyday" / "Look.cube"
    ).read_text()
    assert "#LUMIXPHOTOSTYLE STD" in (
        output / "5_STD-base" / "Std.cube"
    ).read_text()
    assert summary.manifest_path.exists()
