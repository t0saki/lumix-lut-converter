# lumix-lut-converter

把以 Panasonic V-Log/V-Gamut 为输入基底的创意 LUT，近似重建为可在 LUMIX
机内以 `Like709` 为 Base Photo Style 使用的 33 点 `.cube`。

项目面向“已经烘焙完成、拿不到原始调色工程”的 LUT。它不能恢复 Standard/709
管线已经裁掉的动态范围，但会尽量把基底转换误差压到可测量的最低程度。

## 方法

设 Panasonic 官方技术 LUT 为：

```text
T: V-Log/V-Gamut → V709 legal range
```

原创意 LUT 为 `L`。目标 LUT 对 Like709 full-range 输入 `x` 执行：

```text
x
→ 映射到 10-bit legal RGB（64–940）
→ 数值求解 T⁻¹
→ L
```

也就是：

```text
L_Like709(x) = L(T⁻¹(Legalise(x)))
```

实现细节：

- Panasonic 官方 `VLog_to_V709_forV35` 33 点 LUT 作为权威参考；
- 全流程 `numpy.float64`；
- 3D LUT 使用四面体插值；
- KD-tree 初值 + 有界阻尼 Gauss–Newton 数值反演；
- 10-bit legal range 使用 `64/1023` 至 `940/1023`；
- 使用 Rec.709 解码、XYZ、CIELAB 与 CIEDE2000 验证反演误差；
- 输出 33 点 `.cube` 并写入 `#LUMIXPHOTOSTYLE 709L`；
- 原始 LUT 只读，输出目录包含可复查的 `manifest.json` 和参考文件 SHA-256。

## 安装与测试

```bash
uv sync
uv run pytest
```

## 转换当前 LUMIX LUT 集合

```bash
uv run lumix-lut-converter convert \
  --source /path/to/source-luts \
  --reference-zip /path/to/VLog_to_V709_forV35_EN.zip \
  --output /path/to/output-like709
```

默认把 `5_STD-base` 视为已有 Standard 基底：不做 V709 反演，只重采样并补写
`#LUMIXPHOTOSTYLE STD`。其余 `.vlt/.cube` 均转换为 Like709 Base。

输出目录必须为空，避免覆盖用户文件。
如果上一次由本项目启动的批处理被中断，可以增加 `--resume`，它只会重写对应的
生成文件，并在完整结束后写入 manifest。

## 生成相机校准目标

```bash
uv run lumix-lut-converter generate-targets \
  --reference-zip /path/to/VLog_to_V709_forV35_EN.zip \
  --output /path/to/calibration-targets \
  --width 3840 --height 2160 --cube-levels 9
```

输出包含带 sRGB ICC 的 SDR PNG、全屏查看器、机器可读色块坐标、相机参考 LUT
和三组拍摄清单，用于标定 S9 实际的 V-Log/V709、Like709 与 Standard 管线。

## 限制

- `Like709` 机内实现不保证与 VariCam 的 V709 技术 LUT 完全相同；
- Like709 的 Knee、画质参数、白平衡等应保持统一；
- 原 LUT 已包含的裁切、色域压缩和 17 点采样无法逆转；
- ΔE00 验证衡量的是官方技术 LUT 的数值反演精度，不代表相机实拍的最终误差；
- 若未来有同机位、同曝光、同白平衡的大规模 Standard/V-Log 配对色卡数据，可增加
  相机实测的 Standard Base 后端。
