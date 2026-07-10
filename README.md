# LUMIX LUT 色彩探索 / LUMIX LUT Color Lab

用公开的色彩科学模型理解、验证并转换 Panasonic LUMIX 机内 LUT。

项目最实用的能力，是把以 **V-Log / V-Gamut** 为输入的创意 LUT，重建为可在
**Standard Photo Style + sRGB + 低 ISO** 下使用的 33 点 `.cube`。这样可以保留原 LUT
的创意映射，同时避免为了加载 V-Log Base LUT 而被相机限制在较高的最低 ISO。

仓库与 Python 包名继续使用 `lumix-lut-converter`，以保持已有脚本和链接兼容；项目定位
已经从单一“LUT 转换器”扩展为 LUMIX LUT 管线的开源研究、实验和最佳实践集合。

## 结论先行：推荐方案

相机端使用：

- Base Photo Style：`Standard`；
- JPEG 色彩空间：`sRGB`；
- ISO：按 Standard 的正常范围使用，例如 ISO 100；
- 白平衡、曝光和其他画质参数：按实际拍摄需要设置。

电脑端将 V-Log LUT 转为 Standard/sRGB Base：

```bash
git clone https://github.com/t0saki/lumix-lut-converter.git
cd lumix-lut-converter
uv sync

uv run lumix-lut-converter convert-cst \
  --source /path/to/source-luts \
  --output /path/to/output-standard-srgb
```

`source-luts` 按 `分类目录/LUT 文件` 组织，可以混合 `.cube` 和 `.vlt`；输出目录应为空。
默认输出 33 点 `.cube`，并写入：

```text
#LUMIXPHOTOSTYLE STD
```

生成结果可直接交给 LUMIX Lab 或通过 SD 卡导入相机。输出目录还会包含：

- `_technical/sRGB_to_VLog_VGamut_33.cube`：不含创意风格的技术适配器；
- `manifest.json`：输入、输出、参数和文件校验信息；
- 已经属于 Standard Base 的 LUT：不重复转换，只统一采样和标记。

> 这是基于公开模型的高精度“基底重映射”，不是严格意义的无损逆变换。原 LUT 已有的
> 裁切、色域压缩和低网格采样无法恢复；LUMIX Standard 的内部 tone curve 与 LUT 插入
> 位置也没有完整公开，因此重要用途仍建议做一组固定机位的 V-Log / Standard A/B。

## 原理

Panasonic 已公开 V-Log 编码函数、V-Gamut 色度坐标以及 V-Gamut 与 BT.709 的转换矩阵。
而 sRGB 与 BT.709 使用相同的 RGB 原色和 D65 白点，所以可以建立解析适配器：

```text
Standard/sRGB LUT 输入
→ sRGB EOTF（解码到线性光）
→ 线性 BT.709/sRGB 转 V-Gamut
→ Panasonic V-Log OETF
→ 原 V-Log 创意 LUT
```

若原始创意 LUT 为 `L`，解析适配器为 `F`，新 LUT 就是：

```text
L_STD(x) = L(F(x))
```

这个方法不需要先拍摄整套屏幕色卡，也不需要主观拟合 Panasonic Standard。项目以
`float64` 计算、四面体插值和 3D LUT 合成输出最终结果。33 点技术适配器与解析公式在
20 万个随机 RGB 样本上的平均最大通道误差约为 **0.23 个 12-bit code value**，99% 样本
低于 **1.27 code values**。

Panasonic 官方资料：
[V-Log / V-Gamut Reference Manual](https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf)

## 为什么推荐 sRGB

我们比较和讨论过 Adobe RGB，但当前最佳实践仍选 sRGB：

- LUMIX LUT、LUMIX Lab 和绝大多数浏览/分享链路以 sRGB/BT.709 原色为共同基础；
- sRGB 与 BT.709 原色一致，解析矩阵链路明确，不需要额外的色域假设；
- 相机生成的 sRGB JPEG 更容易被软件一致解释；
- Adobe RGB 只有在整条拍摄、标记、显示和导出链路都明确色彩管理时才有意义，且不会
  自动带来更准确的 V-Log LUT 还原。

这里的选择并不声称 Panasonic Standard 的内部渲染“等于 sRGB 曲线”。它是针对 LUT
输入与 JPEG 输出链路最可验证、兼容性最好的工作模型。

## 实拍研究得到什么

项目不是只停留在公式推演。我们生成了 4K 定位色卡，完成过 V-Log / Standard 配对
拍摄、RAW 线性值测量和真实创意 LUT A/B。主要发现包括：

- 同光圈快门、两边都标 ISO 640 时，Standard RAW 的线性信号约为 V-Log 的
  **6.08 倍（约 2.60 EV）**；ISO 数字不能脱离 Photo Style 直接比较。
- V-Log ISO 640 与 Standard ISO 100 在实际光度响应上大致处于同一量级，但现有一组
  非严格控制 A/B 仍有约 **0.33 EV** 的归一化差异，不能把“6.4 倍”当作精确常数。
- 一组真实 Fuji Classic Negative 创意 LUT A/B 中，解析版本与原 V-Log 版本的综合色彩
  已经非常接近；用一条共享 tone curve 对齐亮度后，平均最大通道残差约
  **3.53/255**，95% 低于 **7.37/255**。主要剩余差异是亮度/tone，而非明显色偏。
- 实拍色卡拟合可以很好地学习特定设置，但也会把 ISO、曝光、白平衡和机内 tone curve
  一并学进去；因此它更适合作为研究和验证工具，而不是通用转换的首选。
- 实机测试中，LUMIX S9 可导入不高于 33 点的 `.cube` / `.vlt`，65 点被拒绝；LUMIX
  Lab 的兼容目标应使用 33 点 `.cube`。65 点适合桌面软件或数值研究。

完整的实验条件、V-Log 公式、矩阵、RAW 比值、色卡拟合误差和 A/B 指标见：
[研究发现与方法说明](docs/research-findings.zh-CN.md)。

## 其他探索工具

### V-Log → Like709 Base

保留了最初的 Like709 路线。它对 Panasonic 官方 `VLog_to_V709_forV35` 33 点技术 LUT
做高精度数值反演，再与创意 LUT 合成：

```bash
uv run lumix-lut-converter convert \
  --source /path/to/source-luts \
  --reference-zip /path/to/VLog_to_V709_forV35_EN.zip \
  --output /path/to/output-like709
```

这条路线适合研究、交叉验证或确实想使用 Like709 Base 的用户；低 ISO 使用场景优先选择
`convert-cst`。

### 生成与分析实拍色卡

```bash
uv run lumix-lut-converter generate-targets \
  --output /path/to/calibration-targets \
  --width 3840 --height 2160 --cube-levels 9

uv run lumix-lut-converter analyze-captures \
  --captures /path/to/jpeg-captures \
  --manifest /path/to/calibration-targets/manifest.json \
  --output /path/to/capture-analysis
```

目标图包含机器可读的四角定位标记和色块坐标；分析结果包括定位叠加图与采样 CSV。

### 拟合相机实测适配器

```bash
uv run lumix-lut-converter fit-captures \
  --samples /path/to/capture-analysis/samples.csv \
  --manifest /path/to/calibration-targets/manifest.json \
  --output /path/to/capture-analysis

uv run lumix-lut-converter convert-empirical \
  --source /path/to/source-luts \
  --adapter /path/to/capture-analysis/STD_to_VLog_camera_fit_33.cube \
  --output /path/to/output-standard-empirical
```

若使用这条实验路线，应固定机位、手动白平衡、手动对焦，并让同一组内的光圈、快门和
照明完全不变。需要覆盖其他曝光时，应补拍完整的一组，而不是逐张改变曝光。

## 开发与验证

```bash
uv sync
uv run pytest
```

核心实现使用 `numpy.float64`。Like709 数值反演使用 KD-tree 初值、有界阻尼
Gauss–Newton、四面体插值，并用 Rec.709、CIELAB 与 CIEDE2000 检查误差。

## 范围与版权

- 本仓库不分发第三方创意 LUT；请转换你有权使用的 LUT 文件。
- Panasonic 官方 LUT 和文档的权利归原权利人所有。
- 不同机型、固件、Photo Style 参数、白平衡、i.Dynamic、Highlight/Shadow 等设置都可能
  改变最终结果。
- Standard 管线已经裁掉的高光或色域信息无法通过后续 LUT 恢复。

## 致谢

特别感谢 [@Jackchou00](https://github.com/Jackchou00) 的精准指导。他指出应优先从
Panasonic 已公开的 V-Log encoding function 与 V-Gamut 定义建立 colour space
transform，并提醒我们验证 V-Log ISO 640 与普通 Photo Style 低 ISO 之间的实际增益关系。
这让研究从复杂的经验反推及时转向了更简洁、可解释、可复现的官方模型路线。
