# LUMIX LUT 色彩探索：研究发现与方法说明

本文记录项目从“反推 Panasonic Standard”到“使用官方 V-Log / V-Gamut 模型做解析
CST”的研究过程。重点不是宣称破解了相机未公开的内部处理，而是把已知条件、实测证据、
误差和边界公开，便于复现与继续讨论。

## 1. 问题是什么

LUMIX 可以让创意 LUT 依附于不同 Base Photo Style。很多现成 LUT 以 V-Log / V-Gamut
为输入，因此相机需要工作在 V-Log 管线；在照片模式中，这通常意味着较高的最低 ISO。

我们的目标是构造一个新的 LUT，使它接收 Standard/sRGB 的 RGB，却尽量产生与“原生
V-Log + 原创意 LUT”相同的视觉结果：

```text
Standard/sRGB 输入 → 技术适配器 F → V-Log/V-Gamut 编码 → 原创意 LUT L
```

即：

```text
L_STD(x) = L(F(x))
```

这不是从最终 JPEG 恢复场景线性 RAW，也不是恢复被原 LUT 或相机裁掉的信息。它是把
已知的输入基底转换与创意映射预先合成到一张新 3D LUT 中。

## 2. 为什么不把“主观拟合 Standard”作为首选

早期思路是拍摄屏幕色卡，用 V-Log 与 Standard JPEG 配对，直接拟合
`Standard → V-Log`。这条路线能工作，但它同时吸收很多不属于色彩空间本身的变量：

- ISO 标称值与实际模拟/数字增益；
- 快门、屏幕亮度和相机曝光误差；
- 自动白平衡或白平衡微调；
- Standard 的 tone curve、饱和度、对比度和其他 Photo Style 参数；
- JPEG 编码、色彩管理和显示器光谱特性。

因此一次实拍拟合很容易只对“这台相机 + 这组设置 + 这块屏幕”成立。Panasonic 已经公开
V-Log 和 V-Gamut 的技术定义后，优先使用解析模型更简洁，也更容易审计。实拍仍然重要，
但角色应该是验证模型和测量相机未公开部分，而不是替代公开公式。

## 3. 官方 V-Log 编码函数

Panasonic 的参考文档给出 V-Log OETF。输入为归一化线性光 `L`，输出为 V-Log code
value `V`：

```text
当 L < 0.01：
V = 5.6 × L + 0.125

否则：
V = 0.241514 × log10(L + 0.00873) + 0.598206
```

对应逆函数的 encoded 分界点为 `V = 0.181`。文档也给出典型的 10-bit code value：

| 反射率 | 10-bit code value |
| --- | ---: |
| 0% | 128 |
| 18% | 433 |
| 90% | 602 |

来源：[Panasonic V-Log / V-Gamut Reference Manual](https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf)

## 4. V-Gamut 与 sRGB/BT.709

Panasonic 公布的 V-Gamut 原色与白点为：

| 原色 | x | y |
| --- | ---: | ---: |
| R | 0.730 | 0.280 |
| G | 0.165 | 0.840 |
| B | 0.100 | -0.030 |
| White | D65 | D65 |

官方 V-Gamut 到线性 BT.709 矩阵是：

```text
[ 1.806576  -0.695697  -0.110879 ]
[-0.170090   1.305955  -0.135865 ]
[-0.025206  -0.154468   1.179674 ]
```

项目计算其逆矩阵，将线性 BT.709/sRGB 转换到线性 V-Gamut：

```text
[0.585196050438  0.322641503991  0.092162445571]
[0.078588423325  0.819627144853  0.101784431822]
[0.022794304377  0.114216866321  0.862988829302]
```

sRGB 和 BT.709 的 RGB 原色及 D65 白点相同，但传递函数不同。本项目的解析路径会先用
sRGB EOTF 把 code values 解码为线性光，不能把 sRGB 数字直接当作线性 BT.709。

完整适配器是：

```text
x_sRGB
→ sRGB EOTF
→ M_BT709_to_VGamut
→ V-Log OETF
→ x_VLog
```

然后在 `float64` 中对原创意 LUT 做四面体插值，合成为 33 点机内 LUT。

## 5. 为什么最终选择 sRGB，而不是 Adobe RGB

Adobe RGB 能表示更宽的绿色区域，但“色域更宽”不等于“V-Log LUT 还原更准”。对当前
用途，sRGB 有几个决定性优势：

1. sRGB 与 BT.709 原色一致，Panasonic 官方矩阵可以直接进入模型。
2. LUMIX Lab、相机 JPEG、操作系统预览和网络输出对 sRGB 的解释最稳定。
3. Adobe RGB 需要相机输出、ICC 标记、查看器、编辑软件和最终导出全链路正确管理；任何
   环节忽略 profile 都会产生明显偏色。
4. 原创意 LUT 最终往往也是面向 SDR/Rec.709 观看；增加 Adobe RGB 中间环节不会恢复
   原 LUT 已经不存在的数据。

这不表示 Adobe RGB 永远无用。如果目标是建立一条严格色彩管理的宽色域静态摄影链路，
可以另行实现 Adobe RGB EOTF 与 Adobe RGB→XYZ→V-Gamut 的适配器，并重新做实机验证。
但它不是目前最简单、最稳健的机内 LUT 方案。

## 6. 解析适配器的数值精度

我们用 33 点技术适配器与未采样的解析公式比较，在 200,000 个随机 RGB 样本上测得：

| 指标 | 12-bit 最大通道 code value 误差 |
| --- | ---: |
| 平均 | 0.228 |
| P95 | 0.586 |
| P99 | 1.264 |
| 最大 | 3.636 |

中性轴三个通道的最大散布约为 `2.22e-16`，即计算精度范围内保持中性。这里衡量的是
“33 点 LUT 对解析适配器的采样误差”，不是最终相机 JPEG 与目标照片的综合色差。

## 7. RAW 实测：ISO 数字不能跨 Photo Style 直接比较

### 7.1 同光圈、同快门、两边都 ISO 640

一组 V-Log 与 Standard RAW 使用相同光圈、快门和 ISO 640。扣除 RAW black level
128、以 white level 4079 归一化后，对 Bayer 四通道做稳健线性比较：

| 通道 | Standard 640 / V-Log 640 斜率 |
| --- | ---: |
| R | 6.0617 |
| G1 | 6.0939 |
| G2 | 6.0859 |
| B | 6.0654 |

合并通道、强制过原点的斜率为 **6.0848**，像素比值中位数为 **6.1304**，约
**2.60–2.62 EV**。这说明 LUMIX 在不同 Photo Style 下对 ISO 标称和增益的处理并不
相同；不能认为“都显示 ISO 640”就拥有相同 RAW 线性曝光。

### 7.2 V-Log ISO 640 是否等于 Standard ISO 100

另一组实际使用创意 LUT 的对比中，V-Log 为 ISO 640、1/25 s，Standard 为 ISO 100、
1/5 s，光圈均为 f/2.8。RAW 的 Standard/V-Log 信号比约为 **6.27**；除以 5 倍快门时间
后约为 **1.254**，即仍有约 **0.33 EV** 差异。

这个结果方向上支持“V-Log 640 与普通 Photo Style 100 大致对应”的判断，却不是严格的
6.4 倍验证，因为这组照片的快门不同，并且使用了 AWB、AF、IBIS 等实际拍摄设置。要测定
更精确的机型常数，应固定照明、白平衡、对焦、机位和光圈快门，只切换 Photo Style 与
ISO，并重复多档灰阶和多次拍摄。

对 LUT 使用者而言，更重要的实践结论是：**进入 Standard Base 后，可以回到 Standard
自己的低 ISO 工作范围，不必为了数字相等而把 Standard 也设为 ISO 640。**

## 8. 屏幕色卡拟合实验

项目生成了带四角定位标记的 4K SDR PNG，包括灰阶、通道阶梯、规则 RGB cube 和真实
HDR/风景图像。分析器可以自动定位透视区域并提取机器可读色块。

第一组样本因自动白平衡变化，不适合建立统一色彩映射。第二组固定在 5500 K、手动对焦，
39/39 张均成功定位，共提取 3,552 个配对样本。

经验适配器在非裁切样本上的误差：

| 数据集 | 平均最大通道误差 | P95 |
| --- | ---: | ---: |
| 训练集 | 0.52/255 | 1.19/255 |
| 验证集 | 2.80/255 | 8.28/255 |

这些数字证明自动定位和拟合流程有效，但两种 Photo Style 当时都设为 ISO 640，所以模型
也学习了前述约 6.08 倍的增益差异。它不能不加区分地作为“Standard ISO 100 通用逆向
LUT”。因此经验后端被保留用于研究、机型特定校正和解析模型残差测量，而不再是首页推荐
路线。

## 9. 真实创意 LUT A/B

我们比较了：

- 原生 V-Log + 原 Fuji Classic Negative V-Log LUT；
- Standard/sRGB + 解析 CST 合成的 Fuji Classic Negative LUT。

这组照片的机位和光圈相同，但快门不同，因此不能把逐像素差异全部归因于 LUT。直接比较
JPEG，最大通道 code value 差的均值为 **11.89/255**、中位数 **12/255**、P95
**17/255**；Standard 版本的中位亮度约暗 **0.229 EV**。

用一条对所有像素共享的单调 tone curve 对齐亮度后：

| 指标 | 最大通道残差 |
| --- | ---: |
| 平均 | 3.526/255 |
| P95 | 7.367/255 |
| P99 | 10.687/255 |

tone 对齐后的 RGB 有符号残差中位数为 `[-1.16, +0.30, -0.69] / 255`。这说明解析 CST
在真实创意 LUT 上已经得到相当接近的色彩关系，主要可见差异更像曝光与 Standard/V-Log
tone rendering，而不是大幅色相错误。

仍建议补做一组严格 A/B：三脚架、固定手动白平衡、手动对焦、关闭自动动态优化，保证
光圈快门和照明不变；V-Log 用 ISO 640，Standard 用 ISO 100，并拍摄灰卡、肤色、高饱和
物体和高光过渡。它用于确认具体机型的残余 tone 偏差，不再需要重新拍完整 3D 色卡才能
使用解析转换。

## 10. Like709 数值反演路线

项目最初使用 Panasonic 官方 `VLog_to_V709_forV35` 33 点 LUT，构造：

```text
T: V-Log/V-Gamut → V709 legal range
```

然后数值求解 `T⁻¹`，把原创意 LUT 合成为：

```text
L_Like709(x) = L(T⁻¹(Legalise(x)))
```

实现使用 10-bit legal range `64/1023…940/1023`、KD-tree 初值、有界阻尼
Gauss–Newton 和四面体插值。这条路线仍有研究价值，但存在两个额外假设：LUMIX 的
Like709 Photo Style 要与 VariCam 官方 V709 技术 LUT 足够一致，而且 legal/full range
解释必须吻合。对于“在照片模式绕开 V-Log 最低 ISO”这一目标，解析 Standard/sRGB
路线更直接。

## 11. LUMIX 格式兼容性实测

基于 LUMIX S9 和 LUMIX Lab 的本地导入测试：

| 格式 | 网格 | S9 SD 导入 | LUMIX Lab |
| --- | ---: | --- | --- |
| `.cube` | 17 | 可用 | 非首选 |
| `.cube` | 32 | 可用 | 非首选 |
| `.cube` | 33 | 可用 | 可用 |
| `.vlt` | ≤33 | 可用 | 不作为兼容目标 |
| `.cube` | 65 | 拒绝 | 拒绝/不适合机内 |

因此默认输出 33 点 `.cube`。65 点输出可用于桌面调色软件、误差研究或后续降采样，不应
作为 LUMIX Lab/相机导入文件。

## 12. 当前最佳实践与边界

当前建议：

1. 原 LUT 确认以 V-Log/V-Gamut 为输入。
2. 使用 `convert-cst` 合成 Standard/sRGB 33 点 LUT。
3. 相机选 Standard Base、sRGB，并使用 Standard 的正常 ISO。
4. 用项目生成的技术适配器或一张代表性创意 LUT 做固定机位 A/B。
5. 若某台机型表现出稳定的残余 tone 差异，再用少量严格配对数据建立二级校正；不要先
   把曝光和白平衡漂移一起拟合进 3D LUT。

已知边界：

- Panasonic 没有完整公开 Standard Photo Style 的 tone curve、gamut mapping 和 LUT
  插入位置；
- LUT 只能处理三通道 code values，不能恢复 RAW 动态范围或区分同色异谱；
- 任何输入裁切都会让映射不可逆；
- 17/33 点 LUT 的空间采样会带来有限插值误差；
- 相机的白平衡、曝光、i.Dynamic、Highlight/Shadow、饱和度和固件行为都可能改变结果；
- 不同创意 LUT 对高光与高饱和边界的敏感程度不同，应抽样检查。

所以这里的“最佳实践”是：用公开的官方模型解决可以解析解决的部分，再用受控实拍确认
相机未公开的残差；不是宣称 Standard 与 V-Log 可以在所有像素上严格无损互换。

## 13. 致谢

特别感谢 [@Jackchou00](https://github.com/Jackchou00) 的精准指导。他在研究中指出，
Panasonic 已经公开了 V-Log encoding function 和 V-Gamut，应该优先建立 colour space
transform，而不是把所有变量都交给经验拟合；同时建议从 RAW 验证 V-Log ISO 640 与
普通 Photo Style ISO 100 的增益关系。这两个判断直接促成了当前解析 CST 路线。

所有公式实现、样本分析、误差统计和文档中的判断由本项目基于公开资料与实拍继续完成；
我们也欢迎针对不同 LUMIX 机型、固件和 Photo Style 的可复现实验数据。
