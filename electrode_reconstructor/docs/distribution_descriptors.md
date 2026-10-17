# 弦长分布、球接触分布与两点相关函数

## 依据与接入范围

依据 Prifling et al. (2021), *Large-Scale Statistical Learning for Mass Transport
Prediction in Porous Materials Using 90,000 Artificially Generated Microstructures*，
§2.5–2.7（PDF 第 4–5 页），[论文全文](https://doi.org/10.3389/fmats.2021.786502)。
实现位于外层 `electrode_structure_factors.py`，和瓶颈度共用结构读取及周期距离函数。
这是独立的数字图像实现，未与作者软件逐位对标。

- 三维等边体素，数组轴顺序 x/y/z，长度输出单位 μm。
- 孔隙为标签 0；活性颗粒 1 和 CBD 2 都属于孔隙的补集。独立 API 可通过
  `pore_label` 指定目标相。原有比表面积的 `--liquid-values` 不改变这些描述符的孔隙定义。
- 三个描述符均采用 xyz 周期几何。单文件、批处理和 `electrode-kit.py` 总流程自动计算。
- JSON 和 Excel 同时记录数值、完整曲线、单位、状态和方法版本。
- 与现有瓶颈度相同，新增描述符不自动加入六项 `MULTITASK_FACTOR_NAMES`；模型输出维数、
  旧检查点及训练标准化逻辑保持原样。保存描述符不等于已经用它训练或提高预测精度。
- 只修改外层主源码，嵌套 `electrode_reconstructor/electrode_reconstructor/` 副本未同步。

## 弦长分布 d(L)

沿指定轴扫描孔隙连续段，首尾都为孔隙时合并两段；每条弦等权，而非按弦长加权。
默认 x 轴，可用 `--chord-axis x/y/z` 选择。长度为连续体素数乘体素边长。
输出均值、总体标准差（ddof=0）、5%、10%、…、95% 的 19 个分位数，以及完整
离散支持集、频数、概率质量和经验 CDF。

分位数采用经验分布的广义逆：`Q(p)=inf{l:F(l)>=p}`，不做线性插值。
这是本实现明确选定的分位数约定；论文未给出软件中的具体插值选项。

若一整条扫描线都是孔隙，周期延拓后没有有限的完整弦。实现只计一次观察窗口长度，
设置 `status=window_censored` 并记录 `window_spanning_line_count` 和
`full_pore_line_policy=one_period_censored`。因此含这类扫描线的统计量是窗口截断值，
不能当作无偏的无限周期弦长分布。无孔隙时均值、标准差和分位数为 null，状态 `no_pore`。

## 球接触分布 F_H(r)

严格按 §2.6：Ξ 为孔隙，从 Ξ 的补集（固相）均匀取点，计算到最近孔隙点的距离 H。
因此计算的是 **固相 → 孔隙**，不是孔隙 → 固相的孔径分布。

调用周期 EDT，以孔隙体素为目标，仅采样固相体素；F_H(r) 是距离不大于 r 的固相
体素数除以全部固相体素数。它与瓶颈度 CPSD/MIP 的总体积分数归一化不同。
距离采用体素中心到体素中心约定，不减半个体素；低分辨率下存在数字化误差。
均值、标准差、19 分位数和经验 CDF 使用上述相同约定。

无孔隙或无固相时，距离统计不适用，输出 null、空曲线，状态分别为 `no_pore`、
`no_solid`，不输出 NaN 或 Infinity。有效分布曲线在首个支持点之前取 0，最后一点取 1。

## 两点相关函数 C(r)

以孔隙指示函数 I 计算周期自相关：

`C(d) = IFFT(FFT(I) * conj(FFT(I)))[d] / N`

它是两个点同时处于孔隙的联合概率，满足 `C(0)=孔隙率`。
不减孔隙率平方，也不除以孔隙率，因此不把零点归一化成 1。
全固相和全孔隙的结果分别为恒 0 和恒 1，并保留相应状态。

径向统计对所有周期位移等权，用最小镜像距离。为了覆盖周期对角线并保留零点，
采用单位体素宽度的壳层：第 0 桶只有原点，第 k 桶为 `(k-1)h < |d| <= kh`。
导出半径是桶的上边界，同时保存每桶平均实际距离和位移数。
**这是本项目的离散壳层平均约定，不宣称等于连续空间恰好距离 kh 的点值，
也不宣称与作者未披露的径向分箱方式完全相同。**

曲线长度随网格变化，最大桶为 `ceil(sqrt(sum(floor(n_i/2)^2)))`。
例如 64³ 是 57 点，论文的 192³ 是 168 点；不把 64³ 的数据强行外推成 168 点。
还保存 x/y/z 三条轴向曲线，位移范围分别为 0 至该轴半周期。
径向平均会隐藏各向异性，电极比较应结合方向曲线。有限周期结构的远端值也不必等于孔隙率平方。

## JSON 字段与 API

| 字段 | 内容 |
|---|---|
| `chord_length_mean_um` / `chord_length_std_um` | 弦长均值/标准差 |
| `chord_length_distribution` | 方向、状态、分位数、完整分布和周期约定 |
| `spherical_contact_mean_um` / `spherical_contact_std_um` | 球接触距离均值/标准差 |
| `spherical_contact_distribution` | 距离方向、状态、分位数、完整分布和距离约定 |
| `two_point_correlation` | 径向曲线、分箱信息、三方向曲线及归一化定义 |

`compute_chord_length_distribution(labels, h, axis=0)`、
`compute_spherical_contact_distribution(labels, h)`、
`compute_two_point_correlation(labels, h)` 均可独立调用。

`compute_all_structure_factors()` 返回元组在原来的 8 个结果后追加弦长、球接触、两点相关
三个结果；`process_single_structure_file()` 的返回元组也相应追加三项。项目内调用已更新，
外部使用固定数量解包的脚本需要同步调整。JSON/Excel 写入函数的新增参数均为可选。

## 使用

在 E:/chem 下执行，替换输入路径：

```powershell
# 单文件：Excel 与同目录 JSON；三个描述符默认都会计算
.\Enviroment\Scripts\python.exe .\electrode_reconstructor\electrode_structure_factors.py PATH_TO_PHASE_SLICES.txt --voxel-size-um 0.5 --chord-axis x

# 批处理：更新目标目录中每个样本的 JSON 和 Excel
.\Enviroment\Scripts\python.exe .\electrode_reconstructor\electrode_structure_factors.py --batch-root PATH_TO_DATA --batch-output-format both --voxel-size-um 0.5 --chord-axis z

# 回归验证
.\Enviroment\Scripts\python.exe -B -m unittest discover -s .\electrode_reconstructor\tests -v

# 绘制合成结构示例
.\Enviroment\Scripts\python.exe -B .\electrode_reconstructor\docs\plot_distribution_descriptors_demo.py
```

总流程同样支持 `--chord-axis`。已经存在的文件仍遵循跳过规则；要给历史数据补上新字段，
运行总流程时添加 `--recompute-existing-factors`，并选用原始结构对应的体素尺寸。
本次源码更新不会自动覆盖 `Structure_Data` 的历史结果。

## 验证内容

除原有 15 项瓶颈度验证外，新增 11 项测试，覆盖：已知长度的周期弦、等权统计和
分位数边界；全孔隙线截断标记；轴置换、平移和长度缩放；固相/CBD 到孔隙的方向；
周期 EDT 对独立全配对距离；FFT 对直接孔隙对计数；C(0) 和全位移平均等于孔隙率平方；
退化相、非法输入；单文件/批处理 CLI；总流程写出、跳过和重新计算策略。

`distribution_descriptors_demo.png` 是合成结构曲线，用于解释输出，不是论文实验数据。

## 已有电极样本验证

对 `Structure_Data/test/20260703_02/electrode_test_0001_phase_slices.txt` 执行完整计算，
网格 64³、体素边长 0.5 μm、弦长方向 x、瓶颈度入口 x-low、26 邻域：

| 检查项 | 结果 |
|---|---:|
| 弦长均值 / 标准差 | 2.911738 / 3.139724 μm |
| 球接触距离均值 / 标准差 | 0.648492 / 0.246163 μm |
| 两点相关函数 C(0) | 0.5686988831，与孔隙率相同 |
| 径向曲线点数 | 57 |
| 原有瓶颈度 β | 0.8333333333 |

弦长状态为 `window_censored`，上述弦长统计包含窗口截断线；球接触状态为 `ok`。
完整计算及临时导出本机用时约 4.27 秒，仅作为该样本的运行记录，不代表普遍性能。
数值摘要存于 `distribution_descriptors_validation.json`；JSON 与 Excel 的分位数、
完整分布和径向/轴向相关函数已逐项核对。验证输出位于 `tmp/descriptor_validation/`，
没有覆盖原样本的历史结果。全部 26 项单元/集成测试通过。
