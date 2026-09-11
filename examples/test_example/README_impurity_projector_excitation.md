# Co(SH)4^2- impurity projector 激发能比较

## 结论

不能笼统地说“直接 projector 方法比以前的方法更好”。本次同一 Hamiltonian、同一 CAS(7e,5o) 的测试得到：

- 只保留直接 projector 产生的 8 个标准 Schmidt bath 时，13 轨道嵌入的激发能明显变差。
- 在与旧方法相同的 52 个嵌入轨道预算下，**ANO-R0 Co 3d projector + 39 个 ROMP2 BNO** 的低能 SISO 激发能优于旧 AVAS-projector + ROMP2 方法。
- def2-TZVP projector 在相同 52 轨道预算下仍比旧方法差，说明改善来自“合适的 projector 收缩 + 合适的相关 bath”的组合，不是 `imp_orb` 接口本身。
- ANO-R0 等规模方案只在所考察的低能窗口更好；对全部 60 个 Kramers 对或全部 50 个 spin-free 态，它没有一致优于旧方法。

因此，目前最准确的表述是：**ANO-R0 projector 在 Co(SH)4^2- 的低能激发区显示出约两倍的误差改善，但尚不能据此宣布它是普适更好的基组。**

## 比较定义

所有结果使用同一个 Co(SH)4^2- 几何、Co/def2-TZVP + S,H/6-31G* parent basis、ROHF/X2C/DF 参考和 CAS(7e,5o)。电子态设置固定为 40 个 doublet 和 10 个 quartet：

```text
SA-CASSCF(7e,5o) -> state-specific SC-NEVPT2 -> SISO
statelis = [0, 40, 0, 10]
```

这里的“基准”是仓库已有的全体系 125-AO SA-CASSCF/SC-NEVPT2/SISO 结果，不是实验值。因此本测试衡量的是嵌入对同级别全体系计算的复现程度。

比较方法如下：

| 名称 | impurity | 标准 bath | ROMP2 BNO | 总 bath | 嵌入轨道 |
|---|---|---:|---:|---:|---:|
| 全体系参考 | 无嵌入 | - | - | - | 125 |
| 旧 AVAS+ROMP2 | AVAS Co 3d active MOs | 0 | 47，`eta=1e-6` | 47 | 52 |
| def2 standard | 投影 def2-TZVP Co 3d | 8 | 0 | 8 | 13 |
| ANO standard | 投影 ANO-R0 Co 3d | 8 | 0 | 8 | 13 |
| def2 equal52 | 投影 def2-TZVP Co 3d | 8 | 39，固定数 | 47 | 52 |
| ANO equal52 | 投影 ANO-R0 Co 3d | 8 | 39，固定数 | 47 | 52 |

equal52 使用固定增加 39 个 ROMP2 BNO，是为了控制总嵌入维数与旧方法相同。直接 projector 若原样使用 `eta=1e-6`，def2 和 ANO-R0 分别产生 103 和 118 个嵌入轨道，已经接近全体系，不能作为 52 轨道旧方法的等成本比较。

## 主要误差

低能 SISO 指标取排序后第 1--9 个激发 Kramers 对，即去掉基态 Kramers 对后的九组低能激发；spin-free 指标取最低 quartet 以上的另外 9 个 quartet root。误差单位均为 cm^-1。

| 方法 | 嵌入轨道 | 低能 SISO MAE | 低能 SISO RMSE | 最大低能 SISO 误差 | quartet NEVPT2 MAE | quartet CASSCF MAE |
|---|---:|---:|---:|---:|---:|---:|
| 旧 AVAS+ROMP2 | 52 | 267.949 | 293.960 | 418.008 | 474.558 | **137.800** |
| def2 standard | 13 | 852.329 | 1048.109 | 1881.028 | 1805.998 | 1974.597 |
| ANO standard | 13 | 1308.102 | 2011.210 | 4000.650 | 2389.563 | 5251.837 |
| def2 equal52 | 52 | 910.341 | 1023.312 | 1581.550 | 1202.974 | 721.342 |
| **ANO equal52** | 52 | **134.159** | **157.472** | **259.604** | **343.411** | 272.757 |

相对于旧方法，ANO equal52 的低能 SISO MAE 从 267.949 降到 134.159 cm^-1，下降约 **49.9%**；quartet SC-NEVPT2 MAE 从 474.558 降到 343.411 cm^-1，下降约 **27.6%**。

但在纯 CASSCF 层次，旧方法的 quartet MAE 为 137.800 cm^-1，优于 ANO equal52 的 272.757 cm^-1。低能最终结果的改善是在加入动态相关和 SOC 后出现的，不能归因于 CASSCF 轨道本身全面改善。

## 低能 SISO 能级

表中列出前 10 个 Kramers 对的中心能量；第一列为基态对，其余为激发能，单位 cm^-1。

| pair | 全体系参考 | 旧 AVAS+ROMP2 | ANO equal52 | ANO equal52 误差 |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 48.373 | 44.656 | 42.157 | -6.217 |
| 2 | 2715.996 | 2902.660 | 2975.599 | +259.604 |
| 3 | 2831.831 | 3013.218 | 3085.759 | +253.928 |
| 4 | 7200.673 | 7494.610 | 7276.805 | +76.132 |
| 5 | 7257.230 | 7556.013 | 7324.673 | +67.443 |
| 6 | 7273.235 | 7582.405 | 7373.139 | +99.904 |
| 7 | 7441.494 | 7752.128 | 7530.879 | +89.384 |
| 8 | 9943.441 | 10352.678 | 10121.637 | +178.197 |
| 9 | 10372.148 | 10790.156 | 10548.769 | +176.621 |

ANO equal52 主要改善了约 7200--10500 cm^-1 的能级；第二、第三个激发 Kramers 对仍高估约 254--260 cm^-1。

## 低能 quartet 的 SC-NEVPT2 能级

相对最低 quartet 的能量如下，单位 cm^-1：

```text
full-system reference:
    0.000   2659.302   7228.731   7228.842  10510.525
10510.560  10814.948  21449.653  21449.651  23592.296

previous AVAS+ROMP2:
    0.000   2852.061   7536.065   7536.069  10934.371
10934.372  11261.369  22110.972  22110.977  24439.271

projected ANO-R0 equal52:
    0.000   2929.092   7289.680   7347.984  10698.187
10710.457  10855.998  22120.627  21979.001  24604.183
```

## 全谱限制

若把高能态也纳入，结论不再是 ANO equal52 全面占优：

| 方法 | 全部 60 个 Kramers 对 MAE | 全部 50 个 spin-free NEVPT2 态 MAE |
|---|---:|---:|
| 旧 AVAS+ROMP2 | **887.730** | **988.213** |
| ANO equal52 | 927.172 | 1060.460 |

高能态可能发生顺序变化或态特征交换。当前统计按各自能量/root 顺序比较，没有利用 CI 波函数重叠做逐态追踪；因此全谱 MAE 只应作为总体诊断。若要面向实验光谱优化 projector，应进一步比较主要组态、自然轨道占据、跃迁矩和实验态指认。

## 复现

测试脚本：`examples/test_example/impurity_projector_cosh4_excitation.py`。

在 `DMET` 目录运行全部四个新 case：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u examples/test_example/impurity_projector_cosh4_excitation.py \
  --output /tmp/iao_dmet_cosh4_excitation \
  --verbose 0
```

默认读取：

- 全体系参考日志：`examples/iao_test/CoSH4full.log`
- 全体系 SISO 能级：`examples/iao_test/CoSH4_mag.txt`
- 旧方法结果：`../05avas_dmet/results/co_full/summary.json`
- 旧方法 SISO 能级：`../05avas_dmet/results/co_full/CoSH4_avas_imp_mag.txt`

每个新 case 独立保存 `case_result.json`，总结果保存为 `impurity_projector_cosh4_excitation_results.json`。中断后可在相同输出目录增加 `--resume`，复用已经完成的 case。

本次四组新计算的 SA-CASSCF 均收敛，50 个 SC-NEVPT2 校正均为有限值，SISO 均得到 120 个有限能级；HF-in-HF 误差均小于 `1.3e-11 Eh`。
