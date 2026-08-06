# One bath orbital per bond(每键一个 bath 轨道)

把 QC-DMET `test/sn2.py` 中使用的 **one bath orbital per bond** 方案(Sun &
Chan, JCTC 10, 3784 (2014))移植到 `embed_sim`,并在小分子上测试其选轨道效果。

## 1. 方案原理

默认的 bath 选择(阈值方案)把环境块 1-RDM 所有占据数落在
`[threshold, 2-threshold]` 的自然轨道全部选为 bath;在紧阈值(默认
`threshold=1e-12`)下,对小分子这几乎等于"环境全部进嵌入空间",簇大小等于整个分子。

"每键一个 bath"则**固定 bath 轨道个数** `nbath`,从环境自然轨道里挑占据数
**最接近 1(纠缠最强)** 的前 `nbath` 个作为 bath,其余按占据数冻结
(> 1.5 为 frozen occupied,< 0.5 为 frozen virtual;与 QC-DMET 的
`core_cutoff=0.5` 一致)。`nbath` 的自然取值是杂质–环境之间的键数。

## 2. 使用方法

核心参数 `bath_norb`(所有 `SSDMET` / `AODMET` / `DFSSDMET` /
`DFAODMET` 都支持):

```python
from embed_sim import ssdmet, aodmet

# 方式 A:自动按键数(杂质原子-环境原子之间共价半径连接数)
mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx='0 O.*',
                       bath_norb='per_bond')

# 方式 B:手动指定 bath 轨道数
mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx='0 O.*', bath_norb=2)

# 方式 C:AO-DMET 同样支持
mydmet = aodmet.AODMET(mf, title='h2o', imp_idx='0 O.*',
                       bath_norb='per_bond')

mydmet.build()
```

说明:

- `bath_norb=None`(默认)保持原来的阈值选择,行为不变。
- `bath_norb='per_bond'` 时,键数由几何自动判定(共价半径之和 × 1.3 为成键
  判据,见 `embed_sim/bath_selection.py` 中的 `COVALENT_RADII` 表,目前只列了
  测试用到/常用的元素,新元素请补充)。
- 如果 `nbath` 太小,剩余环境轨道中存在占据数落在 `[0.5, 1.5]` 的"既不能
  冻结为占据、也不能冻结为空"的轨道,会抛出带占据数提示的 `ValueError`
  (与 QC-DMET 的断言逻辑一致),提示增大 `nbath`。
- 开壳层(ROHF 参考)同样支持;DF 变体建议 `es_natorb=False`(见下方已知问题)。
- checkpoint 兼容:`bath_norb` 会写入/参与校验 `<title>_dmet_chk.h5`;
  换用不同的 `bath_norb` 会自动重建嵌入空间。

运行测试(在 `DMET_main` 根目录或把其加入 `PYTHONPATH`):

```bash
python examples/test_example/one_bath_per_bond.py
# 或
sbatch examples/test_example/one_bath_per_bond.sh
```

## 3. 小分子测试结果(6-31G,RHF/ROHF 参考 + x2c)

`E_dmet = E(嵌入空间低阶解) + E(冻结占据轨道)`,误差
`err = E_dmet - E(full RHF)`,`nclust = nimp + nbath`。

| 分子 | 杂质 | nimp | nbath(默认) | nbath(每键) | E_dmet(默认) | E_dmet(每键) | err(每键)/mHa |
|------|------|-----:|-----:|-----:|--------------|--------------|--------------:|
| H2   | H0   | 2    | 1     | 1     | -1.1267634496 | -1.1267634496 | ~0.000 |
| LiH  | Li0  | 9    | 2     | 1     | -7.9799907474 | -7.9689955482 | 10.995 |
| H2O  | O0   | 9    | 3     | 2     | -76.0327461130 | -76.0023749908 | 30.371 |
| CH4  | C0   | 9    | 5     | 4     | -40.1945276509 | -40.1755373475 | 18.990 |
| F2   | F0   | 9    | 9     | 1     | -198.8086101549 | -198.6019534074 | 206.657 |
| N2*  | N0   | 9    | 7     | 3     | -108.9228702376 | -108.4217377610 | 501.132 |
| OH   | O0   | 9    | 2     | 1     | -75.4107782045 | -75.3784574468 | 32.321 |

`*` N2 是三键:`bath_norb='per_bond'` 按原子对只数出 1 个键,会因剩余轨道
占据数仍为 1 而报错;需手动 `bath_norb=3`(脚本里已演示报错信息)。

AODMET(环境-Löwdin)每键方案对比:

| 分子 | E_dmet(每键) | err/mHa |
|------|--------------|--------:|
| H2   | -1.1267634496 | ~0.000 |
| LiH  | -7.9799900878 | 0.001 |
| H2O  | -76.0263974491 | 6.349 |
| CH4  | -40.1944185607 | 0.109 |
| F2   | -199.0758951767 | -267.285 |
| N2*  | -109.8340051691 | -911.135 |

### 结果解读

1. 默认紧阈值方案在小分子上会把整个分子选进嵌入空间(`nbath(默认)` ≈ 全部
   环境轨道),DMET 误差 ~1e-12,但簇大小最大。
2. 每键方案显著缩小嵌入簇(F2 18→10、H2O 12→11、CH4 14→13、LiH 11→10),
   单键、弱纠缠体系(LiH、CH4、H2O 的 AODMET)误差在 ~0–30 mHa,可接受。
3. 对强键/多键体系(F2 的 σ 键、N2 的三键),每键 1 个 bath 明显不足,误差
   达 0.2–0.9 Ha。这与 QC-DMET SN2 场景(饱和 C–C 单键链)不同——单键链上
   每键一个 bath 恰好覆盖纠缠,而多键体系需要更多 bath(可手动增大
   `bath_norb`,并依靠 `ValueError` 提示判断是否足够)。
4. `bath_norb=1` 触发 N2 的报错是特性而非 bug:它把"冻结轨道仍高度纠缠"
   的情况显式暴露出来,避免静默地给出错误能量。

## 4. 实现与进度记录

### 已实现(2026-08-06)

- 新增 `embed_sim/bath_selection.py`:
  - `partition_env_by_bath_count`:固定 bath 数的环境自然轨道划分(最接近 1
    优先),含 `core_cutoff=0.5` 校验与错误提示;
  - `count_imp_env_bonds` / `imp_atom_indices`:几何成键计数(共价半径,
    Bohr→Å 转换,AO→原子映射)。
- `embed_sim/ssdmet.py`:`build_embeded_subspace` 与 `SSDMET` 增加
  `bath_norb` / `bath_core_cutoff`;checkpoint 写入与校验;`density_fit()`
  转发参数。
- `embed_sim/aodmet.py`:同样支持;顺带修复既有 bug——`build()` 中
  `self.fo_ene()` 改为 `self.calc_fo_ene()`(原代码必然 AttributeError)。
- `embed_sim/df.py`:`DFSSDMET` / `DFAODMET` 同样支持(`__init__`、
  `build`、chk)。
- 测试脚本 `one_bath_per_bond.py` + SLURM 包装 + 参考输出
  `one_bath_per_bond.out` / 结果表 `one_bath_per_bond_results.txt`。

### 已知限制

- `uhf_dmet/`、`uhf_dmet_ic/`(UHF 参考,SVD 按自旋构造 bath)尚未接入
  `bath_norb`,如需可在 `uhf_tool.py` 的 `svd_build_embedded_space` 中按奇异值
  截断扩展。
- `DFSSDMET`/`DFAODMET` 在 `es_natorb=True` 时嵌入 RHF 收敛会因
  `es_dm` 形状报错——这是仓库既有问题(默认阈值路径同样触发),与本功能无关;
  使用 DF 变体时请设 `es_natorb=False`。
- 多键体系(如 N2)需手动指定 `bath_norb`;未来可考虑按键级(键序)扩展计数。
