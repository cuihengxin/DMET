# DMET impurity projector：Co(SH)4^2- 测试

## 结论

`imp_orb` 是直接定义 DMET impurity 子空间的 AO 系数矩阵，而不是只供 AVAS 选择活性轨道使用的辅助基组。分子积分和平均场计算仍采用原来的大基组；自定义收缩基仅定义投影目标，例如 Co 3d 或 Ce 4f/5d 的径向形状。

在 Co(SH)4^2- 上，5 个 AVAS active MOs 产生 0 个标准 Schmidt bath orbital；将 5 个 Co 3d 原子参考函数直接投影到分子 AO 空间后，两套参考基都产生 8 个标准 bath orbital。测试没有调用 ROMP2/BNO bath，因此这里的 8 个轨道来自平均场密度的 impurity--environment Schmidt 分解本身。

## 三种不同对象

需要区分以下三者：

1. **Parent AO basis**：用于分子积分、SCF 和嵌入哈密顿量。本测试为 Co/def2-TZVP、S,H/6-31G*，没有被替换。
2. **Projection/reference basis**：只规定希望 impurity 具有怎样的原子轨道形状。它可以是标准基组，也可以是自行收缩的 pGTO。
3. **AVAS active MOs**：分别在占据和虚轨道空间内旋转所得的分子轨道。它们适合选择 CAS，但作为 DMET impurity 时，与 HF 一粒子密度近似对易，因而标准 Schmidt bath 可以退化为零。

因此，自定义收缩基更适合在这里充当 **DMET impurity projector**，不是必须先经过 AVAS 再取 active MOs。

## 实现

新模块 `embed_sim.impurity_projector` 完成两步：

1. 把选中的参考 AO 投影到 parent AO 空间：

   ```text
   S_parent C_raw = <parent AO | reference AO>
   ```

2. 在 parent AO 度量中正交归一化 impurity，并构造其严格 S-正交补空间。完成后的局域基按 `[impurity | environment]` 排列，随后沿用原有 DMET bath 构造。

投影完整度由广义本征值

```text
eig(C_raw^H S_parent C_raw, S_reference)
```

衡量。它接近 1 表示 parent AO basis 能很好地表示给定参考函数；如果明显偏离 1，不能仅凭正交归一化后的轨道看起来正常就接受该基组。

### API 示例

使用标准 Co 参考基：

```python
from embed_sim import ssdmet
from embed_sim.impurity_projector import project_reference_orbitals

imp_orb, projector_info = project_reference_orbitals(
    mol,
    'Co 3d',
    {'default': 'minao', 'Co': 'ano-r0'},
)

mydmet = ssdmet.SSDMET(
    mf,
    title='CoSH4_projected_3d',
    imp_orb=imp_orb,
    threshold=1e-12,
    es_natorb=False,
).density_fit()
mydmet.build(save_chk=False)
```

使用自行收缩的 Ce 4f/5d pGTO：

```python
custom_ce = [
    # PySCF basis 数据，例如 gto.basis.parse(...) 的返回值
]

imp_orb, projector_info = project_reference_orbitals(
    mol,
    ['Ce 4f', 'Ce 5d'],
    {'default': 'minao', 'Ce': custom_ce},
)

mydmet = ssdmet.SSDMET(
    mf, imp_orb=imp_orb, es_natorb=False,
).density_fit()
mydmet.build(save_chk=False)
```

这里 `custom_ce` 的收缩系数决定 projector 的径向形状，但 Hamiltonian 仍使用 `mol` 中的 parent basis。建议优化目标同时约束：原子/离子不同价态的 4f、5d 投影完整度，分子中的子空间稳定性，以及最终 DMET 能量或密度等可观测量；不要只拟合孤立 Ce 原子的轨道能量。

## Co(SH)4^2- 测试设置

- 几何、电荷和自旋：沿用 `examples/iao_test` 中的 Co(SH)4^2-，charge = -2，spin = 3。
- Parent basis：Co/def2-TZVP，S,H/6-31G*；125 AO，97 电子。
- 平均场：ROHF + X2C + density fitting。
- impurity：5 个 Co 3d 轨道。
- bath：只使用原有标准密度矩阵 Schmidt bath，阈值 `1e-12`；未使用 ROMP2/BNO 扩展。
- PySCF：2.11.0。

运行命令：

```bash
cd DMET
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u examples/test_example/impurity_projector_cosh4.py \
  --init-chk examples/iao_test/CoSH4_rohf.chk \
  --output /tmp/iao_dmet_cosh4_projector_test \
  --verbose 4
```

checkpoint 只提供初始密度；脚本会重新收敛 SCF。本次 ROHF 能量为 `-2986.296641091377 Eh`。

## 测试结果

| impurity 定义 | nimp | 标准 bath | 嵌入轨道数 | frozen occ/vir | `||D_IE||` | HF-in-HF 误差 / Eh |
|---|---:|---:|---:|---:|---:|---:|
| AVAS active MOs（对照） | 5 | 0 | 5 | 45 / 75 | 6.56680e-15 | -3.63798e-12 |
| 投影 def2-TZVP Co 3d | 5 | 8 | 13 | 42 / 70 | 1.50510 | -5.45697e-12 |
| 投影 ANO-R0 Co 3d（以解析后的 basis 数据传入） | 5 | 8 | 13 | 42 / 70 | 0.466461 | -1.22782e-11 |

标准 bath 的环境密度本征值为：

```text
projected_def2tzvp:
0.2584466185  0.2584466196  0.2619814364  0.5583510322
0.5856054656  1.9843495935  1.9855567726  1.9855567893

projected_ano_r0:
0.0013887529  0.0015963023  0.0015963024  0.0018752301
0.0030037272  1.9268124183  1.9268778888  1.9268779254
```

参考 AO 在 parent AO 空间中的投影完整度：

```text
def2-TZVP:  1.0000000000  1.0000000000  1.0000000000
             1.0000000000  1.0000000000

ANO-R0:     0.9999177948  0.9999178512  0.9999279677
             0.9999279677  0.9999280084
```

所有三组计算中，输入 impurity 与最终嵌入空间前 5 列之间的子空间重叠奇异值均在数值精度内为 1；完整 `[frozen occupied | embedded | frozen virtual]` 分区的 S-正交误差小于 `1.4e-13`；HF-in-HF 误差小于 `1.3e-11 Eh`。

这说明 projector 被严格保留，轨道分区完整，并且嵌入的一电子构造通过了 HF 极限检查。HF-in-HF 一致性不是相关方法精度的证明。

两套直接 projector 都得到 8 个 bath，但其占据数分布和 `||D_IE||` 明显不同，说明收缩方式确实改变了 impurity--environment 耦合的表示。**bath 数量相同不等于基组质量相同**；后续优化应比较目标化学过程中的能量差、密度、轨道占据和对几何/氧化态的可迁移性。

ANO-R0 一例不是拟合所得的“最优 Co 基”，而是用解析后的 PySCF basis 数据验证：接口确实接受与用户自定义 pGTO 收缩完全相同的数据结构。

## 自动检查与预期行为

脚本会直接失败并给出异常，如果出现以下任一情况：

- 输入 impurity 子空间未被嵌入基严格保留；
- `[frozen occupied | embedded | frozen virtual]` 不是完整 S-正交分区；
- HF-in-HF 误差大于 `1e-7 Eh`；
- 统计的 bath 数与环境密度本征值不一致；
- AVAS-MO 对照不为 0 bath；
- 直接 projector 没有产生 bath；
- reference-to-parent 投影完整度小于 0.99。

## 当前限制

- `imp_orb` 当前接入 `SSDMET` 和 `DFSSDMET`，尚未接入 `AODMET`/`DFAODMET`。
- `imp_orb` 不能与 `imp_idx`、`basis_rot`、`iaopao` 或 `ip_iao` 同时使用。
- `bath_norb='per_bond'` 依赖 AO 的原子归属，对一般 projector 没有唯一意义；请使用阈值选择或显式整数。
- 输入 projector 若在 AO 度量下线性相关，会报错而不是静默删除轨道，因为删除列会改变 impurity 定义。
- 自定义参考函数必须被 parent AO basis 充分张成；应检查返回的 `projector_info['projection_eigenvalues']`，而不是只检查正交性。

原有 `05avas_dmet` 目录仍可作为 AVAS-MO/ROMP2 bath 的独立对照。本测试专门隔离“直接原子 projector 是否能恢复标准 DMET bath”这一问题。

激发能并不会因为标准 bath 非零而自动改善。完整的 SA-CASSCF/SC-NEVPT2/SISO 对比和限制见 `README_impurity_projector_excitation.md`。
