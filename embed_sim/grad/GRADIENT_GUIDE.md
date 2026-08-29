# SSDMET 解析核梯度：代码文档与使用指南

> 面向对象：想读懂 `embed_sim/grad/` 代码、学会计算 DMET 解析梯度、
> 并用它做几何优化的研究者。
> 配套理论推导：`8dmet4reac/DMET_ROHF梯度推导.md`（含完整公式与数值锚点）。
> 代码位置：`DMET/embed_sim/grad/`（`ssdmet.py`, `linalg.py`, `numgrad.py`）。

---

## 0. 快速上手

### 0.1 闭壳层（RHF-in-RHF）

```python
from pyscf import gto, scf
from embed_sim import ssdmet

mol = gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96', basis='sto-3g')
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-12          # 关键：梯度精度受密度收敛限制，见 §6.2
mf.kernel()

d = ssdmet.SSDMET(mf, title='h2o', imp_idx=[0], bath_norb=1)
d.build(save_chk=False)

de = d.nuc_grad_method().kernel()     # 返回 (natm, 3) 的梯度，Eh/Bohr
```

### 0.2 开壳层（ROHF-in-ROHF）

```python
mol = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; H 0.02 -0.89 0.07',
            basis='sto-3g', spin=1)              # CH3 自由基
mf = scf.rohf.ROHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-12
mf.kernel()

d = ssdmet.SSDMET(mf, title='ch3', imp_idx=[0], bath_norb=1)
d.build(save_chk=False)
de = d.nuc_grad_method().kernel()
```

### 0.3 几何优化（见 §5 详述）

```python
from pyscf.geomopt.berny_solver import GeometryOptimizer
opt = GeometryOptimizer(dmet_optimizer_wrapper(mol, imp_idx=[0], bath_norb=1))
mol_opt = opt.kernel()              # 返回优化后的 mol
```

---

## 1. 代码结构

```
DMET/embed_sim/grad/
├── __init__.py        # 导出 SSDMETGradients, Gradients
├── ssdmet.py          # 核心：SSDMETGradients 类（八步流水线）
├── linalg.py          # 矩阵/子空间导数原语
└── numgrad.py         # 数值梯度工具（验证用）
```

### 1.1 `grad/ssdmet.py` — `SSDMETGradients`

入口：`mydmet.nuc_grad_method(**kwargs)` 返回该类实例，`.kernel()` 算梯度。

| 方法 | 作用 | 关键产物 |
|---|---|---|
| `__init__` | 初始化，建 `mfgrad`（PySCF 参考梯度）+ 闭壳层 veff 辅助 | — |
| `check_support` | 校验体系（RHF/ROHF、无 x2c/DF/bath_option） | `self.openshell` |
| `decompose` | 重建 Löwdin 分解，缓存所有中间量 | `X,Y,w_s,u_s,w,V,γ,L_raw,L_act,U,...` |
| `make_densities` | 核/活性密度、嵌入 r1/r2、Fock | `dm_core,dm_act,r1,r1a,r1b,r2,veff_core,veff_act` |
| `orb_grad` | 固定嵌入解，算 `dE/dC`、`dE/dCc` | `Ge, Gc, eri_aeee` |
| `lo_grad` | 链式到 Löwdin `X` 和环境本征向量 | `B_X, G_gamma, G_P, B_Y` |
| `solve_z` | 全局 CPHF / Z-vector（RHF 与 ROHF 分支） | `Z, moFbar, zvec_ao, zeta, vhf_s1occ` |
| `make_sbar` | 总 `dE/dS`（Pulay 通道） | `sbar` |
| `contract` | 与 `∂h/∂R, ∂g/∂R, ∂S/∂R` 收缩 | `de`（梯度） |
| `kernel` | 主流程：依次调用上述 | `de` + 核排斥梯度 |

### 1.2 `grad/linalg.py` — 导数原语（两个独立验证过的函数）

| 函数 | 作用 | 数学 |
|---|---|---|
| `matfun_grad(w, u, B, f, fp)` | `d/dS Σ_{uv} B_uv f(S)_uv` | Daleckii–Krein 除差公式，对 `S^{±1/2}` 的简并本征值安全 |
| `eig_subspace_grad(w, v, bmat, group)` | 本征**子空间**导数（bath/frozen 组间耦合，组内相消） | 对 bath 内简并、`es_natorb` 旋转规范免疫 |

### 1.3 `grad/numgrad.py` — 数值验证工具

| 函数 | 作用 |
|---|---|
| `numerical_grad(efunc, mol, step)` | 中心差分梯度 |
| `ssdmet_energy(mol, imp_idx, ...)` | RHF+DMET 总能量（做 FD 时用） |

---

## 2. 数学原理（一句话版，公式见推导文档）

**能量是 CASCI 型泛函**（单片段、变分求解器）：

```
E = E_nuc + Tr[h D_c] + ½Tr[v(D_c)D_c] + Tr[r1 h1] + ½ r2:g_es
```

对变分嵌入解（HF/FCI/CASSCF）**驻定** ⟹ `r1,r2` 固定即可（Hellmann–Feynman），
**不需要**多片段民主分配那套 CP-CI/CP-HF 伴随层。

**响应树**（梯度 = 各通道贡献之和）：

```
R → S, h, g                    （显式积分导数，contract）
  → X = S^{-1/2}               （Löwdin，matfun_grad）
  → γ = Y P Y (Y=S^{1/2})      （环境本征子空间，eig_subspace_grad）
  → P                          （全局 ROHF/RHF 密度响应，CPHF/Z-vector）
```

**Z-vector 折叠**（唯一 ROHF 的难点，全部已数值验证）：

```
sbar_ROHF = sbar_lowdin
          − sym(Σ_σ Zmat_σ F_σ P_σ)                    (zPFP, 系数 −1)
          − sym(Σ_σ ½ P_σ v_σ(W) P_σ)                  (vhf,  系数 −½)
          − ¼ sym(C_occ (moFbar_oo ⊙ (n_i+n_j)) C_occ^T)  (oo,  系数 −¼)
```

关键历史 bug：oo 项的权重必须是 **`(n_i+n_j)`**（来自含 `S^{-1}` 的 Pulay
`−½Tr[dS(P G_P S^{-1}+S^{-1}G_P P)]`）。RHF 下 n=2 自动还原为 `−2 C moFbar C^T`。

---

## 3. 逐方法详解（学习代码的主线）

### 3.1 `kernel()` — 主流程

```python
def kernel(self, atmlst=None):
    self.check_support()       # 0. 校验
    self.decompose()           # 1. 重建 Löwdin + 本征分解（缓存）
    self.make_densities()      # 2. 密度 + Fock
    self.orb_grad()            # 3. dE/dC, dE/dCc
    self.lo_grad()             # 4. 链式到 γ
    self.solve_z()             # 5. 全局 CPHF / Z-vector
    self.make_sbar()           # 6. dE/dS（Pulay）
    de = self.contract(atmlst) # 7. 与积分导数收缩
    de = de + self.mfgrad.grad_nuc(atmlst=atmlst)   # 核排斥
    return de
```

五步对应响应树的五个通道，每一步都是"中间量 → 链式下一层"。

### 3.2 `decompose()` — 为什么重建而不是复用 `build` 的结果

`build()` 只存了**最终**的嵌入轨道（`es_orb`, `fo_orb`），梯度需要**中间量**：
`S` 的本征分解、γ 环境块的本征分解、bath/frozen 的分组、未旋转的原始本征向量。
`decompose()` 用与 `build` 完全相同的路径重建并**全部缓存**，同时做一致性校验
（`nes/nfo/nfv`、`es_orb` 与重算的 `L_act` 匹配、`fo_orb` 投影匹配）。

```
X = u_s·diag(w_s^{-1/2})·u_s^T      # S^{-1/2}
Y = u_s·diag(w_s^{+1/2})·u_s^T      # S^{+1/2}
γ = Y P Y
环境块 γ_EE = V·diag(w)·V^T         # 单次本征分解
L_raw: [imp=I, env×bath=V列],  Lc_raw: env×frozen-occ=V列
```

### 3.3 `make_densities()` — 密度与嵌入 RDM

```python
dm_core = 2 Cc Cc^T                # 冻结核（双占据）
r1s = es_mf.make_rdm1()            # 嵌入解
r1a, r1b = r1s[0], r1s[1]          # ROHF：分自旋
r1 = r1a + r1b
r2 = r1⊗r1 − r1a[ps]r1a[rq] − r1b[ps]r1b[rq]   # 自旋求和单行列式 2-RDM
dm_act = C r1 C^T
veff_core = J(dm_core) − ½K(dm_core)     # 闭壳层有效势
veff_act  = J(dm_act)  − ½K(dm_act)
fock_core = h + veff_core                # = d(½Tr[v(D_c)D_c]+Tr[r1 h1])/dD_c 的 F_c
fock_tot  = fock_core + veff_act         # = dE/dD_c
```

**注意**：核与活性相互作用用**闭壳层** `v=J−½K`（不是分自旋 `v_σ=J−K_σ`）。
理由：冻结核双占据，自旋求和后 `Σ_σ K(D_act,σ)` 合并为 `K(D_act)`（推导 §2）。

### 3.4 `orb_grad()` — 固定嵌入解的轨道导数

```
Ge = 2 F_c C r1 + 2 g2,   g2[μ,u] = Σ_{v,w,x} (μ v|w x) r2[u,v,w,x]     # dE/dC
Gc = 4 F_tot Cc                                                        # dE/dCc
```

- `g2` 用 `mol.intor('int2e')` 显式收缩（不要用 `ao2mo.general` 配 `np.eye`，
  那是踩过的坑，见 §6.3）。
- 驻定性自检：`C^T Ge` 应**对称**。非对称说明嵌入解不是变分的（如 CCSD 作求解器
  时需要弛豫密度，本期不支持）。

### 3.5 `lo_grad()` — 链式法则到 γ

```
B_X = Ge L_act^T + Gc Lc_raw^T          # dE/dX（L 固定）
b_es = X Ge U^T                          # 转回原始本征向量规范
b_fo = X Gc
bmat[env, bath] = b_es[env, nimp:],  bmat[env, fo] = b_fo[env]
G_M = eig_subspace_grad(w, V, bmat)      # 子空间导数（组内相消，规范免疫）
G_P = Y G_γ Y                            # dE/dP（P 通道）
B_Y = G_γ Y P（对称化）                   # dE/dY（S 通道）
```

**为什么 `es_natorb` 旋转规范无关**：能量对嵌入空间内旋转不变（§2 驻定性），
所以 `dE/dL` 只需投影到"环境×bath/frozen"块，U（natorb 旋转）通过 `b_es = X Ge U^T`
被吸收。

### 3.6 `solve_z()` — 全局 CPHF / Z-vector（RHF 与 ROHF）

**物理**：`E` 通过 `γ = Y P Y` 依赖全局密度 `P`，`∂E/∂P = G_P`。`P(R)` 是 SCF 解，
其响应用 Z-vector 消去：`Tr[G_P dP/dR]` 被折叠成"Z × 驻定条件显式导数"。

**RHF**（`_solve_z_rhf`）：
```
moFbar = C^T G_P C
xvo = 2 moFbar[vir, occ]            # ⚠️ 因子 2（踩坑，见 §6.3）
Z = cphf.solve(fvind, mo_energy, mo_occ, xvo)[0]
zvec_ao = C (Z+Z^T) C^T             # 密度响应 → hcore/2e 通道
zeta    = C (Z ⊙ ε) C^T             # 能量加权 → sbar
vhf_s1occ = p_occ v(Zmat) p_occ
```

**ROHF**（`_solve_z_rohf`）：
- 旋转空间 `uniq_ab = {vir-docc, vir-socc, socc-docc}`（`_rohf_rotation_masks`）。
- 轨道 Hessian 用 `newton_ah.gen_g_hop_rohf`，**实测它是真实 Hessian 的一半**，
  所以 Z 方程是 `h_op(Z) = −½ dL/dθ`，打包 RHS 用 `1×moFbar`（不是 2×）。
- 密度响应分自旋：`dPa = C(κ_a+κ_a^T)C^T`（κ_a = Z 散射到 var_a），vir-docc 块双计。
- 折叠（全部数值定标过）：
  ```
  zeta  = Zmat_σ F_σ P_σ                          # zPFP, 系数 −1
  vhf   = ½ Σ_σ P_σ v_σ(W) P_σ                    # v_σ = J−K_σ, 系数 −½
  ```

### 3.7 `make_sbar()` — 总 `dE/dS`（Pulay）

```
sbar = matfun_grad(w_s,u_s,B_X, S^{-1/2})        # X 通道
     + matfun_grad(w_s,u_s,B_Y, S^{+1/2})        # Y 通道
     − ¼ sym(C_occ (moFbar_oo ⊙ (n_i+n_j)) C_occ^T)   # 正交归一占-占（oo）
     − sym(zeta) − sym(vhf_s1occ)                      # Z-vector 折叠
```

**oo 项是本期最重要的修复**：`(n_i+n_j)` 权重来自含 `S^{-1}` 的 Pulay
`−½Tr[dS(P G_P S^{-1}+S^{-1}G_P P)]`。RHF 下 n=2 自动还原 `−2 C moFbar C^T`。
（历史：漏 `S^{-1}` 导致截断 bath ~1e-4 系统误差，详见推导文档 §5.5。）

### 3.8 `contract()` — 与积分导数收缩

```
de += Tr[∂h/∂R · dm_hcore]                       # dm_hcore = dm_core+dm_act+Σ_σ Zmat_σ
de += Tr[∂g/∂R 相关 · D_c, D_act, Z 交叉项] ×2    # 分自旋（ROHF）
de += Tr[∂g/∂R · r2 的嵌入式收缩] ×(−2)          # 簇 2-RDM
de += Tr[∂S/∂R · sbar] ×2                         # Pulay
```

- `∂g/∂R` 用 `mol.intor('int2e_ip1')`（返回完整 5 指标张量，无需 unpack_tril）。
- 簇 2e 导数项符号为**负** `−2·einsum('xui,ui->xu', g2R, C)`（FD 验证过）。
- ROHF 的 Z 两电子通道用分自旋 `veff_σ = J−K_σ`（`mfgrad.get_veff`，
  即 `uhf_grad.get_veff`）；核/活性通道用闭壳层 `_veff_deriv`。

---

## 4. 使用指南

### 4.1 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `conv_tol_cphf` | 1e-9 | Z-vector 求解收敛容差；`≤0` 冻结密度响应（内置近似） |
| `max_cycle_cphf` | 50 | Z-vector 最大迭代 |
| `atmlst` | None | 只算部分原子（如 `[0,1]`） |

### 4.2 体系限制（`check_support` 会拒绝）

| 支持 | 不支持 |
|---|---|
| RHF（闭壳层）、ROHF（开壳层） | UHF、x2c、密度拟合、`bath_option` |
| 普通 Löwdin 定位 | `restore_imp`/`iaopao`（非 Löwdin） |
| threshold 或 `bath_norb` 选 bath | `es_natorb` 在 ROHF 分支未显式禁止（但 gauge 免疫） |

### 4.3 收敛设置（重要）

梯度是密度的**线性**函数，精度受嵌入求解器密度残差限制：

```python
mf.conv_tol = 1e-14
mf.conv_tol_grad = 1e-12      # 会把 es_mf 也收紧（SSDMET.ROHF 已继承）
```

`conv_tol_grad` 是 PySCF 的梯度判据，只有它把密度残差压下去，梯度才能到
`~1e-14`（RHF 实测 9e-15）。只用 `conv_tol`（能量判据）梯度只能到 `~1e-9`。

### 4.4 嵌入选 bath 的坑

- 极小嵌入空间（如 2 轨道 3 电子 spin=1）会得到病态的嵌入 ROHF 解，
  梯度误差 ~1e-4。**这不是梯度 bug**（同一折叠在正常嵌入下到 4.9e-14），
  是嵌入本身的问题。避免这种最小嵌入，或增大 `bath_norb`。

---

## 5. 结构优化

### 5.1 标准方式：scipy BFGS（无第三方依赖，推荐）

`pyscf.geomopt.berny_solver` 依赖第三方 `berny` 包（常未安装），
推荐直接包一个能量+梯度函数给 `scipy.optimize.minimize`：

```python
from scipy.optimize import minimize
from pyscf import gto, scf
from embed_sim import ssdmet


def dmet_energy_and_grad(mol, imp_idx, bath_norb=1, conv_tol=1e-12):
    """返回 (E, grad)，grad 单位 Eh/Bohr。"""
    mf = scf.RHF(mol) if mol.spin == 0 else scf.rohf.ROHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.kernel()
    d = ssdmet.SSDMET(mf, title='g', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    E = d.es_mf.e_tot + d.fo_ene
    de = d.nuc_grad_method().kernel()
    return E, de


def optimize(mol0, imp_idx, bath_norb=1):
    def fg(x):
        m = mol0.copy()
        m.set_geom_(x.reshape(-1, 3), unit='Bohr')     # 坐标用 Bohr，与梯度一致
        m.build()
        E, de = dmet_energy_and_grad(m, imp_idx, bath_norb)
        return float(E), de.ravel()

    res = minimize(fg, mol0.atom_coords().ravel(), jac=True,
                   method='BFGS', options={'gtol': 1e-8, 'maxiter': 50})
    return res.x.reshape(-1, 3), res.fun   # 优化坐标(Bohr)、能量
```

已验证：H2 从 R=3.4 Bohr 优化到平衡键长，`|grad|max` 收敛到 2e-11。
完整可跑示例：`examples/test_example/grad_geomopt.py`。

### 5.2 学习用：最速下降循环

```python
coords = mol0.atom_coords().copy()
for _ in range(nsteps):
    m = mol0.copy(); m.set_geom_(coords, unit='Bohr'); m.build()
    E, de = dmet_energy_and_grad(m, imp_idx, bath_norb)
    coords -= 0.05 * de          # 最速下降（多原子体系慢，学习足够）
```

### 5.3 生产建议

- 大体系 / 慢收敛用 L-BFGS-B 或拟牛顿（`scipy.optimize` 自带）。
- 真实 TM 体系（Co、Fe 等）当前**未支持 x2c**，需等二期（见 §7）。

---

## 6. 验证与调试

### 6.1 与数值梯度对拍（金标准之一）

```python
from embed_sim.grad.numgrad import numerical_grad, ssdmet_energy

gd = d.nuc_grad_method().kernel()
num = numerical_grad(lambda m: ssdmet_energy(m, imp_idx=[0], bath_norb=1,
                                             conv_tol=1e-12),
                     mol, step=1e-3)
err = np.abs(gd - num).max()      # RHF 截断 ~3e-7，ROHF 截断 ~6e-7（FD 截断极限）
```

要更高精度用 Richardson 外推 `(4·g(h/2) − g(h))/3`（ROHF 实测到 ~1e-10）。

### 6.2 三条快速自检（任何改动后都该跑）

1. **full-bath 精确性**：全 bath 时 `E_dmet ≡ E_global`，梯度必须逐位等于
   `mf.nuc_grad_method()`（RHF 9e-15 / ROHF 1e-12）。
2. **平移不变性**：`sum(grad, axis=0) ≈ 0`（机器精度内，任何缺项/符号错都会破坏）。
3. **收敛敏感性**：`conv_tol_grad` 收紧时 `grad_dev` 应随密度残差一起降。

### 6.3 踩过的坑（改代码前必读）

| 坑 | 正确做法 |
|---|---|
| `ao2mo.general(mol,(np.eye(nao),C,C,C))` 当"保留 AO 指标" | 用 `einsum('uvwx,va,wb,xc->uabc', mol.intor('int2e'), C,C,C)` |
| `int2e_ip1` 是压缩的 | PySCF 2.11 返回完整 `(3,nao,nao,nao,nao)`，直接收缩 |
| 簇 2e 导数符号 | `−2·einsum('xui,ui->xu', g2R, C)`（负号，FD 验证） |
| RHF CPHF RHS 因子 | `2·moFbar` 不是 `4·`（cphf.solve+折叠自带 2） |
| `gen_g_hop_rohf` 归一化 | 它是真实 Hessian 的**一半**，Z 方程 `h_op(Z)=−½dL/dθ` |
| Pulay 正交归一权重 | 必须含 `S^{-1}` → `(n_i+n_j)` 权重，不是 `n_j` 或对角 `N` |

### 6.4 诊断脚本（`examples/test_example/`）

| 脚本 | 用途 |
|---|---|
| `grad_dmet_test.py` | RHF 三连测（full-bath 精确、截断 vs FD、平移） |
| `grad_rohf_test.py` | ROHF 三连测（含 Z-RHS 标定、Richardson） |
| `grad_linalg_test.py` | `matfun_grad` / `eig_subspace_grad` 单元测试 |
| `verify_grad_convergence.py` | `grad_dev` 随 `conv_tol_grad` 的收敛行为 |
| `diag_rohf_presp*.py` | 把 P 响应分解成 zdens/zPFP/vhf/oo 块，定位误差来源 |
| `calib_rohf_hessian.py` | 测 `gen_g_hop_rohf` 归一化（比值应 = 2） |

---

## 7. 已知限制与二期

- **求解器**：目前只支持嵌入 RHF（HF-in-HF）。CASCI/CASSCF 是变分的（换 r1/r2
  来源即可）；CCSD/MP2 非变分，需弛豫密度（二期）。
- **参考态**：RHF/ROHF；x2c、密度拟合未支持（真实 TM 体系需要，二期）。
- **性能**：`int2e_ip1`/`int2e` 显式收缩，大体系需 outcore（`ao2mo.general` 流式）。
- **小嵌入空间**：病态（§4.4），需避免。

---

## 附：文件索引

| 路径 | 内容 |
|---|---|
| `embed_sim/grad/ssdmet.py` | 梯度实现（本指南 §1.1, §3） |
| `embed_sim/grad/linalg.py` | 导数原语（§1.2） |
| `embed_sim/grad/numgrad.py` | 数值工具（§1.3, §6.1） |
| `examples/test_example/grad_{dmet,rohf,linalg}_test.py` | 验证（§6.4） |
| `8dmet4reac/DMET_ROHF梯度推导.md` | 完整理论推导与数值锚点 |
