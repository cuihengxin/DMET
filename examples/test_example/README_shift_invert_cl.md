# Shift-and-invert (resolvent) CL bath expansion

Implementation: `embed_sim/concentric_loc.py`
Verification: `examples/test_example/shift_invert_cl.py`

## Why

Concentric localization promotes environment orbitals by their **bare coupling**
to the current shell. In the α-family language of
`16bathexpand/theory/CL_BNO_equivalence.tex`,

```
m_α(e) = Σ_pairs |V_pair,e|² / ΔE_pair,e^α
```

CL is α = 0, the energy-optimal greedy ranking is α = 1, and MP2 bath natural
orbitals are α = 2. CL carries **no energy denominator at all**, which is the
main reason BNO wins per promoted orbital.

With `couple_op='fock'` there is a second, sharper mechanism. The cumulative CL
shells span the block Krylov space `K_n(P F P, C_0)`, so the selection filter is
a **polynomial in the pseudo-canonical orbital energy ε**. Lanczos-type
convergence reaches the *extremes* of the spectrum first, so roughly half the
promoted slots go to the highest-lying, most diffuse virtuals — the largest
denominators, the smallest correlation weight. That is the "wasted slots"
failure mode, made concrete.

Shift-and-invert replaces the coupling operator by the resolvent of the same
restricted Fock matrix,

```
R = (P F P − σ)⁻¹
```

so the Krylov filter becomes a polynomial in `1/(ε − σ)`. This puts an energy
denominator into a denominator-free selection **at CL cost** — one small
eigendecomposition of the candidate-block Fock matrix, no correlated
calculation, no MP2.

## Use

```python
from embed_sim import ssdmet, concentric_loc

mydmet = ssdmet.SSDMET(mf, title='x', imp_idx=imp, es_natorb=False)
mydmet.build()

# virtual side
concentric_loc.concentric_localization(
    mydmet, proj_bas='sto-3g', n_shell=2, atoms_A=[0],
    couple_op='shift_invert',      # aliases: 'resolvent', 'si', 'shift-invert'
    sigma=None)                    # None -> 'auto' -> eps_HOMO

# occupied side
concentric_loc.concentric_occ_localization(
    mydmet, proj_bas='sto-3g', n_shell=2, atoms_A=[0],
    couple_op='shift_invert', sigma=None)   # 'auto' -> eps_LUMO
```

`sigma` accepts a float (Hartree) or `'auto'` / `'homo'` / `'lumo'` / `'edge'`.

### Where σ must sit

σ must lie **strictly outside** the candidate spectrum, on the side that makes
`|1/(ε − σ)|` largest for the small-denominator candidates:

| side | candidates | σ | effect |
|---|---|---|---|
| `vir` | FV block | **below** the spectrum (default ε_HOMO) | all weights positive and decreasing in ε → promotes the **low-lying** virtuals |
| `occ` | FO block | **above** the spectrum (default ε_LUMO) | \|weight\| grows with ε → promotes the **shallow** occupied orbitals |

A σ *inside* the spectrum makes the resolvent indefinite and near-singular: one
candidate dominates and the selection turns erratic. That is precisely the
intruder-state pathology shift-and-invert is meant to avoid, so the code
detects it, warns, and relocates σ to the spectrum edge.

### Normalization

The resolvent eigenvalues are affinely mapped onto `[0, 1]`. The CL recursion is
*exactly* invariant under `M -> αM + βI` with `α > 0` — shell and kernel are
mutually orthogonal, so `βI` drops out of every shell SVD and only the singular
*values* are rescaled. What the rescaling fixes is the comparison against the
absolute rank threshold (`threshold=1e-6`).

Two traps here, both hit in practice:

1. **Unit spectral norm is not enough.** As `|σ|` grows all `1/(ε − σ)` become
   equal, so `M → I`, and by the invariance above the identity carries no
   information — the informative part is the *deviation* from the identity,
   which still collapses like `1/|σ|`. Symptom: shell distance decays cleanly
   as `1/|σ|` down to σ = −10², then jumps to **exactly 1.000** at σ = −10³ (a
   distance of exactly 1 is the signature of a rank drop by one: the threshold
   ate a shell vector).
2. **The affine map must not be evaluated from its definition.** For large
   `|σ|`, `w_a − w_min` subtracts two numbers of size `1/|σ|` whose difference
   is of size `1/σ²` — catastrophic cancellation. Use the algebraically
   equivalent, cancellation-free closed form in `resolvent_weights()`:

   ```
   (ε_max − ε_a)(ε_min − σ) / [ (ε_a − σ)(ε_max − ε_min) ]
   ```

## Verification

`python examples/test_example/shift_invert_cl.py` (ethane / 6-31G, impurity =
first carbon). It prints a PASS/FAIL line per check and a summary count.

The design principle: because `R = g(P F P)` with `g` a *function of the
operator it replaces*, the construction comes with exact identities that need no
reference data.

| # | Check | What breaks if it fails |
|---|---|---|
| **T1** | Cumulative shells span `K_n(A, C_0)` for both operators | The shell recursion is not a Krylov construction — Prop. 2.1 does not apply |
| **T2** | **Exhausted** Krylov space identical for `fock` and `shift_invert` | Almost any plumbing bug. `g` is injective on the spectrum, so only the promotion *order* may change, never the exhausted span. Sharpest single test. |
| **T3** | σ → −∞ reproduces plain Fock CL, at **first order in 1/\|σ\|** | The operator build. `(F − σ)⁻¹ = τ(1 + τF + …)`, and the identity term drops out of every shell SVD by orthogonality, so shell *n* must converge to the Fock shell *n*. Asserted as a log-log **rate** (slope ≈ −1), not an absolute floor. Also demonstrates the one-parameter family CL ↔ denominator-weighted. |
| **T4** | Weights positive, non-increasing in ε | σ placement / definiteness |
| **T5** | `C† R_ao C = M` against an independent construction | The AO representation `R_ao = S C M C† S` and the orthonormality it assumes |
| **T6** | `E_es + E_fo − E_HF ≈ 0` at **every** shell count | The reassembly path (`es_int1e`/`es_dm`/`ROHF`/`calc_fo_ene`). Promoting orbitals is a repartition of one total space, so HF-in-HF exactness must survive at every shell count, not just on exhaustion. |
| **T7** | `lo_cloes` orthonormal, `NES + NFO + NFV = nao` | Basis bookkeeping in the reassembly |
| **T8** | Production promoted span == independent reference recursion | The AO ↔ candidate-coordinate plumbing |
| **T10** | σ inside the spectrum is caught and relocated | The safety guard |

**T9** is the physics claim, reported without PASS/FAIL: the ε distribution of
the promoted orbitals. Prediction — Fock CL puts a sizeable fraction above the
FV spectrum midpoint (both Lanczos extremes); shift-invert CL should be near
zero there. This is the figure that makes the "wasted slots" argument visible.

### What the tests do *not* cover

- **Occupied side end-to-end.** T1–T5 are side-agnostic and T10 covers the
  guard, but T6–T9 run the virtual side only. Worse, on this molecule
  `NFO = 0`, so `concentric_occ_localization` never gets exercised with the new
  operator at all. Add the FO analogue on a system with a nonzero FO block.
- **Open shell.** `build_resolvent_op` bails out of the HOMO/LUMO lookup for
  spin-resolved `(2, nmo)` `mo_energy` and falls back to `sigma='edge'`. The
  UHF variant (`concen_loc_uhf.py`) has not been touched.
- **Whether it actually helps.** Correctness ≠ improvement. The acceptance
  criterion is the convergence curve, not these tests.

### This molecule is too small for the physics claim

The correctness checks pass, but T9 shows fock and shift-invert picking
essentially the same orbitals (per-shell overlap 0.988 / 0.979 / 0.984,
identical fractions above the spectrum midpoint). Three reasons, all about the
test system rather than the method:

1. **`NFV = 12`** and `n_shell=3` already promotes all 12 — the space is
   exhausted, so by T2 the two *must* coincide. There is barely any room for
   "which one first" to be a question.
2. **`NFO = 0`** — nothing to expand on the occupied side.
3. **`sigma='homo'` is a blunt filter here**: the resolvent dynamic range
   `w_max/w_min` is only 2.55, versus 24.8 for `sigma='edge'`. Shift-and-invert
   does not remove the two-ended Krylov behaviour, it *relocates* one end onto
   the small-denominator orbitals and *compresses* the other into a tight
   cluster — so the gain scales with how far the resolvent spectrum is spread.
   See `16bathexpand/theory/shift_invert_CL.md` §2.2.

So: this script is the fast correctness suite, **not** the physics comparison.
That needs a bigger basis (FV of order 10², so def2-TZVP-ish), a smaller
impurity (nonzero FO), and a σ scan.

## Next steps (the actual comparison)

1. **Subspace overlap with BNO** at equal promoted count:
   `‖P_CL P_BNO‖_F / √k` for `fock`, `shift_invert`, and `BNO_bath`.
   Prediction: `overlap(shift_invert, BNO) > overlap(fock, BNO)`.
2. **Convergence curves** — error vs `N_EO` for fock-CL / shift-invert-CL /
   BNO. Prediction: shift-invert sits between the two, closer to BNO.
3. **σ scan** — sweep σ from ε_HOMO down to −10⁴ Ha. Combined with T3 this is
   both a correctness check and the paper's one-parameter-family figure,
   interpolating α = 0 (CL) and a denominator-weighted selection.
4. **Cost** — confirm the construction stays at CL cost (one `eigh` of the
   candidate block) versus the local MP2 that BNO requires.

If the curve lands close to BNO, the paper's conclusion strengthens
considerably: CL's disadvantage is not the coupling-driven idea itself, it is
the missing denominator weight — and that can be restored without leaving CL
cost.
