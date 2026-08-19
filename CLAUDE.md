# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**embed_sim** — a single-shot DMET (Density Matrix Embedding Theory) library built on PySCF for quantum embedding of correlated materials and reactions. This is the primary reference implementation for ongoing PhD research (2025–2030). Recent focus: "one bath orbital per bond" scheme for systematically controlling bath size.

## Running Code

### Prerequisites
- Python 3.13 (base conda env)
- PySCF 2.11.0+
- numpy, scipy, h5py, sympy, basis_set_exchange, ASE

### Examples

Each example script is standalone; run from the repo root:

```bash
# Run a single example
python examples/dmet.py
python examples/DMET_with_df.py
python examples/SA-CAS-NEVPT2.py

# Run "one bath orbital per bond" tests (recent feature)
python examples/test_example/one_bath_per_bond.py
python examples/test_example/one_bath_per_bond_ethane_pes.py

# SLURM wrapper (if on a cluster)
sbatch examples/test_example/one_bath_per_bond.sh
```

Outputs (logs, data, plots) appear in the same directory as the script.

### Testing & Validation

Tests are standalone Python scripts in `examples/test_example/`. Correctness is validated against:
- Generated `.log` output files (energy tables, convergence info)
- Expected `.out` reference files
- Literature values and full-electron comparisons (e.g., RMS PES error)

Check `examples/test_example/README_one_bath_per_bond.md` for detailed results on small molecules (H₂, H₂O, CH₄, etc.).

## Architecture

### embed_sim/ — Core DMET library

| Module | Role |
|---|---|
| `ssdmet.py` | Single-shot DMET: impurity + bath from 1-RDM; RHF/ROHF/CASSCF refs; HDF5 checkpointing |
| `df.py` | Density-fitting variant (`DFSSDMET`, `DFAODMET`); O(N³) MP2, preferred for bath expansion |
| `aodmet.py` | AO-based embedding variant; Löwdin orthogonalization in environment block |
| `BNO_bath.py` | Bath natural orbitals from MP2; systematic expansion via occupation threshold η |
| `consistent_bath.py` | Restores bath orbitals across PES geometries for fixed embedded-space size |
| `bath_selection.py` | "One bath per bond" logic: `partition_env_by_bath_count()` selects bath orbitals by occupation (closest to 1 = strongest entanglement) |
| `myavas.py` | AVAS active-space construction; see also `sacasscf_mixer.py` (SA-CASSCF + NEVPT2), `cahf.py` (CAHF reference) |
| `siso.py` | Spin–orbit coupling via spin-mixing; `spin_utils.py` for utilities |
| `pes_guess.py` | Fragment-based initial guesses for reaction-path embedding |
| `pes_scanner.py` | Geometry scanning helper |
| `iao_helper.py` | IAO (Intrinsic Atomic Orbitals) for impurity definition |
| `ic_helper.py` | Internal-coordinate helpers (recent, for structured geometry input) |

### uhf_dmet_ic/ — UHF-specific variants

Spin-polarized DMET for open-shell systems (e.g., radicals, transition metals):
- `uhf_tool.py` — `SSDMET_uhf` class; SVD-based bath construction per spin channel
- `df_uhf_tool.py` — DF variant for UHF
- `consistent_bath.py` — PES consistency for UHF
- `sie_cc.py`, `sie_cc_loic.py` — Spin-integrated, spin-orbit-corrected CC solvers

### examples/ — Use cases

- `dmet.py` — Basic workflow: molecule → RHF → DMET + CASSCF/NEVPT2 → energy
- `DMET_with_df.py` — Density-fitting variant
- `restart.py` — Load from HDF5 checkpoint and resume
- `bath_expansion.py` — Vary bath size and plot error
- `ccsdt_solver.py` — UCC(SD)(T) embedded solver
- `test_example/` — Validation scripts for small molecules and PES

## Key Workflows

### 1. Minimal DMET calculation

```python
from pyscf import gto, scf
from embed_sim import ssdmet

mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='6-31g')
mf = scf.RHF(mol).run()

# Define impurity (e.g., first H atom)
mydmet = ssdmet.SSDMET(mf, title='H2', imp_idx=[0])
mydmet.build()

# Embedded HF energy
print(f"E_embedded = {mydmet.e_tot}")
```

### 2. With embedded correlated solver (SA-CASSCF + NEVPT2)

```python
from embed_sim import sacasscf_mixer

# Define active space on impurity
ncas, nelec, mo = mydmet.avas('H 1s', minao='6-31g', threshold=0.5)

# Embedded SA-CASSCF
es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(mo)

# Add NEVPT2 correlation
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
```

### 3. One bath per bond (recent feature)

Fixes bath size based on impurity–environment bonds; useful for comparing error across molecules:

```python
# Automatic: count bonds by covalent radii
mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx=[0], bath_norb='per_bond')

# Manual: specify bath count
mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx=[0], bath_norb=2)

mydmet.build()
```

For multi-bond systems (N₂, F₂), automatic counting may be insufficient; raise `bath_norb` manually or fix it for PES sweeps.

### 4. Density fitting (for large systems)

```python
from embed_sim import df

mydmet = df.DFSSDMET(mf, title='ethane', imp_idx='0 C.*', 
                     bath_norb='per_bond')
mydmet.build()

# Note: es_natorb=False recommended to avoid shape issues in DF mode
```

### 5. PES (potential energy surface)

Two-stage verification (see `examples/test_example/one_bath_per_bond_ethane_pes.py`):
- **Stage 1**: HF-in-HF exactness check (should be ~1e-12 for full bath, larger for truncated)
- **Stage 2**: MP2-in-HF PES vs. full-electron MP2

## Important Notes

### Checkpoints
- HDF5 checkpoint `<title>_dmet_chk.h5` stores impurity definition, bath selection, and embedded orbitals.
- Changing `bath_norb` triggers automatic checkpoint rebuild.
- Restart: load from checkpoint and resume (see `examples/restart.py`).

### Common Gotchas

1. **bath_norb too small**: If remaining environment orbitals have occupation in `[0.5, 1.5]` (neither frozen-occupied nor frozen-virtual), `partition_env_by_bath_count()` raises `ValueError` with occupation values. Increase `bath_norb`.

2. **Multi-bond systems**: `bath_norb='per_bond'` uses covalent-radius bonding; see `embed_sim/bath_selection.py` for element list. N₂ (triple bond) and F₂ (strong σ) need larger manual `bath_norb`.

3. **PES with geometric changes**: `bath_norb='per_bond'` recounts bonds at each geometry; for long-bond regimes (>1.98 Å for C–C), bonding count drops. Fix `bath_norb` to equilibrium value for consistent comparison.

4. **DF and es_natorb**: `DFSSDMET`/`DFAODMET` with `es_natorb=True` can fail on embedded RHF; use `es_natorb=False`.

5. **UHF_DMET**: `uhf_dmet_ic/` variants do not yet support `bath_norb` feature. Contact code author if needed.

### Related Code

The `embed_sim` module is also vendored into:
- `8dmet4reac/LAMP_emb/embed_sim` — active variant for lamp/reactivity
- `12reaction/dft_corr/embed_sim` — DFT-corrected CASCI
- `5LiCoO2project/0AIMP-open/src/embed_sim` — LiCoO₂ battery project

**Before editing, check which copies scripts in other folders import.**

## Recent Changes (2026-08)

- **One bath orbital per bond** (`bath_norb` parameter): Fixed bath size based on impurity–environment bond count. See `examples/test_example/one_bath_per_bond.md`.
- **`iao_helper.py`**, **`ic_helper.py`**: New helpers for impurity definition and structured geometry input.
- **Extended test suite**: Small-molecule validation (H₂, LiH, H₂O, CH₄, F₂, N₂, OH) + ethane C–C PES two-stage benchmark.

## References

- **Theory**: Sun & Chan, JCTC **10**, 3784 (2014) — one bath per bond scheme (QC-DMET origin).
- **Thesis/Papers**: See `../../1文献/` and `../../0讨论与汇报/` for embedded theory and local methods literature.
- **PySCF**: https://pyscf.org/ — underlying quantum chemistry kernel.
