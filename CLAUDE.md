# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a quantum-chemistry research code for **Density Matrix Embedding Theory (DMET)** built on top of [PySCF](https://pyscf.org/). It targets open-shell transition-metal complexes: it partitions a full molecular SCF into an *impurity + bath* embedded cluster and a frozen environment, then solves the cluster with a high-level method (SA-CASSCF, NEVPT2, CCSD(T)) and computes magnetic/spectroscopic properties (spin-orbit coupling, oscillator strengths).

There is no build system, package manager, or test framework. It is a set of Python modules run as scripts against a PySCF/Conda environment.

## Running

Examples are the entry points — each is a standalone PySCF script run directly with Python:

```bash
python examples/dmet.py          # basic single-shot DMET -> SA-CASSCF -> SISO
python examples/DMET_with_df.py  # with density fitting
python examples/bath_expansion.py
```

The `examples/test_example/*.sh` files are SLURM batch scripts (they `source activate mokit-py39` then `python <script>.py > <script>.log`). Reference outputs live alongside them as `*.out` / `*.log`. Use these to sanity-check changes against known-good energies.

**Import path**: scripts import `from embed_sim import ...`, so the repo root must be on `PYTHONPATH` (or run from repo root). Some older scripts hardcode a `sys.path.append(...)` — ignore/remove those paths when adapting.

**Dependencies**: `pyscf`, `numpy`, `scipy`, `sympy` (used in `siso.py` for Wigner 3j), `h5py`. The reference Conda env is `mokit-py39` (Python 3.9).

## Packages

Three parallel DMET implementations coexist, differing by reference wavefunction:

- **`embed_sim/`** — the main, actively developed package. **ROHF/RHF-reference** single-shot DMET. This is what all `examples/` use. Start here.
- **`uhf_dmet/`** — **UHF-reference** DMET variant (spin-unrestricted). Reuses `embed_sim` (`from embed_sim import ssdmet`) plus its own `concen_loc_uhf`, `uhf_tool`. See `uhf_dmet/README.md`.
- **`uhf_dmet_ic/`** — UHF-reference variant with "ic" (internally-contracted / localized-IC) treatment. Near-duplicate of `uhf_dmet` with `sie_cc_loic.py`.

`embed_sim/old/` holds superseded versions (`ssdmet2.py`, `ssdmet3.py`, …) — do not edit or import from these.

## Core architecture (`embed_sim`)

The canonical workflow, chained across modules, is: **full SCF → embed → active-space solver → properties.**

1. **`ssdmet.py` — `SSDMET` class.** The heart of the code.
   - Input: a converged PySCF mean-field/CAS object (`RHF`/`ROHF`/`CAHF`/`CASSCF`) plus an impurity spec `imp_idx` (an AO-label pattern like `'Co.*'` or `'Co 3d'`, resolved via `gto.mole._aolabels2baslst`; defaults to atom 0 if unset).
   - `build()`: Löwdin-orthogonalizes AOs, diagonalizes the environment block of the 1-RDM to split orbitals into **impurity + bath (embedded space)** vs **frozen occupied (`fo_orb`) / frozen virtual (`fv_orb`)**, builds the embedded 1e (`es_int1e`) and 2e (`es_int2e`) integrals, and returns a small ROHF/RHF object `es_mf` living in the embedded space (its `get_hcore`/`get_ovlp`/`_eri` are monkey-patched to the embedded integrals).
   - Results are cached to `<title>_dmet_chk.h5` and reloaded when `dm`/`imp_idx`/`threshold` match (`load_chk`/`save_chk`).
   - Solvers on the cluster: `.avas(...)` (pick active space), `.ccsdt_solver()`, `.mp2_solver()`.
   - `.total_mf()` / `.total_cas(es_cas)` reassemble the embedded solution back into a full-molecule object (frozen orbs + embedded orbs) for downstream property code.
   - `.density_fit()` returns a `DFSSDMET` (see `df.py`) using the mean-field's `with_df`.
   - `bath_option={'MP2': eta}` (also `RMP2`/`ROMP2`/`UMP2`) triggers MP2 bath expansion via `BNO_bath.py` (requires `es_natorb=False`).

2. **`sacasscf_mixer.py`** — builds a **state-averaged CASSCF that mixes multiple spin multiplicities** (`mcscf.state_average_mix_`). `statelis` is an array indexed by `2S` giving the number of roots per multiplicity; `read_statelis(mc)` recovers it from a solver. `sacasscf_nevpt2(...)` adds SC-NEVPT2 corrections. This is the standard cluster solver.

3. **`siso.py` — `SISO` class.** State-interaction spin-orbit coupling on top of the (state-averaged, multi-multiplicity) CAS solution — builds and diagonalizes the SOC Hamiltonian to get magnetic properties. Consumes the full-molecule CAS from `mydmet.total_cas(es_cas)`. `fosc.py`/`examples/fosc.py` compute oscillator strengths / transition dipoles between SOC states.

4. **`cahf.py` — `CAHF` class.** Configuration-averaged HF, an alternative single-determinant reference with fractional-occupation active shell; usable as the `mf_or_cas` input to `SSDMET`.

5. **`spin_utils.py`** — small helpers used everywhere: `unpack_nelec(nelec, spin)`, `gen_statelis(ncas, nelecas)`.

### Supporting modules
- `myavas.py` — AVAS active-space selection adapted to run inside the embedded space (`canonicalize=False`, honoring `ncore`/`nunocc`).
- `df.py` (`DFSSDMET`), `df_uhf_tool.py` — density-fitting variants.
- `BNO_bath.py` — Bath Natural Orbital / MP2 bath expansion (`get_RMP2_bath`, `get_UMP2_bath`, `get_ROMP2_bath`).
- `concentric_loc.py`, `concen_loc_uhf.py` — concentric localization of the environment/virtual space.
- `consistent_bath.py`, `sweep.py`, `pes_scanner.py` — keep the bath definition **consistent across a geometry scan** (PES). `sweep.py` supports "consistent" (fix bath to first geometry) vs "sweep" (grow bath from previous geometry) strategies; used for potential-energy-surface scans.
- `rdiis.py` — RDIIS convergence accelerator for the SCF (helps spin-symmetry-broken cases); see `get_rdiis_property` in `ssdmet.py`.
- `sie_cc.py`, `sie_cc_loic.py`, `uhf_tool.py` — self-energy/CCSD(T) embedding tooling for the UHF variants.
- `aodmet.py` — DMET where the impurity partition is done directly in the AO block (env-only Löwdin), rather than the full-space Löwdin used by `ssdmet.py`.
- `fragment_init_guess.py`, `env_analysis_utils.py`, `spin_utils.py` — initial-guess construction, IAO/environment analysis, spin bookkeeping.

## Conventions and gotchas

- **x2c everywhere**: mean-field objects are built as `scf.ROHF(mol).x2c()`. Consequently `get_hcore()` (not `get_hcore(mol)`) must be used to retrieve the 1e Hamiltonian — the scalar-relativistic term is folded into the object, not recomputable from `mol` alone (see the comment in `make_es_int1e`).
- **Orbital-space naming** in code: `caolo` = AO→Löwdin, `cloao` = Löwdin→AO, `cloes`/`lo2es` = Löwdin→embedded-space, `caoes` = AO→embedded-space, `es_orb` = AO coeffs of the embedded (impurity+bath) orbitals, `fo_orb`/`fv_orb` = frozen occ/virt.
- **`imp_idx` is orbital indices, not atom indices** after the setter runs — it's resolved from an AO-label pattern to a list of AO basis functions.
- **`title`** threads through the whole pipeline as the filename stem for checkpoints (`<title>_rohf.chk`, `<title>_dmet_chk.h5`) and property outputs (`<title>_mag.txt`, molden files). Keep it consistent within a run.
- **Energy reporting**: `ccsdt_solver` reports both a "Direct" total (`e_cluster + fo_ene`, bath-size sensitive) and a "Corr" total (`global_mf + fragment_corr`, recommended for PES). Prefer the correction-based number for smooth curves.
- ROHF density matrices are shape `(2, nao, nao)` (α/β); RHF is `(nao, nao)`. `mf_or_cas_make_rdm1s` normalizes this and `open_shell` flags which path is taken.
