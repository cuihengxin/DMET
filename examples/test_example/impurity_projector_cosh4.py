"""Co(SH)4^2- test for a projection-basis-defined DMET impurity.

The test compares three five-orbital Co 3d definitions using one converged
ROHF density:

1. AVAS active MOs (occupied/virtual rotations; expected standard bath = 0),
2. direct projection of def2-TZVP Co 3d reference AOs,
3. direct projection of contracted ANO-R0 Co 3d reference AOs.

Only the standard mean-field DMET bath is used.  The test writes a compact
JSON result and verifies projector preservation, the full orbital partition,
and the HF-in-HF exact condition.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyscf
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed_sim import myavas, ssdmet
from embed_sim.impurity_projector import project_reference_orbitals


def get_mol(verbose=4):
    """Geometry and electronic structure used by the existing CoSH4 tests."""
    return gto.M(
        atom="""Co
S 1 2.30186590
S 1 2.30186590 2 109.47122060
S 1 2.30186590 3 109.47122065 2 -120.00000001 0
S 1 2.30186590 4 109.47122060 3 120.00000001 0
H 2 1.30714645 1 109.47121982 4 -60.0 0
H 4 1.30714645 1 109.47121982 3 60.0 0
H 5 1.30714645 1 109.47121982 4 -180.0 0
H 3 1.30714645 1 109.47121982 4 60.0 0
""",
        basis={'Co': 'def2tzvp', 'S': '6-31G*', 'H': '6-31G*'},
        symmetry=False, spin=3, charge=-2, verbose=verbose)


def checkpoint_density(mol, filename):
    """Use a matching ROHF checkpoint only as a new SCF initial density."""
    old, data = scf.chkfile.load_scf(str(filename))
    if (old.nao_nr() != mol.nao_nr() or old.nelectron != mol.nelectron
            or old.spin != mol.spin or old._basis != mol._basis
            or not np.array_equal(old.atom_charges(), mol.atom_charges())
            or not np.allclose(old.atom_coords(), mol.atom_coords(),
                               atol=1e-8, rtol=0)):
        raise ValueError('initial checkpoint does not match the CoSH4 test')
    coeff = np.asarray(data['mo_coeff'])
    occ = np.asarray(data['mo_occ'])
    return np.array([
        (coeff * (occ > 0)) @ coeff.conj().T,
        (coeff * (occ == 2)) @ coeff.conj().T,
    ])


def avas_impurity(mf):
    """Return only the AVAS active columns, for a particle/hole comparison."""
    av = myavas.AVAS(
        mf, ['Co 3d'], minao='def2tzvp', threshold=0.5,
        with_iao=False, openshell_option=3)
    ncas, nelec, mo = av.kernel()
    ncas, nelec = int(ncas), int(nelec)
    start = (mf.mol.nelectron - nelec) // 2
    imp = mo[:, start:start + ncas]
    return imp, {
        'ncas': ncas,
        'nelec': nelec,
        'threshold': float(av.threshold),
        'full_basis_indices': list(range(start, start + ncas)),
        'active_occupied_projection_eigenvalues':
            av.occ_weights[av.occ_weights >= av.threshold].tolist(),
        'active_virtual_projection_eigenvalues':
            av.vir_weights[av.vir_weights >= av.threshold].tolist(),
    }


def run_dmet_case(mf, name, imp_orb, output, source_info):
    """Build one projected-impurity DMET case and collect invariant checks."""
    title = str(output / name)
    dmet = ssdmet.SSDMET(
        mf, title=title, imp_orb=imp_orb,
        es_natorb=False, threshold=1e-12,
        verbose=mf.verbose).density_fit()
    dmet.build(
        save_chk=False,
        chk_fname_load=str(output / ('unused_' + name)))

    s = mf.get_ovlp()
    nimp = imp_orb.shape[1]
    actual_imp = dmet.es_orb[:, :nimp]
    singular_values = np.linalg.svd(
        imp_orb.conj().T @ s @ actual_imp, compute_uv=False)
    partition = np.hstack((dmet.fo_orb, dmet.es_orb, dmet.fv_orb))
    partition_error = np.linalg.norm(
        partition.conj().T @ s @ partition - np.eye(mf.mol.nao_nr()))

    ldm = dmet.cloao @ dmet.dm @ dmet.cloao.conj().T
    env_occ = np.linalg.eigvalsh(ldm[nimp:, nimp:])
    bath_occ = env_occ[
        (env_occ >= dmet.threshold) & (env_occ <= 2 - dmet.threshold)]
    cross_norm = np.linalg.norm(ldm[:nimp, nimp:])
    hf_error = dmet.es_mf.e_tot + dmet.fo_ene - mf.e_tot

    result = {
        'source': source_info,
        'nimp': nimp,
        'standard_bath_count': int(dmet.nes - nimp),
        'embedded_orbitals': int(dmet.nes),
        'frozen_occupied': int(dmet.nfo),
        'frozen_virtual': int(dmet.nfv),
        'density_impurity_environment_norm': float(cross_norm),
        'bath_occupations': np.real_if_close(bath_occ).real.tolist(),
        'impurity_subspace_singular_values': singular_values.tolist(),
        'partition_orth_error': float(partition_error),
        'hf_in_hf_error_Eh': float(hf_error),
        'projector_completion': dmet.impurity_projector_info,
    }

    if singular_values.min() < 1 - 1e-7:
        raise AssertionError(f'{name}: DMET did not preserve the impurity span')
    if partition_error > 1e-7:
        raise AssertionError(f'{name}: incomplete/nonorthogonal partition')
    if abs(hf_error) > 1e-7:
        raise AssertionError(f'{name}: HF-in-HF exact condition failed')
    if result['standard_bath_count'] != len(result['bath_occupations']):
        raise AssertionError(f'{name}: bath count does not match its spectrum')
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_chk = args.init_chk.resolve() if args.init_chk else None

    mol = get_mol(args.verbose)
    mf = scf.ROHF(mol).x2c().density_fit()
    mf.chkfile = str(output / 'CoSH4_projector_rohf.chk')
    mf.level_shift = 0.1
    mf.max_cycle = 1000
    mf.max_memory = args.memory
    dm0 = checkpoint_density(mol, initial_chk) if initial_chk else None
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError('full-system CoSH4 ROHF did not converge')

    avas_orb, avas_info = avas_impurity(mf)
    def2_orb, def2_info = project_reference_orbitals(
        mol, 'Co 3d', {'default': 'minao', 'Co': 'def2tzvp'})
    def2_info['reference_basis'] = 'def2tzvp'
    # Passing parsed PySCF basis data exercises the same API used by a
    # user-defined pGTO contraction, rather than only accepting basis names.
    ano_r0_co = gto.basis.load('ano-r0', 'Co')
    ano_orb, ano_info = project_reference_orbitals(
        mol, 'Co 3d', {'default': 'minao', 'Co': ano_r0_co})
    ano_info['reference_basis'] = 'parsed ANO-R0 Co basis data'

    old_cwd = Path.cwd()
    os.chdir(output)
    try:
        cases = {
            'avas_mo': run_dmet_case(
                mf, 'avas_mo', avas_orb, output, avas_info),
            'projected_def2tzvp': run_dmet_case(
                mf, 'projected_def2tzvp', def2_orb, output, def2_info),
            'projected_ano_r0': run_dmet_case(
                mf, 'projected_ano_r0', ano_orb, output, ano_info),
        }
    finally:
        os.chdir(old_cwd)

    if cases['avas_mo']['standard_bath_count'] != 0:
        raise AssertionError('AVAS-MO control should have zero HF Schmidt bath')
    for name in ('projected_def2tzvp', 'projected_ano_r0'):
        if cases[name]['standard_bath_count'] == 0:
            raise AssertionError(
                f'{name}: direct atomic projector should generate a bath')
        if cases[name]['source']['projection_min'] < 0.99:
            raise AssertionError(
                f'{name}: parent AO basis does not span the reference basis')

    report = {
        'system': 'Co(SH)4^2-',
        'parent_basis': {'Co': 'def2tzvp', 'S': '6-31G*', 'H': '6-31G*'},
        'charge': mol.charge,
        'spin': mol.spin,
        'nao': mol.nao_nr(),
        'nelectron': mol.nelectron,
        'pyscf_version': pyscf.__version__,
        'rohf_energy_Eh': float(mf.e_tot),
        'cases': cases,
    }
    result_file = output / 'impurity_projector_cosh4_results.json'
    result_file.write_text(json.dumps(report, indent=2) + '\n')

    print('\nProjection-basis DMET summary')
    print('case                         nimp  bath  nes   ||D_IE||       HF error/Eh')
    for name, result in cases.items():
        print(f'{name:28s} {result["nimp"]:4d} '
              f'{result["standard_bath_count"]:5d} '
              f'{result["embedded_orbitals"]:4d} '
              f'{result["density_impurity_environment_norm"]:12.5e} '
              f'{result["hf_in_hf_error_Eh"]:15.7e}')
    print('result:', result_file)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--init-chk', type=Path,
        help='matching CoSH4 ROHF checkpoint used only as an initial guess')
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).resolve().parent / 'impurity_projector_cosh4_run')
    parser.add_argument('--memory', type=int, default=4000)
    parser.add_argument('--verbose', type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
