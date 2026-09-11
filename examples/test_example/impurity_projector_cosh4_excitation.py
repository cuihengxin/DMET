"""Compare Co(SH)4^2- excitation energies for DMET impurity definitions.

The numerical reference is the existing full-system
SA-CASSCF(7e,5o)/SC-NEVPT2/SISO calculation.  The previous embedded method is
the validated AVAS-impurity + ROMP2(eta=1e-6) calculation.  New calculations
use directly projected Co 3d impurity orbitals, either with only the standard
Schmidt bath or with enough fixed-count ROMP2 orbitals to match the previous
52-orbital embedded-space size.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed_sim import sacasscf_mixer, siso, ssdmet
from embed_sim.impurity_projector import project_reference_orbitals
from examples.test_example.impurity_projector_cosh4 import (
    checkpoint_density,
    get_mol,
)


HARTREE_TO_CM = 219474.63
STATELIS = [0, 40, 0, 10]
CASE_OPTIONS = {
    'projected_def2_standard': ('def2tzvp', None),
    'projected_ano_standard': ('ano-r0', None),
    # The direct projector already has 8 standard bath orbitals.  Adding 39
    # ROMP2 BNOs gives 47 total bath / 52 embedded orbitals, exactly matching
    # the size of the previous AVAS+ROMP2 calculation.
    'projected_def2_equal52': ('def2tzvp', 39),
    'projected_ano_equal52': ('ano-r0', 39),
}


def _float_list(values):
    return np.real_if_close(values).real.tolist()


def load_full_system_reference(log_file, mag_file):
    """Read the existing same-Hamiltonian all-electron reference result."""
    text = log_file.read_text()
    state_energies = [float(value) for value in re.findall(
        r'^\s*State\s+\d+\s+weight\s+\S+\s+E\s*=\s*([-+0-9.eE]+)',
        text, re.MULTILINE)]
    corrections = [float(value) for value in re.findall(
        r'^Nevpt2 Energy\s*=\s*([-+0-9.eE]+)', text, re.MULTILINE)]
    rohf_energies = [float(value) for value in re.findall(
        r'^converged SCF energy\s*=\s*([-+0-9.eE]+)',
        text, re.MULTILINE)]
    nstate = sum(STATELIS)
    if (len(state_energies) < nstate or len(corrections) < nstate
            or not rohf_energies):
        raise ValueError(
            f'incomplete reference log: found {len(state_energies)} states '
            f'and {len(corrections)} NEVPT2 corrections')
    return {
        'description': 'full-system SA-CASSCF/SC-NEVPT2/SISO',
        'rohf_energy_Eh': rohf_energies[0],
        'casscf_state_energies_Eh': state_energies[-nstate:],
        'nevpt2_corrections_Eh': corrections[-nstate:],
        'siso_energies_cm-1': np.loadtxt(mag_file).reshape(-1).tolist(),
        'source_log': str(log_file),
        'source_mag': str(mag_file),
    }


def load_previous_method(summary_file, mag_file):
    """Read the validated AVAS-impurity + ROMP2 result."""
    data = json.loads(summary_file.read_text())
    if data.get('completed_stage') != 'siso':
        raise ValueError('previous-method result did not complete SISO')
    if (data.get('system') != 'co'
            or data.get('avas_threshold') != 0.5
            or data.get('romp2_eta') != 1e-6
            or data['casscf']['ncas'] != 5
            or data['casscf']['nelec'] != 7
            or data['casscf']['states'] != STATELIS):
        raise ValueError('previous-method result has incompatible settings')
    return {
        'description': 'AVAS impurity + ROMP2 eta=1e-6',
        'rohf_energy_Eh': float(data['rohf_energy']),
        'nimp': int(data['embedding']['nimp']),
        'nbath': int(data['embedding']['nbath']),
        'nes': int(data['embedding']['nes']),
        'nfo': int(data['embedding']['nfo']),
        'nfv': int(data['embedding']['nfv']),
        'embedding_electrons': int(
            data['embedding']['embedding_electrons']),
        'hf_in_hf_error_Eh': float(data['embedding']['hf_energy_error']),
        'casscf_state_energies_Eh': data['casscf']['state_energies'],
        'nevpt2_corrections_Eh': data['nevpt2_corrections'],
        'siso_energies_cm-1': np.loadtxt(mag_file).reshape(-1).tolist(),
        'source_summary': str(summary_file),
        'source_mag': str(mag_file),
    }


def spectra(result):
    """Build comparable spin-free and spin-orbit excitation spectra."""
    casscf = np.asarray(result['casscf_state_energies_Eh'])
    corrected = casscf + np.asarray(result['nevpt2_corrections_Eh'])
    siso_energy = np.asarray(result['siso_energies_cm-1'])
    if casscf.size != sum(STATELIS) or corrected.size != sum(STATELIS):
        raise ValueError('expected 40 doublet and 10 quartet spin-free states')
    if siso_energy.size != 120 or siso_energy.size % 2:
        raise ValueError('expected 120 SISO levels (60 Kramers pairs)')

    quartet = slice(40, 50)
    kramers = 0.5 * (siso_energy[0::2] + siso_energy[1::2])
    return {
        'casscf_all_cm-1': _float_list(
            (casscf - np.min(casscf)) * HARTREE_TO_CM),
        'nevpt2_all_cm-1': _float_list(
            (corrected - np.min(corrected)) * HARTREE_TO_CM),
        'casscf_quartet_cm-1': _float_list(
            (casscf[quartet] - np.min(casscf[quartet])) * HARTREE_TO_CM),
        'nevpt2_quartet_cm-1': _float_list(
            (corrected[quartet] - np.min(corrected[quartet]))
            * HARTREE_TO_CM),
        'siso_kramers_centers_cm-1': _float_list(kramers),
    }


def error_metrics(values, reference, start=0, stop=None):
    values = np.asarray(values)[start:stop]
    reference = np.asarray(reference)[start:stop]
    if values.shape != reference.shape:
        raise ValueError(
            f'cannot compare spectra with shapes {values.shape} and '
            f'{reference.shape}')
    error = values - reference
    return {
        'start_index': start,
        'stop_index_exclusive': stop,
        'signed_errors_cm-1': _float_list(error),
        'mae_cm-1': float(np.mean(np.abs(error))),
        'rmse_cm-1': float(np.sqrt(np.mean(error**2))),
        'max_abs_error_cm-1': float(np.max(np.abs(error))),
    }


def comparisons(result_spectra, reference_spectra):
    """Errors against the full-system result; ground zeros are excluded."""
    return {
        'casscf_quartet_roots_1_to_9': error_metrics(
            result_spectra['casscf_quartet_cm-1'],
            reference_spectra['casscf_quartet_cm-1'], 1, 10),
        'casscf_all_50_states': error_metrics(
            result_spectra['casscf_all_cm-1'],
            reference_spectra['casscf_all_cm-1'], 0, 50),
        'nevpt2_quartet_roots_1_to_9': error_metrics(
            result_spectra['nevpt2_quartet_cm-1'],
            reference_spectra['nevpt2_quartet_cm-1'], 1, 10),
        'nevpt2_all_50_states': error_metrics(
            result_spectra['nevpt2_all_cm-1'],
            reference_spectra['nevpt2_all_cm-1'], 0, 50),
        'siso_kramers_pairs_1_to_9': error_metrics(
            result_spectra['siso_kramers_centers_cm-1'],
            reference_spectra['siso_kramers_centers_cm-1'], 1, 10),
        'siso_all_60_kramers_pairs': error_metrics(
            result_spectra['siso_kramers_centers_cm-1'],
            reference_spectra['siso_kramers_centers_cm-1'], 0, 60),
    }


def make_projector(mol, basis_name):
    if basis_name == 'def2tzvp':
        reference = {'default': 'minao', 'Co': 'def2tzvp'}
    elif basis_name == 'ano-r0':
        reference = {
            'default': 'minao',
            'Co': gto.basis.load('ano-r0', 'Co'),
        }
    else:
        raise ValueError(f'unknown projector basis {basis_name!r}')
    imp_orb, info = project_reference_orbitals(mol, 'Co 3d', reference)
    info['reference_basis'] = basis_name
    return imp_orb, info


def run_case(mf, case_name, basis_name, romp2_count, output, verbose):
    case_dir = output / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    imp_orb, projector_info = make_projector(mf.mol, basis_name)
    bath_option = (None if romp2_count is None
                   else {'ROMP2': int(romp2_count)})

    old_cwd = Path.cwd()
    os.chdir(case_dir)
    try:
        dmet = ssdmet.SSDMET(
            mf,
            title=str(case_dir / case_name),
            imp_orb=imp_orb,
            bath_option=bath_option,
            es_natorb=False,
            threshold=1e-12,
            verbose=verbose,
        ).density_fit()
        dmet.build(
            save_chk=False,
            chk_fname_load=str(case_dir / 'unused_projector_checkpoint'))

        ncas, nelec, es_mo = dmet.avas(
            'Co 3d', minao='def2tzvp', threshold=0.5,
            openshell_option=3)
        ncas, nelec = int(ncas), int(nelec)
        if (ncas, nelec) != (5, 7):
            raise AssertionError(
                f'{case_name}: expected CAS(7e,5o), got CAS({nelec}e,{ncas}o)')

        mc = sacasscf_mixer.sacasscf_mixer(
            dmet.es_mf, ncas, nelec, statelis=STATELIS)
        mc.max_memory = mf.max_memory
        mc.max_cycle_macro = 100
        mc.verbose = verbose
        mc.kernel(es_mo)
        if not mc.converged:
            raise RuntimeError(f'{case_name}: SA-CASSCF did not converge')
        casscf_energies = np.asarray(mc.e_states).copy()

        corrections = sacasscf_mixer.sacasscf_nevpt2(mc)
        if (corrections.shape != casscf_energies.shape
                or not np.isfinite(corrections).all()):
            raise RuntimeError(f'{case_name}: invalid NEVPT2 corrections')
        mc.fcisolver.e_states = casscf_energies + corrections

        total_mc = dmet.total_cas(mc)
        siso_job = siso.SISO(
            str(case_dir / case_name), total_mc,
            save_mag=True, verbose=verbose)
        siso_job.kernel()
        siso_energies = np.asarray(siso_job.mag_ene)

        result = {
            'description': (
                f'direct {basis_name} Co 3d projector + '
                + ('standard bath only' if romp2_count is None
                   else f'{romp2_count} fixed-count ROMP2 BNOs')),
            'projector': projector_info,
            'romp2_additional_count': romp2_count,
            'nimp': int(imp_orb.shape[1]),
            'nbath': int(dmet.nes - imp_orb.shape[1]),
            'nes': int(dmet.nes),
            'nfo': int(dmet.nfo),
            'nfv': int(dmet.nfv),
            'embedding_electrons': int(dmet.es_mf.mol.nelectron),
            'hf_in_hf_error_Eh': float(
                dmet.es_mf.e_tot + dmet.fo_ene - mf.e_tot),
            'casscf_converged': bool(mc.converged),
            'casscf_ncas': ncas,
            'casscf_nelec': nelec,
            'casscf_statelis': STATELIS,
            'casscf_state_energies_Eh': _float_list(casscf_energies),
            'nevpt2_corrections_Eh': _float_list(corrections),
            'siso_energies_cm-1': _float_list(siso_energies),
        }
        (case_dir / 'case_result.json').write_text(
            json.dumps(result, indent=2) + '\n')
        return result
    finally:
        os.chdir(old_cwd)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = load_full_system_reference(
        args.reference_log.resolve(), args.reference_mag.resolve())
    previous = load_previous_method(
        args.previous_summary.resolve(), args.previous_mag.resolve())
    reference['spectra'] = spectra(reference)
    previous['spectra'] = spectra(previous)
    previous['comparison_to_full_system'] = comparisons(
        previous['spectra'], reference['spectra'])

    mol = get_mol(args.verbose)
    mf = scf.ROHF(mol).x2c().density_fit()
    mf.chkfile = str(output / 'CoSH4_excitation_rohf.chk')
    mf.level_shift = 0.1
    mf.max_cycle = 1000
    mf.max_memory = args.memory
    dm0 = checkpoint_density(mol, args.init_chk.resolve())
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError('full-system CoSH4 ROHF did not converge')
    for name, energy in (
            ('full-system reference', reference['rohf_energy_Eh']),
            ('previous method', previous['rohf_energy_Eh'])):
        if abs(mf.e_tot - energy) > 1e-6:
            raise ValueError(
                f'{name} ROHF energy {energy:.12f} does not match the '
                f'current calculation {mf.e_tot:.12f}')

    report = {
        'system': 'Co(SH)4^2-',
        'comparison_basis': (
            'same parent Hamiltonian and CAS(7e,5o); full-system '
            'SA-CASSCF/SC-NEVPT2/SISO is the numerical reference'),
        'rohf_energy_Eh': float(mf.e_tot),
        'reference': reference,
        'previous_method': previous,
        'new_cases': {},
    }
    result_file = output / 'impurity_projector_cosh4_excitation_results.json'

    for case_name in args.cases:
        basis_name, romp2_count = CASE_OPTIONS[case_name]
        print(f'\n===== excitation case: {case_name} =====', flush=True)
        case_result_file = output / case_name / 'case_result.json'
        if args.resume and case_result_file.is_file():
            print('reuse completed case:', case_result_file, flush=True)
            result = json.loads(case_result_file.read_text())
        else:
            result = run_case(
                mf, case_name, basis_name, romp2_count, output, args.verbose)
        result['spectra'] = spectra(result)
        result['comparison_to_full_system'] = comparisons(
            result['spectra'], reference['spectra'])
        report['new_cases'][case_name] = result
        result_file.write_text(json.dumps(report, indent=2) + '\n')

    metric_path = ('comparison_to_full_system',
                   'siso_kramers_pairs_1_to_9', 'mae_cm-1')

    def metric(item):
        value = item
        for key in metric_path:
            value = value[key]
        return value

    old_mae = metric(previous)
    report['ranking_by_low_siso_mae_cm-1'] = sorted(
        [{'method': 'previous_avas_romp2', 'mae_cm-1': old_mae}]
        + [{'method': name, 'mae_cm-1': metric(result)}
           for name, result in report['new_cases'].items()],
        key=lambda item: item['mae_cm-1'])
    for name, result in report['new_cases'].items():
        result['better_than_previous_low_siso'] = metric(result) < old_mae
    result_file.write_text(json.dumps(report, indent=2) + '\n')

    print('\nExcitation comparison to full-system reference')
    print('method                         nes  low-SISO MAE/cm-1  '
          'quartet-NEVPT2 MAE/cm-1')
    rows = [('previous_avas_romp2', previous)] + list(
        report['new_cases'].items())
    for name, result in rows:
        print(
            f'{name:30s} {result["nes"]:4d} '
            f'{result["comparison_to_full_system"]["siso_kramers_pairs_1_to_9"]["mae_cm-1"]:18.3f} '
            f'{result["comparison_to_full_system"]["nevpt2_quartet_roots_1_to_9"]["mae_cm-1"]:24.3f}')
    print('result:', result_file)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--init-chk', type=Path,
        default=ROOT / 'examples/iao_test/CoSH4_rohf.chk')
    parser.add_argument(
        '--reference-log', type=Path,
        default=ROOT / 'examples/iao_test/CoSH4full.log')
    parser.add_argument(
        '--reference-mag', type=Path,
        default=ROOT / 'examples/iao_test/CoSH4_mag.txt')
    parser.add_argument(
        '--previous-summary', type=Path,
        default=ROOT.parent / '05avas_dmet/results/co_full/summary.json')
    parser.add_argument(
        '--previous-mag', type=Path,
        default=(ROOT.parent
                 / '05avas_dmet/results/co_full/CoSH4_avas_imp_mag.txt'))
    parser.add_argument(
        '--output', type=Path,
        default=(Path(__file__).resolve().parent
                 / 'impurity_projector_cosh4_excitation_run'))
    parser.add_argument(
        '--cases', nargs='+', choices=tuple(CASE_OPTIONS),
        default=list(CASE_OPTIONS))
    parser.add_argument(
        '--resume', action='store_true',
        help='reuse a completed case_result.json instead of rerunning it')
    parser.add_argument('--memory', type=int, default=4000)
    parser.add_argument('--verbose', type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
