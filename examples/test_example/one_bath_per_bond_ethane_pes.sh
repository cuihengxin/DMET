#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J ethane_pes
#SBATCH -n 16
#SBATCH -o one_bath_per_bond_ethane_pes.out
##SBATCH -e one_bath_per_bond_ethane_pes.err

# Ethane C-C stretch PES test for the "one bath orbital per bond" selection.
#
# Usage:
#   (1) direct:  python one_bath_per_bond_ethane_pes.py
#   (2) SLURM:   sbatch one_bath_per_bond_ethane_pes.sh

echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> one_bath_per_bond_ethane_pes.log
start_ts=$(date +%s)

source activate mokit-py39
export PYTHONPATH=$PYTHONPATH:$PWD/../..
python one_bath_per_bond_ethane_pes.py >> one_bath_per_bond_ethane_pes.out 2>&1

end_ts=$(date +%s)
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> one_bath_per_bond_ethane_pes.log
elapsed=$((end_ts - start_ts))
if date --version >/dev/null 2>&1; then
    printf "elapsed: %s\n" "$(date -ud "@$elapsed" +'%H:%M:%S')" >> one_bath_per_bond_ethane_pes.log
else
    printf "elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) >> one_bath_per_bond_ethane_pes.log
fi
