#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J one_bath_per_bond
#SBATCH -n 16
#SBATCH -o one_bath_per_bond.out
##SBATCH -e one_bath_per_bond.err

# Small-molecule tests of the "one bath orbital per bond" bath selection
# (Sun & Chan, JCTC 10, 3784 (2014)), ported from the QC-DMET SN2 test.
#
# Usage:
#   (1) direct:  python one_bath_per_bond.py
#   (2) SLURM:   sbatch one_bath_per_bond.sh

echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> one_bath_per_bond.log
start_ts=$(date +%s)

source activate mokit-py39
export PYTHONPATH=$PYTHONPATH:$PWD/../..
python one_bath_per_bond.py >> one_bath_per_bond.out 2>&1

end_ts=$(date +%s)
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> one_bath_per_bond.log
elapsed=$((end_ts - start_ts))
if date --version >/dev/null 2>&1; then
    printf "elapsed: %s\n" "$(date -ud "@$elapsed" +'%H:%M:%S')" >> one_bath_per_bond.log
else
    printf "elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) >> one_bath_per_bond.log
fi
