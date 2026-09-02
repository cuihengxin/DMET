#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J ethane_1bath
#SBATCH -n 16
#SBATCH -o ethane_one_bath_pes.out
##SBATCH -e ethane_one_bath_pes.err

# Reproduce the "one bath orbital per bond" ethane C-C stretch PES.
# Usage: (1) direct  python ethane_one_bath_pes.py
#        (2) SLURM   sbatch run.sh

echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> ethane_one_bath_pes.log
start_ts=$(date +%s)

source activate mokit-py39
export PYTHONPATH=$PYTHONPATH:$PWD/../../..
python ethane_one_bath_pes.py > ethane_one_bath_pes.out 2>&1

end_ts=$(date +%s)
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> ethane_one_bath_pes.log
elapsed=$((end_ts - start_ts))
if date --version >/dev/null 2>&1; then
    printf "elapsed: %s\n" "$(date -ud "@$elapsed" +'%H:%M:%S')" >> ethane_one_bath_pes.log
else
    printf "elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) >> ethane_one_bath_pes.log
fi
