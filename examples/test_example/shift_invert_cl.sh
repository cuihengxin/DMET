#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J pywork
#SBATCH -n 16
#SBATCH -o shift_invert_cl.out
#SBACTH -e shift_invert_cl.err

echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" > shift_invert_cl.log
python shift_invert_cl.py >> shift_invert_cl.log 2>&1
status=$?
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> shift_invert_cl.log
echo "exit status: $status" >> shift_invert_cl.log
exit $status
