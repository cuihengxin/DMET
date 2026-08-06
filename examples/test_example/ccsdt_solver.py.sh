#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J pywork
#SBATCH -n 16
##SBATCH -c 16
#SBATCH -o ccsdt_solver.py.out
#SBACTH -e ccsdt_solver.py.err
##SBATCH -w node06
##SBATCH --mem=32GB


#touch JobProcessing.state
#echo `date` >> JobProcessing.state 
echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> ccsdt_solver.py.log
start_ts=$(date +%s)
#module load icc/latest
#module load mkl/32/2023.2.0
source activate mokit-py39
python ccsdt_solver.py.py >ccsdt_solver.py.log

end_ts=$(date +%s)
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> ccsdt_solver.py.log
#echo `date` >> $HOME/finish
#echo `pwd` >> $HOME/finish
#echo `date` >> JobProcessing.state
# 格式化为 天 小时:分:秒（有 GNU date 的系统）
elapsed=$((end_ts - start_ts))
if date --version >/dev/null 2>&1; then
    printf "elapsed: %s\n" "$(date -ud "@$elapsed" +'%H:%M:%S')" >> ccsdt_solver.py.log
else
    # 备选：手工计算 H:M:S
    days=$((elapsed/86400)); hh=$(( (elapsed%86400)/3600 )); mm=$(( (elapsed%3600)/60 )); ss=$((elapsed%60))
    if [ $days -gt 0 ]; then
        printf "elapsed: %dd %02d:%02d:%02d\n" $days $hh $mm $ss >> ccsdt_solver.py.log
    else
        printf "elapsed: %02d:%02d:%02d\n" $hh $mm $ss >> ccsdt_solver.py.log
    fi
fi