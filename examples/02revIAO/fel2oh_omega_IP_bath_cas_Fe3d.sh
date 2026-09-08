#!/bin/bash
#SBATCH -o fel2oh_omega_IP_bath_cas_Fe3d.out
#SBATCH -e fel2oh_omega_IP_bath_cas_Fe3d.err
#SBATCH -p C064M0256G
#SBATCH --qos=low
#SBATCH -J pywork
#SBATCH --nodes=1
#SBATCH -n 48
##SBATCH --mem=240gb


JOB_ID=${SLURM_JOB_ID:-N/A}
WORK_DIR=$PWD
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# 记录输出/日志文件名（根据脚本中的实际名称填写）
LOG_FILE="fel2oh_omega_IP_bath_cas_Fe3d.log"          # 程序日志（由 python fel2oh_omega_IP_bath_cas_Fe3d.py >fel2oh_omega_IP_bath_cas_Fe3d.log 生成）
SLURM_OUT="fel2oh_omega_IP_bath_cas_Fe3d.out"         # Slurm 标准输出文件（-o 指定）
SLURM_ERR="fel2oh_omega_IP_bath_cas_Fe3d.err"         # Slurm 错误输出文件（-e 指定）
# =================================
echo "JobID: $JOB_ID Start: $START_TIME Path: $WORK_DIR Log: $LOG_FILE" >> ~/job_history.txt
echo "$WORK_DIR/$LOG_FILE" >> ~/job_history.txt
echo "# ===============================================================" >> ~/job_history.txt
export TMPDIR=/lustre/home/2501110357/scratch
export PYSCF_TMPDIR=/lustre/home/2501110357/scratch
export HDF5_USE_FILE_LOCKING=FALSE

echo "start time: $START_TIME" >> fel2oh_omega_IP_bath_cas_Fe3d.log
start_ts=$(date +%s)

source activate mokit-py39
python fel2oh_omega_IP_bath_cas_Fe3d.py >fel2oh_omega_IP_bath_cas_Fe3d.log

end_ts=$(date +%s)
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "end time: $END_TIME" >> fel2oh_omega_IP_bath_cas_Fe3d.log

elapsed=$((end_ts - start_ts))
# 计算持续时间
if date --version >/dev/null 2>&1; then
    DURATION=$(date -ud "@$elapsed" +'%H:%M:%S')
    printf "elapsed: %s\n" "$DURATION" >> fel2oh_omega_IP_bath_cas_Fe3d.log
else
    days=$((elapsed/86400)); hh=$(( (elapsed%86400)/3600 )); mm=$(( (elapsed%3600)/60 )); ss=$((elapsed%60))
    if [ $days -gt 0 ]; then
        DURATION=$(printf "%dd %02d:%02d:%02d" $days $hh $mm $ss)
    else
        DURATION=$(printf "%02d:%02d:%02d" $hh $mm $ss)
    fi
    printf "elapsed: %s\n" "$DURATION" >> fel2oh_omega_IP_bath_cas_Fe3d.log
fi

# ========= 将作业及输出文件信息追加到 ~/job_history.txt =========
echo "JobID: $JOB_ID  Path: $WORK_DIR  Start: $START_TIME  End: $END_TIME  Duration: $DURATION  Log: $LOG_FILE" >> ~/job_history.txt
echo "$WORK_DIR/$LOG_FILE" >> ~/job_history.txt
echo "# ===============================================================" >> ~/job_history.txt