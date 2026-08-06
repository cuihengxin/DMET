#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J pywork
#SBATCH -n 16
##SBATCH -c 16
#SBATCH -o CAHF.out
#SBACTH -e CAHF.err
##SBATCH -w node06
##SBATCH --mem=32GB


#touch JobProcessing.state
#echo `date` >> JobProcessing.state 
echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> CAHF.log
source activate mokit-py39
python CAHF.py >CAHF.log
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> CAHF.log
#echo `date` >> $HOME/finish
#echo `pwd` >> $HOME/finish
#echo `date` >> JobProcessing.state