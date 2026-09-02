#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -J pywork
#SBATCH -n 16
##SBATCH -c 16
#SBATCH -o bath_expansion.out
#SBACTH -e bath_expansion.err
##SBATCH -w node06
##SBATCH --mem=32GB


#touch JobProcessing.state
#echo `date` >> JobProcessing.state 
echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> bath_expansion.log
#source activate mokit-py39
python bath_expansion.py >bath_expansion.log
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> bath_expansion.log
#echo `date` >> $HOME/finish
#echo `pwd` >> $HOME/finish
#echo `date` >> JobProcessing.state