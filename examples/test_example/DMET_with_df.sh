#!/bin/bash
#SBATCH --partition=amd
#SBATCH -N 1
#SBATCH -n 16
##SBATCH -c 16
#SBATCH -o DMET_with_df.out
#SBACTH -e DMET_with_df.err
##SBATCH -w node06
##SBATCH --mem=32GB


#touch JobProcessing.state
#echo `date` >> JobProcessing.state 

python3 DMET_with_df.py >DMET_with_df.log

#echo `date` >> $HOME/finish
#echo `pwd` >> $HOME/finish
#echo `date` >> JobProcessing.state