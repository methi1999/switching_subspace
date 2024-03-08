#!/bin/bash

#SBATCH --account=gts-awu36                   # account to which job is charged, ex: GT-gburdell3
#SBATCH -J vae               # name of job
#SBATCH -N1 --ntasks-per-node=4
// #SBATCH -q embers                     # job is submitted to inferno queue
#SBATCH -q inferno                     # job is submitted to inferno queue
#SBATCH -oRes-%j.out
#SBATCH --mail-type=ALL
#SBATCH -t 0-23:59:00

echo "Started on `/bin/hostname`" 
echo $pwd
module load anaconda3/2021.05
conda activate whisker
cd $SLURM_SUBMIT_DIR

python hyperparam.py