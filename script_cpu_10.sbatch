#!/bin/bash

#SBATCH --account=gts-awu36                   # account to which job is charged, ex: GT-gburdell3
#SBATCH -J vae               # name of job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G                        # Memory per core 
#SBATCH -q inferno                     # job is submitted to inferno queue
#SBATCH -oRes-%j.out
#SBATCH --mail-type=NONE
#SBATCH --time 04:00:00

echo "Started on `/bin/hostname`" 
echo $pwd
module load anaconda3/2021.05
conda activate whisker
cd $SLURM_SUBMIT_DIR

# srun runs it as many timeas as n-tasks-per-node 
srun python hyperparam.py --idx 10