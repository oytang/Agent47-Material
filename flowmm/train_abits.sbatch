#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=train_abits
#SBATCH --output=train_abits.out
#SBATCH --error=train_abits.err
#SBATCH --time=24:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive


module load cuda/11.8

source activate flowmm

which python

python scripts_model/run.py data=jarvis_3d model=abits_params

squeue -j $SLURM_JOBID