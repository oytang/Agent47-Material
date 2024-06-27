#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=train_null
#SBATCH --output=train_null.out
#SBATCH --error=train_null.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# source ~/.bashrc
module load cuda/11.8
# module load python
# conda init
# conda deactivate
source activate flowmm

which python

python scripts_model/run.py data=jarvis_3d model=null_params