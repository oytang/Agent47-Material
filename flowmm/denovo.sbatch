#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=dng_flowmm
#SBATCH --output=dng_flowmm.out
#SBATCH --error=dng_flowmm.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive

# source ~/.bashrc
module load cuda/11.8
module list
# module load python
# conda init
# conda deactivate
conda activate flowmm

which python

ckpt=runs/trash/2024-06-20/01-00-06/abits_params-rfm_cspnet-oqdld5i5/lightning_logs/version_0/checkpoints/epoch=679-step=72080.ckpt
subdir=runs/trash/2024-06-20/01-00-06/abits_params-rfm_cspnet-oqdld5i5/lightning_logs/version_0/checkpoints/
slope=5.0
# python scripts_model/evaluate.py generate ${ckpt} --inference_anneal_slope ${slope} && \
# python scripts_model/evaluate.py consolidate ${ckpt} && \
python scripts_model/evaluate.py old_eval_metrics ${ckpt} --stage test && \
python scripts_model/evaluate.py lattice_metrics ${ckpt} --stage test

squeue -j $SLURM_JOBID
