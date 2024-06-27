#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=dng_flowmm
#SBATCH --output=dng_flowmm.out
#SBATCH --error=dng_flowmm.err
#SBATCH --time=24:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive


module load cuda/11.8

conda activate flowmm

which python

ckpt=/scratch/midway3/yifengt/flowmm/runs/trash/2024-06-26/22-35-30/abits_params-rfm_cspnet-477idffz/lightning_logs/version_0/checkpoints/epoch=714-step=75790.ckpt
subdir=/scratch/midway3/yifengt/flowmm/runs/trash/2024-06-26/22-35-30/abits_params-rfm_cspnet-477idffz/lightning_logs/version_0/checkpoints/
slope=5.0
python scripts_model/evaluate.py generate ${ckpt} --subdir ${subdir} --inference_anneal_slope ${slope} && \
python scripts_model/evaluate.py consolidate ${ckpt} --subdir ${subdir} && \
python scripts_model/evaluate.py old_eval_metrics ${ckpt} --subdir ${subdir} --stage test && \
python scripts_model/evaluate.py lattice_metrics ${ckpt} --subdir ${subdir} --stage test

squeue -j $SLURM_JOBID
