#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=dng_flowmm
#SBATCH --output=dng_flowmm.out
#SBATCH --error=dng_flowmm.err
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

ckpt=runs/trash/2024-06-19/20-37-46/null_params-rfm_cspnet-jgdq0cz6/lightning_logs/version_0/checkpoints/epoch=564-step=59890.ckpt
slope=5.0
python scripts_model/evaluate.py generate ${ckpt} --inference_anneal_slope ${slope} && \
python scripts_model/evaluate.py consolidate ${ckpt} && \
python scripts_model/evaluate.py old_eval_metrics ${ckpt} --stage test && \
python scripts_model/evaluate.py lattice_metrics ${ckpt} --stage test