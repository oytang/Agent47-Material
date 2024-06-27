#!/bin/bash
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=sun
#SBATCH --output=sun.out
#SBATCH --error=sun.err
#SBATCH --time=24:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --nodes=1
#SBATCH --exclusive


module load cuda/11.8

conda activate flowmm

which python

ckpt=/scratch/midway3/yifengt/flowmm/runs/trash/2024-06-26/22-35-30/abits_params-rfm_cspnet-477idffz/lightning_logs/version_0/checkpoints/epoch=714-step=75790.ckpt
subdir=/scratch/midway3/yifengt/flowmm/runs/trash/2024-06-26/22-35-30/abits_params-rfm_cspnet-477idffz/lightning_logs/version_0/checkpoints/
slope=5.0

# get the path to the structures 
eval_for_dft_pt=$(python scripts_model/evaluate.py consolidate "${ckpt}" --subdir "${subdir}" --path_eval_pt eval_for_dft.pt | tail -n 1)

# get the eval_for_dft_json
parent=${eval_for_dft_pt%/*}  # retain part before the last slash
eval_for_dft_json="${eval_for_dft_pt%.*}.json"  # retain part before the period, add .json
log_dir="${parent}/chgnet_log_dir"

# set other flags, if you are using slurm.
num_jobs=20
slurm_partition=YOUR_SLURM_PARTITION

# # prerelax
# python scripts_analysis/prerelax.py "$eval_for_dft_pt" "$eval_for_dft_json" "$log_dir" --num_jobs "$num_jobs"

json_e_above_hull="${parent}/ehulls.json"
# python scripts_analysis/ehull.py "${eval_for_dft_json}" "${json_e_above_hull}"

sun_json=sun.json
python scripts_analysis/novelty.py "${eval_for_dft_json}" "${sun_json}" --ehulls "${json_e_above_hull}" --e_above_hull_column "e_hull_per_atom"



