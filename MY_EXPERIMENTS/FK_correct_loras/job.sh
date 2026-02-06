#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=fkc_loras
#SBATCH --output=logs/%j_fkc_out.txt
#SBATCH --error=logs/%j_fkc_err.txt
#SBATCH --time=03:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

# 使用绝对路径运行 Python，并加上 -u 实现实时日志刷新
srun -u /slurm-storage/shucli/.conda/envs/diffusers/bin/python ~/PROJECT_FOLDER/diffusers/MY_EXPERIMENTS/FK_correct_loras/main.py