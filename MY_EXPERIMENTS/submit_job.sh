#!/bin/bash
#SBATCH -N 1                          # 申请1个节点
#SBATCH --job-name=sdxl_multi_lora
#SBATCH --output=logs/%j_out.txt      # 确保 logs 文件夹已创建
#SBATCH --error=logs/%j_err.txt
#SBATCH --time=00:20:00               # 20分钟足够跑完
#SBATCH --mem=36000                   # 36G 内存
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU
#SBATCH --qos=short                   # 刚才报错的关键：改为 short
#SBATCH --partition=normal            # 改为 normal

# 防止 .local 库干扰
export PYTHONNOUSERSITE=1

# 使用绝对路径运行 Python，并加上 -u 实现实时日志刷新
srun -u /slurm-storage/shucli/.conda/envs/diffusers/bin/python ~/PROJECT_FOLDER/diffusers/MY_EXPERIMENTS/Two_loras.py