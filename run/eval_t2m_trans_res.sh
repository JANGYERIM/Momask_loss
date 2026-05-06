#!/usr/bin/bash

#SBATCH -J MM_Eval_base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A-Eval_M2_exceptRtrans_Re.out


cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
  --vq_name rvq_baseline \
  --name mtrans_baseline \
  --dataset_name t2m \
  --gpu_id 0 \
  --cond_scale 4 \
  --time_steps 10 \
  --which_epoch all

#  --use_res_model \
#--res_name rtrans_baseline \