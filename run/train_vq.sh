#!/usr/bin/bash

#SBATCH -J MM_RVQ_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v12
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Momask_loss1/logs/slurm-%A_rvq_baseline.out

cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_vq.py \
  --name rvq_baseline \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 256 \
  --num_quantizers 6 \
  --max_epoch 50 \
  --quantize_dropout_prob 0.2

sbatch run/train_transformers.sh
echo "Submitted train_both_transformers.sh"
