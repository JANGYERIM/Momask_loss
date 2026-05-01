#!/usr/bin/bash

#SBATCH -J MM_BothTransBaseline
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-g1
#SBATCH -t 2-0
#SBATCH -o /dev/null

cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name mtrans_baseline \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --vq_name rvq_baseline \
  > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_m_baseline.log 2>&1 &

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_res_transformer.py \
  --name rtrans_baseline \
  --gpu_id 1 \
  --dataset_name t2m \
  --batch_size 64 \
  --vq_name rvq_baseline \
  > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_r_baseline.log 2>&1 &

wait
echo "Both training finished."
