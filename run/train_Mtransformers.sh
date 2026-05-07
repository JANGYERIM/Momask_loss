#!/usr/bin/bash

#SBATCH -J MM_MTrans_v4-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v12
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Momask_loss1/logs/slurm-%A_mtrans_loss_v4-3.out

cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH
teacher_path=/data4/local_datasets/HumanML3D/Translate/

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name mtrans_v4-3 \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --teacher_train_text_dir ${teacher_path}/TranslateTrainText/version2 \
  --teacher_val_text_dir ${teacher_path}/TranslateValText/version2 \
  --vq_name rvq_baseline \
