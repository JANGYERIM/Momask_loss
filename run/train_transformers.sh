#!/usr/bin/bash

#SBATCH -J MM_BothTrans_v1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v12
#SBATCH -t 2-0
#SBATCH -o /dev/null

cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH
teacher_path=/data4/local_datasets/HumanML3D/Translate/

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name mtrans_v1 \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --teacher_train_text_dir ${teacher_path}/TranslateTrainText/version2 \
  --teacher_val_text_dir ${teacher_path}/TranslateValText/version2 \
  --vq_name rvq_baseline \
  > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_m_v1.log 2>&1 &

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_res_transformer.py \
  --name rtrans_v1 \
  --gpu_id 1 \
  --dataset_name t2m \
  --batch_size 64 \
  --teacher_train_text_dir ${teacher_path}/TranslateTrainText/version2 \
  --teacher_val_text_dir ${teacher_path}/TranslateValText/version2 \
  --vq_name rvq_baseline \
  > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_r_v1.log 2>&1 &

wait
echo "Both training finished."
