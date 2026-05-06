#!/usr/bin/bash

#SBATCH -J MM_EvalAll
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v5
#SBATCH -t 1-0
#SBATCH -o /dev/null

if [ -z "$SLURM_JOB_ID" ]; then
    result=$(sbatch "$0")
    echo "$result"
    echo "제출 완료"
    exit 0
fi

cd /data/dpfla3573/code/Momask_loss1
export PYTHONPATH=/data/dpfla3573/code/Momask_loss1:$PYTHONPATH

set -e

# ── 공통 설정 ─────────────────────────────────────────────────────────────
GPU_ID=0
DATASET=t2m   

VQ_NAME=rvq_baseline
TRANS_NAME=mtrans_v2
RES_NAME=r_baseline

# ── Step 1: Transformer + Residual 평가 ──────────────────────────────────
/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
    --gpu_id      ${GPU_ID} \
    --dataset_name ${DATASET} \
    --name        ${TRANS_NAME} \
    --vq_name     ${VQ_NAME} \
    --res_name    ${RES_NAME} \
    --use_res_model \
    --which_epoch net_best_fid \
    --ext         text2motion \
    --num_batch   2 \
    --repeat_times 1 \
    --cond_scale  4 \
    --time_steps  18 \
    > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_eval_trans_v2.log 2>&1

# ── Step 2: VQ 평가 ──────────────────────────────────────────────────────
/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_vq.py \
    --gpu_id      ${GPU_ID} \
    --dataset_name ${DATASET} \
    --name        ${VQ_NAME} \
    --which_epoch net_best_fid \
    --ext         default \
    > /data/dpfla3573/code/Momask_loss1/logs/slurm-${SLURM_JOB_ID}_eval_vq.log 2>&1

echo "Both eval finished."
