"""
각 코드북 토큰(0 ~ nb_code-1)을 100프레임으로 반복 디코딩하여 mp4로 저장.
RVQ layer-0만 사용 (나머지 residual layer는 zero 마스킹).

사용법:
    cd /data/dpfla3573/code/Momask_loss1
    conda activate momask
    python run/visualize_tokens.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import join as pjoin

import torch
import numpy as np

from models.vq.model import RVQVAE
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain

# ============================================================
# CONFIG
# ============================================================
GPU_ID     = 0
VQ_NAME    = 'rvq_baseline'
DATASET    = 't2m'
N_FRAMES   = 120          # 120 프레임 = 30 토큰 (stride 4)
OUTPUT_DIR = './generation/token_vis'
TOKEN_START = 0           # 시작 토큰 ID (일부만 실행할 때)
TOKEN_END   = None        # 끝 토큰 ID (None = 전체)
# ============================================================

device = torch.device('cpu' if GPU_ID == -1 else f'cuda:{GPU_ID}')

# ----- 모델 로드 -----
vq_opt_path = pjoin('./checkpoints', DATASET, VQ_NAME, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=device)
vq_opt.dim_pose = 251 if DATASET == 'kit' else 263

vq_model = RVQVAE(
    vq_opt,
    vq_opt.dim_pose,
    vq_opt.nb_code,
    vq_opt.code_dim,
    vq_opt.output_emb_width,
    vq_opt.down_t,
    vq_opt.stride_t,
    vq_opt.width,
    vq_opt.depth,
    vq_opt.dilation_growth_rate,
    vq_opt.vq_act,
    vq_opt.vq_norm,
)

ckpt = torch.load(
    pjoin('./checkpoints', DATASET, VQ_NAME, 'model', 'net_best_fid.tar'),
    map_location='cpu',
)
model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
vq_model.load_state_dict(ckpt[model_key])
vq_model.eval()
vq_model.to(device)
print(f'Loaded {VQ_NAME}  (nb_code={vq_opt.nb_code}, num_q={vq_opt.num_quantizers})')

# ----- mean / std -----
meta_dir = pjoin('./checkpoints', DATASET, VQ_NAME, 'meta')
mean = np.load(pjoin(meta_dir, 'mean.npy'))
std  = np.load(pjoin(meta_dir, 'std.npy'))

def inv_transform(data):
    return data * std + mean

# ----- 출력 폴더 -----
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- 파라미터 -----
n_tokens       = N_FRAMES // 4          # 25
nb_code        = vq_opt.nb_code         # 512
num_quantizers = vq_opt.num_quantizers  # 6
nb_joints      = 21 if DATASET == 'kit' else 22
kinematic_chain = t2m_kinematic_chain

token_end = nb_code if TOKEN_END is None else TOKEN_END
print(f'Generating token {TOKEN_START} ~ {token_end - 1}  ({token_end - TOKEN_START} videos)')
print(f'N_FRAMES={N_FRAMES}, n_tokens={n_tokens}, output={OUTPUT_DIR}')

with torch.no_grad():
    for token_id in range(TOKEN_START, token_end):
        # indices shape: (1, n_tokens, num_quantizers)
        # layer-0 = token_id, layer-1~5 = -1 (zeroed out in decoder)
        indices = torch.full((1, n_tokens, num_quantizers), -1, dtype=torch.long, device=device)
        indices[0, :, 0] = token_id

        # decode
        pred = vq_model.forward_decoder(indices)        # (1, N_FRAMES, dim_pose)
        data = inv_transform(pred[0].cpu().numpy())     # (N_FRAMES, dim_pose)

        # ric → joint positions
        joint = recover_from_ric(
            torch.from_numpy(data).float(), nb_joints
        ).numpy()                                       # (N_FRAMES, nb_joints, 3)

        # mp4 저장
        save_path = pjoin(OUTPUT_DIR, f'{token_id:04d}.mp4')
        plot_3d_motion(save_path, kinematic_chain, joint,
                       title=f'Token {token_id}', fps=20)

        if token_id % 50 == 0 or token_id == token_end - 1:
            print(f'  [{token_id:4d}/{token_end - 1}] saved → {save_path}')

print('Done.')
