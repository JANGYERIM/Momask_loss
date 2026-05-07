import os, sys                                                                                                                  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                                                                                                    
import numpy as np                                                                                                              
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric                                                                               
from utils.paramUtil import t2m_kinematic_chain
from models.vq.model import RVQVAE
from utils.get_opt import get_opt
from tqdm import tqdm

DEVICE = torch.device('cuda:0')                                                                                                 
TOKEN_IDS = list(range(512))   # 비교하고 싶은 토큰들
SEQ_LEN   = 20           # 몇 번 반복할지 (20 = 80프레임)                                                                       
OUT_DIR   = './generation/1token_vis'                                                                                        
os.makedirs(OUT_DIR, exist_ok=True)                                                                                             
                                                                                                                                
# VQ 모델 로드                                                                                                                  
vq_opt = get_opt('./checkpoints/t2m/rvq_baseline/opt.txt', device=DEVICE)
vq_opt.dim_pose = 263                                                                                                           
vq_model = RVQVAE(vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,                                                     
                vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,                                                      
                vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,                                                      
                vq_opt.vq_act, vq_opt.vq_norm)                                                                                
ckpt = torch.load('./checkpoints/t2m/rvq_baseline/model/net_best_fid.tar', map_location='cpu')                                  
vq_model.load_state_dict(ckpt['vq_model' if 'vq_model' in ckpt else 'net'])                                                     
vq_model.eval().to(DEVICE)                                                                                                      
                                                                                                                                
mean = np.load('./checkpoints/t2m/rvq_baseline/meta/mean.npy')                                                                  
std  = np.load('./checkpoints/t2m/rvq_baseline/meta/std.npy')
                                                                                                                                
for tid in tqdm(TOKEN_IDS):
    with torch.no_grad():
        # (1, SEQ_LEN, num_quantizers) — layer-0만 해당 토큰, 나머지는 0                                                        
        ids = torch.zeros(1, SEQ_LEN, vq_opt.num_quantizers, dtype=torch.long).to(DEVICE)                                       
        ids[:, :, 0] = tid                                                                                                      
                                                                                                                                
        motion = vq_model.forward_decoder(ids)           # (1, T, 263)                                                          
        motion = motion[0].cpu().numpy() * std + mean    # inverse normalize                                                    
                                                                                                                                
    joint = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()                                                      
    save_path = os.path.join(OUT_DIR, f'token_{tid}.mp4')                                                                       
    plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=f'token {tid}', fps=20)                                         
    print(f'saved: {save_path}')
      
'''
gt_ids = torch.tensor([168, 247, 193, ...], dtype=torch.long)  # GT 시퀀스                                                      
                                                                                                                                  
for swap_token, label in [(168, 'gt'), (368, 'pred')]:                                                                          
    ids = gt_ids.clone()                                                                                                        
    ids[0] = swap_token   # 0번 위치만 교체                                                                                     
                                                                                                                                
    full_ids = torch.zeros(1, len(ids), vq_opt.num_quantizers, dtype=torch.long).to(DEVICE)                                     
    full_ids[0, :, 0] = ids.to(DEVICE)                                                                                          
                                                                                                                                
    with torch.no_grad():
        motion = vq_model.forward_decoder(full_ids)[0].cpu().numpy() * std + mean
    joint = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()                                                      
    plot_3d_motion(f'{OUT_DIR}/swap_{label}.mp4', t2m_kinematic_chain, joint,
                    title=f'pos0={swap_token}', fps=20)
'''