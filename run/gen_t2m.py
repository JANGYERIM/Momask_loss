import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
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
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


if __name__ == '__main__':
    # ==================================================================
    # CONFIG — 여기서 직접 수정
    # ==================================================================
    GPU_ID       = 0
    NAME         = 'mtrans_v2'   # M-Transformer 체크포인트명
    TRANS_CKPT   = 'net_best_fid.tar'   # M-Transformer 체크포인트 파일 (latest.tar | net_best_fid.tar | net_ep0050.tar ...)
    RES_NAME     = 'rtrans_baseline'       # R-Transformer 체크포인트명
    DATASET      = 't2m'          # 't2m' | 'kit'
    EXT          = 'mtrans_v2'  # 결과 폴더명 (generation/{EXT}/)
    REPEAT_TIMES = 1
    TIME_STEPS   = 18
    COND_SCALE   = 4.0
    TEMPERATURE  = 0.0
    TOPKR        = 0.9
    SEED         = 10107

    # -------------------------------------------------------
    # IsTrainData = True  : CAPTIONS / MOTION_IDS 를 이 파일에 직접 입력
    #                       MOTION_IDS 의 .npy 를 VQ 인코딩해 실제 토큰 길이 사용
    #                       (length estimator 미사용)
    # IsTrainData = False : 기존 동작 (length estimator 또는 지정 길이)
    # -------------------------------------------------------
    IsTrainData = True

    # IsTrainData = True 일 때만 사용 — 직접 수정
    CAPTIONS = [
        "a man walks forward, kicking up his feet, then turns and goes the opposite direction, still kicking his feet.",
        "a person walks forward with a kicking gait, turns, and walks backward",
    ]
    MOTION_IDS = [    # dataset/HumanML3D/new_joint_vecs/ 아래 파일명 (확장자 제외)
        "001481",
        "001481",
        #"000034",
    ]
    # ==================================================================

    sys.argv = [
        sys.argv[0],
        '--gpu_id',      str(GPU_ID),
        '--name',        NAME,
        '--res_name',    RES_NAME,
        '--dataset_name', DATASET,
        '--ext',         EXT,
        '--repeat_times', str(REPEAT_TIMES),
        '--time_steps',  str(TIME_STEPS),
        '--cond_scale',  str(COND_SCALE),
        '--temperature', str(TEMPERATURE),
        '--topkr',       str(TOPKR),
        '--seed',        str(SEED),
    ]

    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, TRANS_CKPT)

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    if IsTrainData:
        # ----------------------------------------------------------
        # CAPTIONS, MOTION_IDS 를 위에서 직접 입력
        # MOTION_IDS 의 .npy 를 VQ 인코딩 → 실제 토큰 길이 계산
        # length estimator 미사용
        # ----------------------------------------------------------
        assert len(CAPTIONS) == len(MOTION_IDS), "CAPTIONS 와 MOTION_IDS 개수가 일치해야 합니다"
        prompt_list = list(CAPTIONS)

        token_lens_list = []
        gt_ids_list = []
        with torch.no_grad():
            for motion_id in MOTION_IDS:
                motion = np.load(pjoin(model_opt.motion_dir, motion_id + '.npy'))
                m_len  = (len(motion) // 4) * 4        # 4의 배수로 정렬
                motion = motion[:m_len]
                motion_norm   = (motion - mean) / std
                motion_tensor = torch.from_numpy(motion_norm).unsqueeze(0).float().to(opt.device)

                code_idx, _ = vq_model.encode(motion_tensor)  # (1, T_token, Q)
                token_lens_list.append(code_idx.shape[1])
                gt_ids_list.append(code_idx[0, :, 0].cpu().tolist())  # layer-0 GT ids

        token_lens = torch.LongTensor(token_lens_list).to(opt.device)

    else:
        gt_ids_list = None
        # ----------------------------------------------------------
        # [기존 코드] length estimator 또는 직접 지정된 길이 사용
        # ----------------------------------------------------------
        est_length = False
        if opt.text_prompt != "":
            prompt_list.append(opt.text_prompt)
            if opt.motion_length == 0:
                est_length = True
            else:
                length_list.append(opt.motion_length)
        elif opt.text_path != "":
            with open(opt.text_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    infos = line.split('#')
                    prompt_list.append(infos[0])
                    if len(infos) == 1 or (not infos[1].isdigit()):
                        est_length = True
                        length_list = []
                    else:
                        length_list.append(int(infos[-1]))
        else:
            raise "A text prompt, or a file a text prompts are required!!!"
        # print('loading checkpoint {}'.format(file))

        if est_length:
            print("Since no motion length are specified, we will use estimated motion lengthes!!")
            text_embedding = t2m_transformer.encode_text(prompt_list)
            pred_dis = length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
            token_lens = Categorical(probs).sample()  # (b, seqlen)
            # lengths = torch.multinomial()
        else:
            token_lens = torch.LongTensor(length_list) // 4
            token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            # layer-0 pred_ids before residual refinement
            pred_ids_layer0 = mids.cpu()

            # save pred/gt token ids per repeat
            token_records = []
            for k in range(len(captions)):
                t_len = token_lens[k].item()
                pred = pred_ids_layer0[k, :t_len].tolist()
                name = MOTION_IDS[k] if IsTrainData else f"sample_{k}"
                entry = {"name": name, "pred_ids": pred}
                if gt_ids_list is not None:
                    entry["gt_ids"] = gt_ids_list[k]
                token_records.append(entry)
            token_save_path = pjoin(result_dir, f"token_ids_repeat{r}.json")
            with open(token_save_path, 'w') as f:
                import re
                raw = json.dumps(token_records, indent=2)
                compacted = re.sub(
                    r'\[(\s*-?\d+(?:\s*,\s*-?\d+)*\s*)\]',
                    lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']',
                    raw, flags=re.DOTALL
                )
                f.write(compacted)
            print(f"Token IDs saved to {token_save_path}")

            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)