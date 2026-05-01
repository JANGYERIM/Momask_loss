# CLAUDE.md — Momask_loss1 프로젝트 가이드

## 프로젝트 목적

**MoMask 원본 코드 기반 Loss 실험용 레포**

- 원본 레포: https://github.com/EricGuo5513/momask-codes (그대로 클론)
- 기존 `momask-codes_vis`는 건드리지 않고, 이 레포에서 loss 함수를 수정하여 학습 성능 실험 예정
- 가상환경은 기존 `momask` conda 환경 그대로 사용

---

## 브랜치 관리

| 브랜치 | 설명 |
|--------|------|
| `main` | Baseline (원본 MoMask 구조, 파일 재구성 포함) |
| `expect_token_experiment` | Teacher Caption 기반 난이도 가중 Loss 실험 |

---

## 프로젝트 아키텍처

### 모델 파이프라인

```
텍스트 입력 (CLIP 인코딩)
        ↓
  Length Estimator        ← 모션 길이 예측
        ↓
  M-Transformer           ← 마스크 기반 코드북 토큰 생성 (MaskGIT 방식)
        ↓
  R-Transformer           ← Residual 토큰 계층적 정제
        ↓
  RVQ-VAE Decoder         ← 토큰 → 모션 특징 벡터 디코딩
        ↓
  후처리 (Foot IK 등)
        ↓
  출력: NPY / BVH / MP4
```

### 학습 순서

```
1단계: RVQ 토크나이저 학습     (run/train_vq.py)
           ↓
2단계: Masked Transformer 학습  (run/train_t2m_transformer.py)  ─┐ 병렬 가능
3단계: Residual Transformer 학습 (run/train_res_transformer.py) ─┘
```

---

## 폴더 구조

```
Momask_loss1/
├── run/                        # 학습/평가/생성 진입점 (원본은 루트에 있던 파일들)
│   ├── train_vq.py
│   ├── train_t2m_transformer.py
│   ├── train_res_transformer.py
│   ├── eval_t2m_vq.py
│   ├── eval_t2m_trans_res.py
│   ├── gen_t2m.py
│   └── edit_t2m.py
│
├── models/
│   ├── mask_transformer/
│   │   ├── transformer.py          # MaskTransformer, ResidualTransformer
│   │   ├── transformer_trainer.py  # ← Loss 수정 주요 대상
│   │   └── tools.py                # cal_performance, cal_loss ← Loss 수정 주요 대상
│   └── vq/
│       ├── model.py                # RVQVAE, LengthEstimator
│       ├── encdec.py
│       ├── residual_vq.py
│       ├── quantizer.py
│       ├── resnet.py
│       └── vq_trainer.py
│
├── data/
│   └── t2m_dataset.py
│
├── utils/
│   ├── eval_t2m.py             # FID, R-Precision 등 평가 지표
│   ├── motion_process.py
│   ├── plot_script.py
│   ├── paramUtil.py
│   ├── word_vectorizer.py
│   ├── get_opt.py
│   └── fixseed.py
│
├── options/
│   ├── base_option.py
│   ├── train_option.py
│   ├── eval_option.py
│   └── vq_option.py
│
├── visualization/
│   └── joints2bvh.py           # Joint2BVHConvertor
│
├── common/
│   ├── skeleton.py
│   └── quaternion.py
│
├── checkpoints/                # 아래 원본 pretrain 체크포인트 보관
├── dataset/                    # 심볼릭 링크 → /data4/local_datasets/
├── glove/                      # GloVe 임베딩
├── logs/                       # TensorBoard 로그
└── etc/                        # README, requirements, LICENSE
```

---

## 원본 Pretrain 체크포인트 (논문 공식 모델)

`checkpoints/t2m/` (HumanML3D):

| 폴더명 | 역할 |
|--------|------|
| `Comp_v6_KLD005` | 평가용 모션 인코더 (FID 계산) |
| `text_mot_match` | 평가용 텍스트-모션 매칭 모델 |
| `length_estimator` | 모션 길이 예측기 |
| `rvq_nq6_dc512_nc512_noshare_qdp0.2` | RVQ 토크나이저 |
| `t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns` | Masked Transformer |
| `tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw` | Residual Transformer |

`checkpoints/kit/` (KIT-ML):

| 폴더명 | 역할 |
|--------|------|
| `Comp_v6_KLD005` | 평가용 모션 인코더 |
| `text_mot_match` | 평가용 텍스트-모션 매칭 모델 |
| `rvq_nq6_dc512_nc512_noshare_qdp0.2_k` | RVQ 토크나이저 |
| `t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns_k` | Masked Transformer |
| `tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw_k` | Residual Transformer |

> KIT에는 `length_estimator`가 없음 (t2m 것 공유)

---

## 데이터셋

- **HumanML3D:** `dataset/HumanML3D` → symlink → `/data4/local_datasets/HumanML3D/`
- **KIT-ML:** `dataset/KIT-ML` → symlink → `/data4/local_datasets/KIT-ML/`
- NAS 보관 경로: `/data/datasets/HumanML3D.tar.gz`, `/data/datasets/KIT-ML.tar.gz`
- 노드 변경 시 NAS에서 압축 해제 후 symlink 재설정 필요

---

## Loss 관련 핵심 코드 위치

MaskTransformer 학습 loss는 아래 두 파일이 핵심:

```
models/mask_transformer/tools.py
  └── cal_loss()          # Cross-Entropy (label smoothing 옵션)
  └── cal_performance()   # loss + top-k accuracy 계산

models/mask_transformer/transformer.py
  └── MaskTransformer.forward()   # 마스킹 → logits → cal_performance 호출
  └── ResidualTransformer.forward()

models/mask_transformer/transformer_trainer.py
  └── MaskTransformerTrainer.forward()   # 전체 학습 루프 연결
  └── ResidualTransformerTrainer.forward()
```

---

## 실험 검증 방식

### 추론 시 길이 처리 방침

Loss 실험 검증 시 추론은 **Length Estimator를 사용하지 않고 GT 길이를 직접 사용**한다.

- **이유:** Length Estimator 오차가 끼어들면 모션 품질이 나빠도 Loss 변경 탓인지 길이 추정 탓인지 구분 불가
- **방법:** `m_length // 4` (프레임 수 → 토큰 수)를 `trans.generate()`에 직접 전달
- **참고:** `eval_t2m_trans_res.py`가 이미 이 방식 사용 (`utils/eval_t2m.py:935`)

```python
# 검증 추론 시 길이 처리
mids = trans.generate(clip_text, m_length // 4, ...)  # GT 길이 직접 사용
```

실제 서비스 추론(`gen_t2m.py`)에서는 Length Estimator 사용하지만, 실험 검증 단계에서는 GT 길이 고정.

---

## 실험 접근 방법

### 접근 방법 1: Teacher Caption을 통한 텍스트 증강 학습

#### 배경 및 아이디어

HumanML3D 데이터셋에서 모션 1개당 캡션이 2~4개 존재한다. 원본 MoMask는 학습 시 이 중 하나를 랜덤 샘플링하여 사용한다. 각 캡션은 동일 모션의 부분적인 측면만 기술하는 경향이 있어, 개별 캡션만으로는 모션 전체를 충분히 설명하지 못한다.

**핵심 아이디어:** 한 모션의 복수 캡션을 통합하여 모션 전체를 더 잘 설명하는 **Teacher Caption** 하나를 생성하고, 이를 M-Transformer / R-Transformer 학습 시 원본 캡션과 함께 pair로 활용한다.

#### Teacher Caption 생성 방법

- 모션 1개의 2~4개 캡션을 LLM(예: GPT-4)에 입력
- 중복을 제거하고 내용을 통합한 단일 캡션 생성
- 생성 결과는 사전에 전체 데이터셋에 대해 일괄 생성 후 파일로 저장 (학습 중 온라인 생성 아님)
- 저장 형식: 원본 `.txt` 파일과 동일한 인덱스로 매핑

#### 학습 적용 방식


**난이도 가중 loss**


원본 캡션과 teacher 캡션이 **같은 마스크 위치**를 동시에 예측하게 하고,
둘 다 틀린 위치에 더 강한 loss weight를 부여한다.

```
같은 마스크 위치 M 샘플링
        ↓
원본캡션  → logits1 → 위치별 예측
teacher캡션 → logits2 → 위치별 예측
        ↓
둘 다 틀린 위치   → loss weight 높게 (예: ×2)
하나만 틀린 위치  → 기본 weight
둘 다 맞은 위치   → 기본 weight (또는 낮게)
```

이 방식은 **마스크 공유가 필수**이므로, `MaskTransformer.forward()` 내부의
마스크 샘플링 블록을 분리하여 `trans_forward()`를 직접 두 번 호출하도록 리팩토링 필요.

```python
# 마스크를 밖에서 샘플링
mask, x_ids, labels = sample_mask(ids, m_lens, ...)

# 같은 마스크로 두 캡션 각각 forward
logits1 = self.trans_forward(x_ids, cond_orig,    padding_mask)
logits2 = self.trans_forward(x_ids, cond_teacher, padding_mask)

# 둘 다 틀린 위치에 가중치 부여
weight = compute_difficulty_weight(logits1, logits2, labels)
loss = cal_performance_weighted(logits1, logits2, labels, weight)
```

#### 기대 효과

- 모션-텍스트 alignment 강화: 더 완전한 텍스트 표현이 해당 모션 토큰 분포를 더 잘 anchor
- 텍스트 다양성 증가: 같은 모션을 다양한 표현으로 학습 → 추론 시 다양한 프롬프트에 robust
- 난이도 가중 loss: 두 캡션 모두 예측하지 못하는 hard token에 집중 → 모델이 어려운 부분을 더 빠르게 학습

#### 검증 계획

1. 
- 원본 대비 동일 학습 조건에서 비교 (RVQ는 원본 pretrain 체크포인트 고정)
- M-Transformer 단독 → R-Transformer 포함 순서로 효과 분리

#### 구현 시 수정 대상 파일

```
data/t2m_dataset.py
  └── Text2MotionDataset.__getitem__()
      → teacher caption 파일 로드 및 반환 추가

models/mask_transformer/transformer.py
  └── MaskTransformer.forward()
      → [방법] 마스크 샘플링 블록을 별도 함수로 분리
      → trans_forward()를 두 번 호출하는 구조로 변경

models/mask_transformer/tools.py
  └── [방법] cal_performance_weighted() 추가
      → 위치별 difficulty weight를 받아 weighted CE loss 계산

models/mask_transformer/transformer_trainer.py
  └── MaskTransformerTrainer.forward()
  └── ResidualTransformerTrainer.forward()
      → [방법] teacher_conds 전달 및 가중 loss 수신
```

---

## 환경

- Conda 환경: `momask` (기존 환경 그대로 사용)
- Python 3.7.13 / PyTorch 기반
- CLIP: GitHub 소스 설치 (`pip install git+https://github.com/openai/CLIP.git`)
- ffmpeg 필요 (MP4 출력 시)
