import numpy as np
import torch
import random


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)


## 추가 - DataLoader에서 worker마다 시드 고정
def seed_worker(worker_id):                                                                                                                              
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)                                                                                                                          
      random.seed(worker_seed)