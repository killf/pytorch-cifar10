import random
import numpy as np
import torch
from torch.backends import cudnn

SEED = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    if SEED is not None:
        set_seed(SEED + worker_id)
