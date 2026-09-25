"""Global runtime state and hyperparameter registry for IDDIFF."""

import torch, os, logging, random
import numpy as np
from parse import parse_args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pLog(s: str): logging.info(s)

CORES = 16
os.environ['NUMEXPR_MAX_THREADS'] = '16'



DATA_PATH = r'D:\postgraduate\code\IDDIFF\IDDIFF-main\data\processed'
FILE_PATH = r'D:\postgraduate\code\IDDIFF\IDDIFF-main\checkpoints'

ARG = parse_args()
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"
if ARG.log is not None: logging.basicConfig(filename=ARG.log, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
else: logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

SAVE = ARG.save
LOAD = ARG.load
SEED = ARG.seed
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
EPOCH = ARG.epoch
PATH = ARG.path
dataset = ARG.dataset
patience = ARG.patience

seed_torch(SEED)
os.makedirs(FILE_PATH, exist_ok=True)

dist_mat = torch.from_numpy(np.load(os.path.join(DATA_PATH, dataset.upper(), 'dist_mat.npy')))
device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
# Shared settings for all three IDDIFF modules and model optimization.
conf = {'lr': ARG.lr,
        'decay': ARG.decay,
        'num_layer': ARG.layer,
        'num_heads': ARG.num_heads,
        'hidden': ARG.hidden,
        'dropout': ARG.dropout, 'dp': ARG.dp,
        'max_len': ARG.length,
        'interval': ARG.interval,
        'zeta': ARG.zeta,
        'T': ARG.diffsize,
        'beta_min': ARG.beta_min,
        'beta_max': ARG.beta_max,
        'dt': ARG.stepsize,
        'window_size': ARG.window_size,
        'guidance_w': ARG.guidance_w,
        'dbscan_eps': ARG.dbscan_eps,
        'dbscan_min_samples': ARG.dbscan_min_samples,
        'alpha': ARG.alpha,
        'sample_num': ARG.sample_num,
        'interp_steps': ARG.interp_steps

        }
