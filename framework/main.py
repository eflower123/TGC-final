import sys
import math
import torch
import ctypes
import datetime
import numpy as np
import argparse
import time
import random
import os
from model import TGCtrain

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

FType = torch.FloatTensor
LType = torch.LongTensor


def main_train(args):
    start = datetime.datetime.now()
    the_train = TGCtrain.TGC(args)
    the_train.train()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))


if __name__ == '__main__':
    data = 'patent'
    k_dict = {'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9, 'dblp': 10,
              'brain': 10, 'patent': 6}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--clusters', type=int, default=k_dict[data])
    # dblp/10, arxivAI/5
    parser.add_argument('--epoch', type=int, default=30)
    # dblp/50, arxivAI/200
    parser.add_argument('--neg_size', type=int, default=5)
    parser.add_argument('--hist_len', type=int, default=3)
    # dblp/5, arxivAI/1
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--no_time_loss', action='store_true', default=False,
                        help='Ablation: disable temporal BCE loss')
    parser.add_argument('--no_node_loss', action='store_true', default=False,
                        help='Ablation: disable KL clustering loss')
    parser.add_argument('--no_res_st', action='store_true', default=False,
                        help='Ablation: disable source-target cosine loss')
    parser.add_argument('--no_res_sh', action='store_true', default=False,
                        help='Ablation: disable source-history cosine loss')
    parser.add_argument('--no_res_sn', action='store_true', default=False,
                        help='Ablation: disable source-negative cosine loss')
    parser.add_argument('--no_batch_loss', action='store_true', default=False,
                        help='Ablation: disable all three structural losses (res_st+res_sh+res_sn)')
    args = parser.parse_args()

    main_train(args)
