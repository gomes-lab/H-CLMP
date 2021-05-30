import numpy as np
import torch
import random
from tqdm import tqdm
import gc
import sys
import argparse
import os
from HCLMP.core import train, test

'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

def input_parser():

    parser = argparse.ArgumentParser(
        description=(
            "HCLMP for multiproperty prediction."
        )
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to main data set/training set",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to main data set/training set",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to independent validation set",
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to independent test set"
    )

    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/megnet16-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )

    parser.add_argument(
        "--transfer-type",
        type=str,
        default = 'None',
        choices=['None', 'gen_feat', 'pretrain'],
    )

    parser.add_argument(
        "--gen-feat-dim",
        type=int,
        default = 161,
    )

    parser.add_argument(
        "--feat-dim",
        type=int,
        default = 39,
    )

    parser.add_argument(
        "--label-dim",
        type=int,
        default = 10,
    )

    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )

    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    ) # default value is false

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model",
    )


    parser.add_argument(
        "--lr",
        type=float,
        default = 5e-4,
    )

    parser.add_argument(
        "--decay-times",
        type=int,
        default = 2,
    )

    parser.add_argument(
        "--decay-ratios",
        type=float,
        default = 0.5,
    )

    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda")

    return args

if __name__ == '__main__':

    RNG_SEED = 2
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True

    args = input_parser()

    print(f'Using transfer type {args.transfer_type}')

    assert args.train_path or args.test_path, ('must provide either a train path or test path.')

    if args.train_path:
        sys_name = args.train_path.split('/')[-1].split('.')[0]
    else:
        sys_name = args.test_path.split('/')[-1].split('.')[0]

    args.sys_name = sys_name
    args.save_path = './models/' + sys_name + '/'
    args.result_path = './results/' + sys_name + '/'
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    if args.train:
        train(args)

    if args.evaluate:
        test(args)

