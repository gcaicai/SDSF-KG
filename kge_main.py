from torch.utils.tensorboard import SummaryWriter
import torch
from kge_runner import KGERunner
import os
import logging
import argparse
import json
import pickle
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def init_dir(args):
    # state_new
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.name + '.log')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='a+')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler]
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(console_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./datasets/FB15K237/FB15K237.pkl', type=str)
    parser.add_argument('--name', default='FB15K237-wo-TransE', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--model', default='TransE', choices=['TransE', 'ComplEx', 'ConvR', 'RGCN', 'RotatE', 'DistMult'])

    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--check_per_epoch', default=10, type=int)

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)

    parser.add_argument('--max_round', default=10000, type=int)
    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    parser.add_argument('--seed', default=12345, type=int)

    # hyper-parameters for convR
    parser.add_argument('--convr_num_filters', default=32, type=int)
    parser.add_argument('--convr_kernel_size', default=2, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_dir(args)
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer
    init_logger(args)
    logging.info(args_str)

    # train
    all_data = pickle.load(open(args.data_path, 'rb'))
    learner = KGERunner(args, all_data)
    learner.train()

