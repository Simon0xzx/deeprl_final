from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import argparse
import torch
from torch.multiprocessing import Process
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from gym.configuration import undo_logger_setup
from optimizer import SharedAdam
import time

undo_logger_setup()

# USE_CUDA = torch.cuda.is_available()
# If you don't want to use GPU, uncomment the following
# USE_CUDA = False

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=1,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='MsPacman-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--save-score-level',
    type=int,
    default=20,
    metavar='SSL',
    help='reward score test evaluation must get higher than to save model')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--count-lives',
    default=False,
    metavar='CL',
    help='end of life is end of training episode.')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')

if __name__ == '__main__':
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)

    setup_json = read_config(args.env_config)
    env_conf = setup_json["MsPacman"]
    env = atari_env(args.env, env_conf)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if args.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env))
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []
    for rank in range(0, args.workers):
        # 'FOOD'->0,
        # 'FRUIT'->1,
        # 'EAT'->2,
        # 'GHOST'->3
        for reward_type in range(4):
            p = Process(
                target=train, args=(rank, reward_type, args, shared_model, optimizer, env_conf))
            p.start()
            processes.append(p)
            time.sleep(0.01)
    for p in processes:
        time.sleep(0.01)
        p.join()