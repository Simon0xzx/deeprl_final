from __future__ import division
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from agent import Agent
from torch.autograd import Variable
import itertools
import logging
from utils import setup_logger
import time


def train(rank, reward_type, args, shared_model, optimizer, env_conf):
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf)
    env.seed(args.seed + rank)
    
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0

    player = Agent(None, env, args, None, reward_type)
    player.model = A3Clstm(
        player.env.observation_space.shape[0], player.env.action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train()

    for i in itertools.count():
        if i%10==0: print("reward type {0}, iter {1}\n".format(reward_type, i))
        player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()
            if args.count_lives:
                player.check_state()
            if player.done:
                break

        if player.done:
            num_tests += 1
            player.current_life = 0
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            player.eps_len = 0
            player.current_life = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 40)
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()
