import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import hra
import hra_agg_img
from dqn_utils import *
from atari_wrappers import *

def hra_model(img_in, num_actions, scope, reuse=False):
    # decomposite objective into 4 different parts: seek_food, avoid_ghost, seek_fruit, and eat_ghost
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("seek_food"):
            food = layers.fully_connected(out, num_outputs=128,             activation_fn=tf.nn.relu)
            food = layers.fully_connected(food, num_outputs=64,             activation_fn=tf.nn.relu)
            food = layers.fully_connected(food, num_outputs=num_actions,    activation_fn=None)
        with tf.variable_scope("avoid_ghost"):
            avoid = layers.fully_connected(out, num_outputs=128,            activation_fn=tf.nn.relu)
            avoid = layers.fully_connected(avoid, num_outputs=64,           activation_fn=tf.nn.relu)
            avoid = layers.fully_connected(avoid, num_outputs=num_actions,  activation_fn=None)
        with tf.variable_scope("seek_fruit"):
            fruit = layers.fully_connected(out, num_outputs=128,            activation_fn=tf.nn.relu)
            fruit = layers.fully_connected(fruit, num_outputs=64,           activation_fn=tf.nn.relu)
            fruit = layers.fully_connected(fruit, num_outputs=num_actions,  activation_fn=None)
        with tf.variable_scope("eat_ghost"):
            eat = layers.fully_connected(out, num_outputs=128,         activation_fn=tf.nn.relu)
            eat = layers.fully_connected(eat, num_outputs=64,          activation_fn=tf.nn.relu)
            eat = layers.fully_connected(eat, num_outputs=num_actions, activation_fn=None)
        return (food, avoid, fruit, eat), out


def atari_learn(env,
                session,
                num_timesteps, learning_freq = 4):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = hra.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (num_iterations / 4, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    time, mean_ep_reward, best_ep_reward = hra_agg_img.learn(
        env,
        q_func=hra_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=128,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=learning_freq,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()
    return time, mean_ep_reward, best_ep_reward


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env = gym.make(task)
    set_global_seeds(seed)
    env.seed(seed)
    expt_dir = '/tmp/pacman_dqn/hra_video'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env, rha=True)
    return env

def main():
    seed = np.random.randint(1000) # Use a seed of zero (you may want to randomize the seed!)
    env = get_env('MsPacman-v0', seed)
    session = get_session()
    time, mean_ep_reward, best_ep_reward = atari_learn(env, session,
                                    num_timesteps=20000000)
    plt.plot(time, mean_ep_reward)
    plt.plot(time, best_ep_reward)
    plt.legend(["mean_rewards", "best_rewards"], loc='best')
    plt.show()

if __name__ == "__main__":
    main()
