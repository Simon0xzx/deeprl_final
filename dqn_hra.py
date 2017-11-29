import sys
import os.path as path
import os, joblib, pickle
import gym.spaces
import itertools, time, logz, inspect
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
import matplotlib.pyplot as plt
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def aggregator(q_values, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = layers.fully_connected(q_values, num_outputs=num_actions, activation_fn=None)
        return out

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    # Log the progress during the trainining
    start = time.time()
    logdir = 'pacman_hra_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('hra_result', logdir)
    logz.configure_output_dir(logdir)
    args = inspect.getargspec(q_func)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    file_path = "rha_result"
    time_name = path.join(file_path, "dqn_rha_t.dat")
    mean_name = path.join(file_path, "dqn_rha_mean.dat")
    best_name = path.join(file_path, "dqn_rha_best.dat")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)

    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_food_t_ph         = tf.placeholder(tf.float32, [None])
    rew_fruit_t_ph        = tf.placeholder(tf.float32, [None])
    rew_avoid_t_ph        = tf.placeholder(tf.float32, [None])
    rew_eat_t_ph          = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.

    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    q_val = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    q_food, q_avoid, q_fruit, q_eat = q_val
    target_val = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    target_food, target_avoid, target_fruit, target_eat = target_val
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    target_q_all = tf.concat([target_food, target_avoid, target_fruit, target_eat], 1)
    target_q_total = aggregator(target_q_all, num_actions, scope="target_q_agg", reuse=False)
    action_selected = tf.argmax(target_q_total, 0)
    agg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_agg')

    q_act_food_t_val = tf.reduce_sum(q_food * tf.one_hot(act_t_ph, num_actions), axis=1)
    q_act_avoid_t_val = tf.reduce_sum(q_avoid * tf.one_hot(act_t_ph, num_actions), axis=1)
    q_act_fruit_t_val = tf.reduce_sum(q_fruit * tf.one_hot(act_t_ph, num_actions), axis=1)
    q_act_eat_t_val = tf.reduce_sum(q_eat * tf.one_hot(act_t_ph, num_actions), axis=1)

    y_food_t_val = rew_food_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_food, axis=1)
    y_avoid_t_val = rew_avoid_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_avoid, axis=1)
    y_fruit_t_val = rew_fruit_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_fruit, axis=1)
    y_eat_t_val = rew_eat_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_eat, axis=1)

    food_error = tf.reduce_mean(tf.losses.huber_loss(y_food_t_val, q_act_food_t_val))
    avoid_error = tf.reduce_mean(tf.losses.huber_loss(y_avoid_t_val, q_act_avoid_t_val))
    fruit_error = tf.reduce_mean(tf.losses.huber_loss(y_fruit_t_val, q_act_fruit_t_val))
    eat_error = tf.reduce_mean(tf.losses.huber_loss(y_eat_t_val, q_act_eat_t_val))

    q_weight_val = tf.reduce_sum(target_q_total * tf.one_hot(act_t_ph, num_actions), axis=1)
    q_weight_y = rew_food_t_ph + rew_avoid_t_ph + rew_fruit_t_ph + rew_eat_t_ph
    q_weight_y += gamma * (1 - done_mask_ph) * tf.reduce_max(target_q_total, axis=1) - q_weight_val

    weight_error = tf.reduce_mean(tf.losses.huber_loss(q_weight_y, q_weight_val))
    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_food_fn = minimize_and_clip(optimizer, food_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)
    train_avoid_fn = minimize_and_clip(optimizer, avoid_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)
    train_fruit_fn = minimize_and_clip(optimizer, fruit_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)
    train_eat_fn = minimize_and_clip(optimizer, eat_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    train_weight = minimize_and_clip(optimizer, weight_error,
                 var_list=agg_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    times, mean_ep_rewards, best_ep_rewards = [], [], []

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        idx = replay_buffer.store_frame(last_obs, rha_shape=4)
        epsilon = exploration.value(t)

        if not model_initialized or np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            obs_input = replay_buffer.encode_recent_observation()[None, :]
            action = session.run(action_selected, feed_dict={obs_t_ph:obs_input})

        obs, reward, done = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)
        if done: obs = env.reset()
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch =  replay_buffer.sample(batch_size)
            rew_food_t_batch = rew_t_batch[:, 0]
            rew_fruit_t_batch = rew_t_batch[:, 1]
            rew_avoid_t_batch = rew_t_batch[:, 2]
            rew_eat_t_batch = rew_t_batch[:, 3]

            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                       obs_t_ph: obs_t_batch,
                       obs_tp1_ph: obs_tp1_batch})
                session.run(update_target_fn)
                model_initialized = True

            session.run(train_food_fn, feed_dict={
                            obs_t_ph:obs_t_batch,
                            act_t_ph:act_t_batch,
                            rew_food_t_ph:rew_food_t_batch,
                            obs_tp1_ph:obs_tp1_batch,
                            done_mask_ph:done_mask_batch,
                            learning_rate:optimizer_spec.lr_schedule.value(t)})
            session.run(train_avoid_fn, feed_dict={
                            obs_t_ph:obs_t_batch,
                            act_t_ph:act_t_batch,
                            rew_avoid_t_ph:rew_avoid_t_batch,
                            obs_tp1_ph:obs_tp1_batch,
                            done_mask_ph:done_mask_batch,
                            learning_rate:optimizer_spec.lr_schedule.value(t)})
            session.run(train_fruit_fn, feed_dict={
                            obs_t_ph:obs_t_batch,
                            act_t_ph:act_t_batch,
                            rew_fruit_t_ph:rew_fruit_t_batch,
                            obs_tp1_ph:obs_tp1_batch,
                            done_mask_ph:done_mask_batch,
                            learning_rate:optimizer_spec.lr_schedule.value(t)})
            session.run(train_eat_fn, feed_dict={
                            obs_t_ph:obs_t_batch,
                            act_t_ph:act_t_batch,
                            rew_eat_t_ph:rew_eat_t_batch,
                            obs_tp1_ph:obs_tp1_batch,
                            done_mask_ph:done_mask_batch,
                            learning_rate:optimizer_spec.lr_schedule.value(t)})

            session.run(train_weight, feed_dict={
                            obs_t_ph:obs_t_batch,
                            act_t_ph:act_t_batch,
                            rew_food_t_ph:rew_food_t_batch,
                            rew_avoid_t_ph:rew_avoid_t_batch,
                            rew_fruit_t_ph:rew_fruit_t_batch,
                            rew_eat_t_ph:rew_eat_t_batch,
                            obs_tp1_ph:obs_tp1_batch,
                            done_mask_ph:done_mask_batch,
                            learning_rate:optimizer_spec.lr_schedule.value(t)})

            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)
                train_food_loss = session.run(food_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                     act_t_ph:act_t_batch,
                                                                     rew_food_t_ph:rew_food_t_batch,
                                                                     obs_tp1_ph:obs_tp1_batch,
                                                                     done_mask_ph:done_mask_batch})
                train_avoid_loss = session.run(avoid_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                     act_t_ph:act_t_batch,
                                                                     rew_avoid_t_ph:rew_avoid_t_batch,
                                                                     obs_tp1_ph:obs_tp1_batch,
                                                                     done_mask_ph:done_mask_batch})
                train_fruit_loss = session.run(fruit_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                     act_t_ph:act_t_batch,
                                                                     rew_fruit_t_ph:rew_fruit_t_batch,
                                                                     obs_tp1_ph:obs_tp1_batch,
                                                                     done_mask_ph:done_mask_batch})
                train_eat_loss = session.run(eat_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                     act_t_ph:act_t_batch,
                                                                     rew_eat_t_ph:rew_eat_t_batch,
                                                                     obs_tp1_ph:obs_tp1_batch,
                                                                     done_mask_ph:done_mask_batch})
                train_loss = 0.25 * (train_food_loss + train_avoid_loss + train_fruit_loss + train_eat_loss)
                print("\n \
                       Food loss: {}\n \
                       Avoid loss: {}\n \
                       Fruit loss: {}\n \
                       Eat loss: {}".format(train_food_loss,
                                            train_avoid_loss,
                                            train_fruit_loss,
                                            train_eat_loss))
                print("Average loss at iteration {} is: {}".format(t, train_loss))
            num_param_updates += 1

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            times.append(t)
            mean_ep_rewards.append(mean_episode_reward)
            best_ep_rewards.append(best_mean_episode_reward)

            joblib.dump(value=times, filename = time_name, compress=3)
            joblib.dump(value=mean_ep_rewards, filename = mean_name, compress=3)
            joblib.dump(value=best_ep_rewards, filename = best_name, compress=3)

            logz.log_tabular("Training Time", time.time() - start)
            logz.log_tabular("Loss", train_loss)
            logz.log_tabular("Iteration", t)
            logz.log_tabular("Mean Reward (/100ep)", mean_episode_reward)
            logz.log_tabular("Best Mean Reward", best_mean_episode_reward)
            logz.log_tabular("Episodes", len(episode_rewards))
            logz.log_tabular("Exploration", exploration.value(t))
            logz.log_tabular("Learning Rate", optimizer_spec.lr_schedule.value(t))
            logz.dump_tabular()
            # logz.pickle_tf_vars()

            sys.stdout.flush()


    return times, mean_ep_rewards, best_ep_rewards
