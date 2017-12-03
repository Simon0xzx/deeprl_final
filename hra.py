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
    time_name = path.join(logdir, "rha_t.dat")
    mean_name = path.join(logdir, "rha_mean.dat")
    best_name = path.join(logdir, "rha_best.dat")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    times, mean_ep_rewards, best_ep_rewards = [], [], []


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

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######

    q_val = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    q_food, q_avoid, q_fruit, q_eat = q_val
    target_val = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    target_food, target_avoid, target_fruit, target_eat = target_val

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    q_all = 1/4 * (q_food + q_avoid + q_fruit + q_eat)
    action_selected = tf.argmax(q_all, 1)

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

    ######

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
                 var_list=q_func_vars, clip_vb val=grad_norm_clipping)

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

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        idx = replay_buffer.store_frame(last_obs, rha_shape=4)
        epsilon = exploration.value(t)

        if not model_initialized or np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            obs_input = replay_buffer.encode_recent_observation()[None, :]
            action = session.run(action_selected, feed_dict={obs_t_ph:obs_input})
        obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)
        if done: obs = env.reset()
        last_obs = obs

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####

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

                train_loss = (train_food_loss + train_avoid_loss + train_fruit_loss + train_eat_loss)/4.
                print("Loss at iteration {} is: {}".format(t, train_loss))
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
            sys.stdout.flush()

    return times, mean_ep_rewards, best_ep_rewards
