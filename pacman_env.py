import gym
from gym import wrappers
import numpy as np

env = gym.make('MsPacman-v0')
obs = env.reset()
env.render()
start_time = 90
iteration = 0
ACTION = {
    'NOOP':         0,
    'UP':           1,
    'RIGHT':        2,
    'LEFT':         3,
    'DOWN':         4,
    'UPRIGHT':      5,
    'UPLEFT':       6,
    'DOWNRIGHT':    7,
    'DOWNLEFT':     8
}
def cast_action(action):
    next_obs, rew, done, info = env.step(ACTION[action])
    env.render()
    return next_obs, rew, done, info

while True:
    action = np.random.randint(0,9)
    if iteration >= start_time:
        import ipdb; ipdb.set_trace()
    next_obs, rew, done, info = cast_action('NOOP')
    obs = next_obs
    print("Iteration {}".format(iteration))
    iteration += 1
