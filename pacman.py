import time

import pygame
import gym


frame_time = 1.0 / 15  # seconds


pygame.init()

env = gym.make('MsPacman-v0')
env.frameskip = 3
env.reset()


# directions mapped on keyboard
actions = {'u': 6, 'i': 1, 'o': 5,
           'j': 3, 'k': 0, 'l': 2,
           'm': 8, ',': 4, '.': 7}
actions = {ord(key): value
           for key, value in actions.items()}


then = 0
done = False
last_key = ord('k')
while True:
    now = time.time()
    if frame_time < now - then:
        if not done:
            _, _, done, _ = env.step(actions[last_key])
        else:
            env.reset()
            done = False
        env.render()
        then = now
        last_key = ord('k')
    event = pygame.event.poll()
    if event.type == pygame.KEYDOWN:
        last_key = event.dict["key"]
