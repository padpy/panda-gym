import gym

import panda_gym
from time import sleep

env = gym.make("PandaPickAndPlaceJointsDiscrete-v3", render_mode="human")

observation = env.reset()

for _ in range(1000):
    action = 9
    observation, reward, terminated, info = env.step(action)
    sleep(1/24)

    if terminated:
        observation = env.reset()

env.close()
