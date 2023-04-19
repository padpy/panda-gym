import gym
from numpngw import write_apng  # pip install numpngw

import panda_gym

env = gym.make("PandaGraspDiscrete-v3", render_mode="rgb_array")
images = []


observation = env.reset()
images.append(env.render())

for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)
    images.append(env.render())

    if terminated:
        observation = env.reset()
        images.append(env.render())

env.close()

write_apng("stack.png", images, delay=40)
