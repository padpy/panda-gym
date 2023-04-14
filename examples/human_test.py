import gym

import panda_gym
import keyboard
import numpy as np
from time import sleep
import sys

env = gym.make("PandaGraspDense-v3", render_mode="human",)

observation = env.reset()

while True:
    if keyboard.is_pressed("q"):
        break

    action = np.array([0, 0, 0, 0], dtype=np.float32)
    if keyboard.is_pressed("8"):
        action += np.array([-0.05, 0, 0, 0])
    if keyboard.is_pressed("2"):
        action += np.array([0.05, 0, 0, 0])
    if keyboard.is_pressed("4"):
        action += np.array([0, -0.05, 0, 0])
    if keyboard.is_pressed("6"):
        action += np.array([0, 0.05, 0, 0])
    if keyboard.is_pressed("down arrow"):
        action += np.array([0, 0, -0.05, 0])
    if keyboard.is_pressed("up arrow"):
        action += np.array([0, 0, 0.05, 0])
    if keyboard.is_pressed("right arrow"):
        action += np.array([0, 0, 0, -0.05])
    if keyboard.is_pressed("left arrow"):
        action += np.array([0, 0, 0, 0.05])

    sys.stdout.write("\033[K")
    print(f'Reward: {env.env.task.compute_reward(env.env.task.get_achieved_goal(), env.env.task.goal, None)}, Grasped: {env.env.task.is_success(None, None)}')
    sys.stdout.write("\033[F")

    action = np.array(action)
    observation, reward, terminated, info = env.step(action)

    sleep(1/24)

env.close()
