from stable_baselines3 import PPO

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_utils import make_env

env = make_env("adult#001", reward_func="RBRewardNormalized")()

for i in range(1):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"BG: {info['rbppo_bg_mgdl']}")
        print(f"Reward: {info['rbppo_reward_total']}")
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {i + 1}: Total Reward: {total_reward}")
