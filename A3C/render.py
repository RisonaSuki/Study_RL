import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.policy_mu = nn.Linear(256, action_dim)
        self.policy_sigma = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        mu = torch.tanh(self.policy_mu(x))
        sigma = torch.exp(self.policy_sigma(x))
        value = self.value(x)
        return mu, sigma, value

env = gym.make("Ant-v4",render_mode="rgb_array")
input_dim = env.observation_space.shape[0] 
action_dim = env.action_space.shape[0] 

model = ActorCritic(input_dim, action_dim)
model.load_state_dict(torch.load('./best_model.pth'))

def test_model(model, render=False):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 1000:  # 假设我们测试1000步
        if render:
            env.render()
            print("Rendering")
            time.sleep(0.1)
        mu, sigma, _ = model(obs)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().squeeze()
        action = torch.clamp(action, -1.0, 1.0)
        action = action.detach().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    env.close()
    print(f"Total reward in test: {total_reward}")

test_model(model, render=True)
