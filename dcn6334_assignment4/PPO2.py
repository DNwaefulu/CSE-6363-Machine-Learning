# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:35:38 2023

@author: dnwae
"""

from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    # is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    height: int  # (vertical dimension) of the observation image or frame
    width: int  # (horizontal dimension) of the observation image or frame
    channels: int  # Representing the red, green, and blue color channels
    action_repeat: int  # Number of times to repeat an action before observing a new frame
    # proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved
    
    
params = Params(
    total_episodes = 3,
    learning_rate = 0.1,
    gamma = 0.95,
    epsilon = 0.4,
    map_size = 5,
    seed = 123,
    n_runs = 20,
    action_size = None,
    state_size = None,
    height = None,
    width = None,
    channels = None,
    action_repeat = None,
    savefig_folder = Path('C:\dnwae\Documents\CSE 6363 Machine Learning'),  # Root folder where plots are saved, kwargs
)
params

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.squeeze(0)[action])

        return action, log_prob

# PPO hyperparameters
gamma = 0.99
clip_ratio = 0.2
ppo_epochs = 10
num_steps = 2048
batch_size = 64

def reinforce(env, policy, optimizer, num_steps, gamma):
    all_rewards = []
    for episode in range(num_steps):
        log_probs = []
        rewards = []
        state = env.reset()

        state = state[0]

        while True:
            action, log_prob = policy.act(state)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            all_rewards.append(reward)

            if terminated or truncated:
                break

        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {sum(rewards)}")

    # Visualize rewards over time
    import matplotlib.pyplot as plt
    plt.plot(all_rewards)
    plt.show()

    
# Create the Atari environment
env = gym.make('ALE/Solaris-v5')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy = Policy(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
reinforce(env, policy, optimizer)
