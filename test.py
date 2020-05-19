import torch
import gym
import pickle
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Normal
from models import *
from utils import *
from dataset import *

ENV_NAME = 'Ant-v2'
policy = torch.load('ant_policy.pt')
env = gym.make(ENV_NAME)


### EVALUATE POLICY
R = 0
policy.cpu()
obs = env.reset()
t = 0
episode = 0
while episode < 10:
    obs = torch.from_numpy(obs).float()
    a_mu, a_sigma = policy(obs)
    env.render()
    a = Normal(loc=a_mu, scale=a_sigma).sample()
    next_obs, reward, done, _ = env.step(a.detach().numpy())
    R += reward
    t += 1
    if done or t > 5000:
        obs = env.reset()
        episode += 1
        t = 0
    obs = next_obs

print(R/10)

## RANDOM POLICY
R = 0
obs = env.reset()
t = 0
episode = 0
while episode < 10:
    a = env.action_space.sample()
    next_obs, reward, done, _ = env.step(a)
    R += reward
    t += 1
    if done or t > 5000:
        obs = env.reset()
        episode += 1
        t = 0
    obs = next_obs

print(R/10)