import torch
import gym
import pickle
import numpy as np
import os
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Normal
from models import *
from utils import *
from dataset import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()



ENV_NAME = args.env
policy = torch.load(os.path.join(args.model_path,ENV_NAME+'.pt'))
policy.cpu()
env = gym.make(ENV_NAME)

max_steps = env.spec.timestep_limit


### EVALUATE POLICY

returns  = []
for i in range(args.runs):
    print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        a_mu, a_sigma = policy(torch.from_numpy(obs).float())
        a = Normal(loc=a_mu, scale=a_sigma).sample()
        obs, r, done, _ = env.step(a.detach().numpy())
        if args.render:
            env.render()
        totalr += r
        steps += 1
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
            break
    returns.append(totalr)

print('returns', returns)
print('mean return', np.mean(returns))
print('std of return', np.std(returns))