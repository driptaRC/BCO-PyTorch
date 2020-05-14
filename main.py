import torch
import gym
import pickle
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from utils import *
from dataset import *

ENV_NAME = 'Ant-v2'
DEMO_DIR = ''
M = 50000

with open(DEMO_DIR, 'rb') as f:
	demos = pickle.load(f)

env = gym.make(ENV_NAME)

policy = policy_continuous(env.observation_space.shape[0],32,env.action_space.n)
inv_model = inv_model_continuous(env.observation_space.shape[0],100,env.action_space.n)

### GET SAMPLES FOR LEARNING INVERSE MODEL
transitions = get_inv_samples(env, policy, M, 'continuous')

### LEARN THE INVERSE MODEL
inv_dataset = transition_dataset(transitions)
inv_loader = DataLoader(inv_dataset, batch_size=128, shuffle=True, num_workers=4)

inv_opt = optim.Adam(inv_model.parameters(), lr=1e-3)
inv_loss = nn.MSELoss()

for epoch in range(50): 
    for i, data in enumerate(dataloader):
    	s, a, s_prime = data
    	inv_opt.zero_grad()
    	a_pred = inv_model(s, s_prime)
        loss = inv_loss(a_pred, a)
        loss.backward()
        inv_opt.step()

### GET ACTIONS FOR DEMOS
trajs = get_action_labels(inv_model, state_trajs, env_type):
bc_dataset = imitation_dataset(trajs)
bc_loader = DataLoader(bc_dataset, batch_size=128, shuffle=True, num_workers=4)

### PERFORM BEHAVIORAL CLONING
bc_opt = optim.Adam(policy.parameters(), lr=1e-3)
bc_loss = nn.MSELoss()

for epoch in range(50):  
    for i, data in enumerate(dataloader):
    	s, a = data
    	bc_opt.zero_grad()
    	a_pred = policy(s.float())
        loss = bc_loss(a_pred, a)
        loss.backward()
        bc_opt.optimizer.step()

