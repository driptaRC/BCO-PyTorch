import torch
import gym
import argparse
import os
import pickle
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import Normal
from models import *
from utils import *
from dataset import *


parser = argparse.ArgumentParser()
parser.add_argument('--expert_path', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--runs', type=int, default=2)
parser.add_argument('--inv_samples', type=int, default=5000)
args = parser.parse_args()


ENV_NAME = args.env
DEMO_DIR = os.path.join(args.expert_path, args.env+'_state.pkl')
M = args.inv_samples

with open(DEMO_DIR, 'rb') as f:
	demos = pickle.load(f)

env = gym.make(ENV_NAME)

policy = policy_continuous(env.observation_space.shape[0],64,env.action_space.shape[0]).cuda()
inv_model = inv_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0]).cuda()

inv_dataset_list = []
use_policy = False
for steps in range(args.runs):
    print('######## STEP %d #######'%(steps+1))
    ### GET SAMPLES FOR LEARNING INVERSE MODEL
    print('Collecting transitions for learning inverse model....')
    if steps > 0:
        use_policy = True


    transitions = gen_inv_samples(env, policy.cpu(), M, 'continuous', use_policy)
    print('Done!')

    ### LEARN THE INVERSE MODEL
    print('Learning inverse model....')
    inv_dataset = transition_dataset(transitions)
    inv_dataset_list.append(inv_dataset)
    inv_dataset_final = ConcatDataset(inv_dataset_list)
    inv_loader = DataLoader(inv_dataset_final, batch_size=256, shuffle=True, num_workers=4)

    inv_opt = optim.Adam(inv_model.parameters(), lr=1e-3, weight_decay=0.0001)
    # inv_loss = nn.MSELoss()
    inv_loss = nn.L1Loss()

    for epoch in range(100): 
        running_loss = 0
        for i, data in enumerate(inv_loader):
            s, a, s_prime = data
            inv_opt.zero_grad()
            a_pred = inv_model(s.float().cuda(), s_prime.float().cuda())
            loss = inv_loss(a_pred, a.float().cuda())
            loss.backward()
            running_loss += loss.item()
            if i%100 == 99:
                print('Epoch:%d Batch:%d Loss:%.5f'%(epoch, i+1, running_loss/100))
                running_loss = 0
            inv_opt.step()
    print('Done!')

    ### GET ACTIONS FOR DEMOS
    inv_model.cpu()
    print('Getting labels for demos....')
    trajs = get_action_labels(inv_model, demos, 'continuous')
    print('Done!')
    bc_dataset = imitation_dataset(trajs)
    bc_loader = DataLoader(bc_dataset, batch_size=256, shuffle=True, num_workers=4)
    inv_model.cuda()

    ### PERFORM BEHAVIORAL CLONING
    print('Learning policy....')
    policy.cuda()
    bc_opt = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0001)
    bc_loss = nn.MSELoss()
    # bc_loss = nn.L1Loss()

    for epoch in range(50):  
        running_loss = 0
        for i, data in enumerate(bc_loader):
            s, a = data
            bc_opt.zero_grad()
            a_mu, a_sigma = policy(s.float().cuda())
            a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
            loss = bc_loss(a_pred, a.cuda())
            running_loss += loss.item()
            loss.backward()
            if i%20 == 19:
                print('Epoch:%d Batch:%d Loss:%.3f'%(epoch, i+1, running_loss/20))
                running_loss = 0
            bc_opt.step()
    print('Done!')

torch.save(policy, ENV_NAME+'.pt')
