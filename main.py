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
DEMO_DIR = 'Ant-v2_traj_50.pickle'
M = 50000

with open(DEMO_DIR, 'rb') as f:
	demos = pickle.load(f)

env = gym.make(ENV_NAME)

policy = policy_continuous(env.observation_space.shape[0],32,env.action_space.shape[0]).cuda()
inv_model = inv_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0]).cuda()


for steps in range(20):
    print('######## STEP %d #######'%steps)
    ### GET SAMPLES FOR LEARNING INVERSE MODEL
    if steps !=0:
        M = 5000
    print('Collecting transitions for learning inverse model....')
    transitions = gen_inv_samples(env, policy.cpu(), M, 'continuous')
    print('Done!')

    ### LEARN THE INVERSE MODEL
    print('Learning inverse model....')
    inv_dataset = transition_dataset(transitions)
    inv_loader = DataLoader(inv_dataset, batch_size=128, shuffle=True, num_workers=4)

    inv_opt = optim.Adam(inv_model.parameters(), lr=1e-4)
    inv_loss = nn.MSELoss()

    for epoch in range(50): 
        for i, data in enumerate(inv_loader):
            s, a, s_prime = data
            inv_opt.zero_grad()
            a_pred = inv_model(s.float().cuda(), s_prime.float().cuda())
            loss = inv_loss(a_pred, a.float().cuda())
            loss.backward()
            if i%20 == 0:
                print('Epoch:%d Batch:%d Loss:%.5f'%(epoch, i, loss.item()))
            inv_opt.step()
    print('Done!')

    ### GET ACTIONS FOR DEMOS
    inv_model.cpu()
    print('Getting labels for demos....')
    trajs = get_action_labels(inv_model, demos, 'continuous')
    print('Done!')
    bc_dataset = imitation_dataset(trajs)
    bc_loader = DataLoader(bc_dataset, batch_size=128, shuffle=True, num_workers=4)
    inv_model.cuda()

    ### PERFORM BEHAVIORAL CLONING
    print('Learning policy....')
    policy.cuda()
    bc_opt = optim.Adam(policy.parameters(), lr=1e-4)
    bc_loss = nn.MSELoss()

    for epoch in range(50):  
        for i, data in enumerate(bc_loader):
            s, a = data
            bc_opt.zero_grad()
            a_mu, a_sigma = policy(s.float().cuda())
            a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
            loss = bc_loss(a_pred, a.cuda())
            loss.backward()
            if i%20 == 0:
                print('Epoch:%d Batch:%d Loss:%.3f'%(epoch, i, loss.item()))
            bc_opt.step()
    print('Done!')

torch.save(policy, 'ant_policy.pt')







 