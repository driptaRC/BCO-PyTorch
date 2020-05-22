import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Normal
import pickle
import gym
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader

class policy(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(policy, self).__init__()
		self.linear_1 = nn.Linear(state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_m = nn.Linear(hidden_dim, action_dim)
		self.linear_v = nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.001)
		x_m = self.linear_m(x)
		x_v = self.linear_v(x)
		return x_m, x_v

class imitation_dataset(Dataset):
    def __init__(self, demos):
        self.data = []
        for i, state in enumerate(demos['observations']):
        	action = demos['actions'][i][0]
        	self.data.append((state, action))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a = self.data[idx]        
        return s, a



### ARGS
ENV_NAME = 'Ant-v2'
DEMO_DIR = os.path.join('experts', ENV_NAME+'.pkl')
RENDER = False

### PREPARE DATA
with open(DEMO_DIR, 'rb') as f:
	demos = pickle.load(f)

env = gym.make(ENV_NAME)

pi = policy(env.observation_space.shape[0],32,env.action_space.shape[0]).cuda()

dataset = imitation_dataset(demos)

loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
optimizer = optim.Adam(pi.parameters(), lr=1e-3, weight_decay=0.0001)
criterion = nn.MSELoss()


### TRAIN POLICY
for epoch in range(30):  
    running_loss = 0
    for i, data in enumerate(loader):
        s, a = data
        optimizer.zero_grad()
        a_mu, a_sigma = pi(s.float().cuda())
        a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
        loss = criterion(a_pred, a.cuda())
        running_loss += loss.item()
        loss.backward()
        if i%20 == 19:
            print('Epoch:%d Batch:%d Loss:%.3f'%(epoch, i+1, running_loss/20))
            running_loss = 0
        optimizer.step()
print('Done!')

### EVALUATE POLICY
max_steps = env.spec.timestep_limit
pi.cpu()
returns = []
for i in range(10):
	print('iter', i)
	obs = env.reset()
	done = False
	totalr = 0.
	steps = 0
	while not done:
		a_mu, a_sigma = pi(torch.from_numpy(obs).float())
		a = Normal(loc=a_mu, scale=a_sigma).sample()
		obs, r, done, _ = env.step(a.detach().numpy())
		if RENDER:
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