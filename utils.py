import gym
import torch
import numpy as np
from torch.distributions import Normal
from torch.distributions import Categorical

def select_action_continuous(state, policy):
	state = torch.from_numpy(state).float()
	mean, sigma = policy(state)
	pi = Normal(loc=mean, scale=sigma)
	action = pi.sample()
	return action.numpy()

def select_action_discrete(state, policy):
	state = torch.from_numpy(state).float()
	probs = policy(state)
	pi = Categorical(probs)
	action = pi.sample()
	return action.numpy()

def gen_inv_samples(env, policy, num_samples, env_type, use_policy):
	count = 0
	transitions = []
	s = env.reset()
	t = 0
	while count < num_samples:
		if env_type == 'continuous':
			if use_policy:
				a = select_action_continuous(s, policy)
			else:
				a = env.action_space.sample()
		else:
			a = select_action_discrete(s, policy)
		s_prime, _, done, _ = env.step(a)
		transitions.append([s, a, s_prime])
		count += 1
		t += 1
		if done == True or t>1000:
			s = env.reset()
			t = 0
		else:
			s = s_prime
	return transitions

def get_action_labels(inv_model, states, env_type):
	state_trajs = []
	state_traj_ep = []
	for i in range(len(states)):
		if (i+1)%50 == 0:
			state_trajs.append(state_traj_ep)
			state_traj_ep = []
		else:
			state_traj_ep.append(states[i])
	trajs = []
	for state_traj in state_trajs:
		traj = []
		for idx in range(len(state_traj)-1):
			s = state_traj[idx]
			s_prime = state_traj[idx+1]
			a = inv_model(torch.from_numpy(s).unsqueeze(0).float(), torch.from_numpy(s_prime).unsqueeze(0).float())
			if env_type == 'continuous':
				traj.append([s, a.detach().numpy()])
			else:
				a = np.max(a.detach().numpy())
				traj.append([s, a])
		trajs.append(traj)
	return trajs