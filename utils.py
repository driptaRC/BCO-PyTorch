import gym
import torch
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

def gen_inv_samples(env, policy, num_samples, env_type):
	count = 0
	transitions = []
	s = env.reset()
	while count < num_samples:
		if env_type == 'continuous':
			a = select_action_continuous(s, policy)
		else:
			a = select_action_discrete(s, policy)
		s_prime, _, done, _ = env.step(a)
		transitions.append([s, a, s_prime])
		count += 1
		if done == True:
			s = env.reset()
		else:
			s = s_prime
	return transitions

def get_action_labels(inv_model, state_trajs, env_type):
	trajs = []
	for state_traj in state_trajs:
		traj = []
		for idx in range(len(state_traj)-1):
			s = state_traj[idx]
			s_prime = state_traj[idx+1]
			a = inv_model(torch.from_numpy(s).float(), torch.from_numpy(s_prime).float())
			if env_type == 'continuous':
				traj.append([s, a.numpy()])
			else:
				a = np.max(a.numpy())
				traj.append([s, a])
		trajs.append(traj)
	return trajs