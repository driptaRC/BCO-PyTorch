import torch
import torch.nn as nn
import torch.nn.functional as F

class policy_continuous(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(policy_continuous, self).__init__()
		self.linear_1 = nn.Linear(state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3_m = nn.Linear(hidden_dim, action_dim)
		self.linear_3_v = nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.001)
		x_m = self.linear_3_m(x)
		x_v = self.linear_3_v(x)
		return x_m, x_v

class policy_discrete(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(policy_continuous, self).__init__()
		self.flag_var = predict_var
		self.linear_1 = nn.Linear(state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3 = nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.01)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.01)
		x = F.softmax(self.linear_3(x), dim=-1)
		return x

class inv_dynamics_continuous(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(inv_dynamics_continuous, self).__init__()
		self.linear_1 = nn.Linear(2*state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3 = nn.Linear(hidden_dim, action_dim)

	def forward(self, s, s_prime):
		x = torch.cat([s, s_prime], dim=1)
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_3(x)
		return x

class inv_dynamics_discrete(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(inv_dynamics_discrete, self).__init__()
		self.linear_1 = nn.Linear(2*state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3 = nn.Linear(hidden_dim, action_dim)

	def forward(self, s, s_prime):
		x = torch.cat([s, s_prime], dim=1)
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.01)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.01)
		x = F.softmax(self.linear_3(x), dim=-1)
		return x