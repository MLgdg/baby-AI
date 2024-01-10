

import torch.nn as nn
import transformers
import os 
import toml
import torch 


class RMSNorm(nn.Module):

	def __init__(self, config):
		super(RMSNorm, self).__init__()
		self.dim = config["model"]["hidden_dim"]
		self.eps = config["hyperparameter"]["eps"]
		self.balanced_distribution = nn.Parameter(torch.ones(self.dim))

	def forward(self, inp):
		out = inp * torch.rsqrt(inp.pow(2).mean(-1, keepdim=True) + self.eps)
		out = out * self.balanced_distribution
		return out 


class EMB(nn.Embedding):
	def __init__(self, config):
		super(EMB, self).__init__()
		hidden_dim = config["model"]["hidden_dim"]
		vocab_dim = config["model"]["vocab_dim"]
		self.emb = nn.Embedding(vocab_dim, hidden_dim)

	def forward(self, inp):
		return self.emb(inp)


class FeedForward(nn.Module):
	def __init__(self, config):
		super(FeedForward, self).__init__()
		hidden_dim = config["model"]["hidden_dim"]
		intermediate_dim = config["model"]["intermediate_dim"]
		self.w1 = nn.Linear(hidden_dim, intermediate_dim)
		self.w2 = nn.Linear(hidden_dim, intermediate_dim)
		self.w3 = nn.Linear(intermediate_dim, hidden_dim)

	def forward(self, inp):
		tmp1 = self.w1(inp)
		tmp2 = self.w2(inp)
		out = self.w3(tmp1 * tmp2 * (1 / (1 + torch.exp(-tmp1))))
		return out 


class Attenion(nn.Module):
	def __init__(self, config):
		super(Attenion, self).__init__()
		hidden_dim = config["model"]["hidden_dim"]
		head_dim = config["model"]["head_dim"]
		n_head = hidden_dim // head_dim
		self.qw = nn.Linear(hidden_dim, hidden_dim)
		self.kw = nn.Linear(hidden_dim, hidden_dim)
		self.vw = nn.Linear(hidden_dim, hidden_dim)
		self.ow = nn.Linear(hidden_dim, hidden_dim)
		assert n_head * head_dim == hidden_dim "fuck error config model hidden_dim,head_dim "

	def generate_causal_matrix(self):

	def forward(self, inp):


class TransformerBlock(nn.Module):


class LLAMA(nn.Module):



if __name__ == '__main__':
	
	config = {"model": {"hidden_dim": 3, "intermediate_dim": 2}, "hyperparameter": {"eps": 1e-6}}
	RMS = RMSNorm(config)
	FF = FeedForward(config)
	test_data = torch.rand(1,2,3)
	out_data = RMS(test_data)
	out_data = FF(out_data)
	print(out_data)








