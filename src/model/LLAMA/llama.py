

import torch.nn as nn
import transformers
import os 
import toml
import torch 


class RMSNorm(nn.Module):

	def __init__(self, config):
		super(RMSNorm, self).__init__()
		self.dim = config["model"]["hidden_size"]
		self.eps = config["hyperparameter"]["eps"]
		self.balanced_distribution = nn.Parameter(torch.ones(self.dim))

	def forward(self, inp):

		out = inp * torch.rsqrt(inp.pow(2).mean(-1, keepdim=True) + self.eps)
		out = out * self.balanced_distribution
		return out 


if __name__ == '__main__':
	
	config = {"model": {"hidden_size": 3}, "hyperparameter": {"eps": 1e-6}}
	model = RMSNorm(config)
	test_data = torch.rand(1,2,3)
	print("inp data: ", test_data)
	out_data = model(test_data)
	print("out data: ", out_data.shape, out_data)