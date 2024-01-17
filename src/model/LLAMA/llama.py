

import torch.nn as nn
import transformers
import os 
import toml
import torch 
import math 


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

class PEMB():


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
		self.n_head = hidden_dim // head_dim
		self.qw = nn.Linear(hidden_dim, hidden_dim)
		self.kw = nn.Linear(hidden_dim, hidden_dim)
		self.vw = nn.Linear(hidden_dim, hidden_dim)
		self.ow = nn.Linear(hidden_dim, hidden_dim)
		assert n_head * head_dim == hidden_dim "fuck error config model hidden_dim,head_dim "

	def attention(self, inp, mask):
		b, s, dim = inp.shape
		q = self.qw(inp).view(b, s, self.n_head, dim//self.n_head).transpose(1, 2)
		k = self.kw(inp).view(b, s, self.n_head, dim//self.n_head).transpose(1, 2)
		v = self.vw(inp).view(b, s, self.n_head, dim//self.n_head).transpose(1, 2)

		#TOOD postion embedding
		qk = q @ k.transpose(2, 3) / math.sqrt(self.head_dim) + mask 
		qk = torch.softmax(qk, dim=-1)
		out = qk @ v
		out = out.transpose(1 ,2).contiguous().view(b, s, dim)
		return self.ow(out) 


	def forward(self, inp, mask):
		return self.attention(inp, mask)


class TransformerBlock(nn.Module):
	def __init__(self, config):
		self.config = config
		super(TransformerBlock, self).__init__()
		self.ff = FeedForward(config)
		self.att = Attenion(config)
		self.ln1 = RMSNorm(config)
		self.ln2 = RMSNorm(config)
		self.dro = torch.nn.Dropout(config["hyperparameter"]["dropout"])

	def forward(self, inp, mask):
		out1 = self.ln1(inp)
		out1 = self.att(out, mask)
		out1 = out1 + inp
		out2 = self.ln2(out1)
		out2 = self.ff(out2)
		return out1 + out2 




class LLAMA(nn.Module):
	def __init__(self, config):
		super(LLAMA, self).__init__()
		dim = config["model"]["hidden_dim"]
		vocab_dim = config["model"]["vocab_dim"]
		layer_num = config["model"]["layer_num"]
		self.emb = EMB(config)
		self.c = nn.Linear(dim, vocab_dim)
		self.transformers = nn.Module(TransformerBlock(config) for i in range(layer_num))

	def generate_causal_matrix(self, tokens, start_pos=0):
		_ seqlen = tokens.shape
		start_pos = 0
		mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
		mask = torch.triu(mask, diagonal=start_pos + 1)#.type_as(h)
		return mask

if __name__ == '__main__':
	
	config = {"model": {"hidden_dim": 3, "intermediate_dim": 2}, "hyperparameter": {"eps": 1e-6}}
	RMS = RMSNorm(config)
	FF = FeedForward(config)
	test_data = torch.rand(1,2,3)
	out_data = RMS(test_data)
	out_data = FF(out_data)
	print(out_data)








