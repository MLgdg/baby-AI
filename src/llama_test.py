import torch 
import toml

from model.LLAMA import llama 

def llama_unit_test():
	config = toml.load("./config/LLAMA/llama.toml")
	model = llama.LLAMA(config)
	data = torch.tensor([[1,2,3],[4,5,6]])
	res = model(data)
	print (res.shape)

if __name__ == '__main__':
	llama_unit_test()

