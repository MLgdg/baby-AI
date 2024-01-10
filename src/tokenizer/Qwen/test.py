
from transformers import AutoTokenizer
import torch 
import os 
import base64



if __name__ == '__main__':

	tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
	s = '我爱你\n大傻逼'
	inputs = tokenizer([s,s]) #实现多个序列编码
	print (inputs)
	inputs = tokenizer.encode(s) #单个文本编码
	print (inputs)
	outputs = tokenizer.decode([198]) #单个文本解码
	print(base64.b64encode(outputs.encode("utf8")))