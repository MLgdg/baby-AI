
import toml



def main():
	path = './llama.toml'
	dic = toml.load(path)
	print(dic)

if __name__ == '__main__':
	main()