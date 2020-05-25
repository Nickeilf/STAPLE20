import numpy as np
import argparse
from random import Random
from tqdm import tqdm

seed = 1234

def shuffle_data(src_path, tgt_path):
	with open(src_path,"rb") as f:
			src = f.readlines()
	with open(tgt_path,"rb") as f:
			tgt = f.readlines()

	assert len(src) == len(tgt)
	src_result = []
	tgt_result = []
	index = list(range(len(src)))

	Random(seed).shuffle(index)
	for i in tqdm(range(len(src))):
		src_result.append(src[index[i]])
		tgt_result.append(tgt[index[i]])

	assert len(src_result) == len(tgt_result)
	assert len(src_result) == len(src)

	with open(src_path,"wb") as f:
			f.writelines(src_result)
	with open(tgt_path,"wb") as f:
			f.writelines(tgt_result)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", help="path of the source data file")
	parser.add_argument("-tgt", help="path of the target data file")
	args = parser.parse_args()

	print("-----shuffle training data-----")
	shuffle_data(args.src, args.tgt)
