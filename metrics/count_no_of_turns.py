import os
import numpy as np
import argparse

def main(args):
    all_files = os.listdir(f"{args.path}")
    debate, meta, helpful, product=[],[],[],[]
    for files in all_files:
        dialogues = open(f"{args.path}/{files}")
        lines = dialogues.readlines()
        lines =[line for line in lines if line.strip()]
        if 'debate' in files:
            debate.append(len(lines))
        elif 'meta_reviews' in files:
            meta.append(len(lines))
        elif 'helpful_reviews' in files:
            helpful.append(len(lines))
        else:
            continue
    print(f'Average for debates:{np.mean(debate)}')
    print(f'Average for meta_reviews:{np.mean(meta)}')
    print(f'Average for products:{np.mean(helpful)}')


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str)
	args = parser.parse_args()
	main(args)

