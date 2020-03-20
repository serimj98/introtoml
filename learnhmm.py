from __future__ import print_function
import sys
import math
import numpy as np

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

def readFile(path):
    with open(path, "r") as f:
        return f.readlines()

def remove_whitespace(whitespace_list):
	new_list = []
	for i in whitespace_list:
		if i.endswith('\n'):
			new_list.append(i[:-1])
		else:
			new_list.append(i)
	return new_list

train = remove_whitespace(readFile(train_input))
index_to_word = remove_whitespace(readFile(index_to_word))
index_to_tag = remove_whitespace(readFile(index_to_tag))

def hmmprior_fun(train, index_to_tag):
	tag_count = dict((index,0) for index in index_to_tag) #python 3.7 dictionaries ordered
	for key in tag_count:
		for line in train:
			if (line.split()[0].split("_")[1] == key): #t_1
				tag_count[key] += 1
	
	prior = np.zeros(len(index_to_tag))
	for key in range(len(tag_count)):
		prior[key] = float((list(tag_count.values())[key] + 1) / (sum(list(tag_count.values())) + len(index_to_tag)))
	return (prior)

def hmmtrans_fun(train, index_to_tag):
	tag_list = []
	for tag1 in index_to_tag:
		for tag2 in index_to_tag:
			tag_list.append((tag1, tag2))

	tag_count = dict((index,0) for index in tag_list)
	for key in range(len(tag_count)):
		for line in train:
			list_line = line.split()
			for word in range(1, len(list_line)):
				prev_t_i = list_line[word-1].split("_")[1]
				t_i = list_line[word].split("_")[1]
				if ((prev_t_i, t_i) == list(tag_count.keys())[key]):
					tag_count[list(tag_count.keys())[key]] += 1

	trans = np.zeros((len(index_to_tag), len(index_to_tag)))
	count = 0
	for row in range(len(trans)):
		for col in range(len(trans[0])):
			prob_sum = 0
			for key in range(len(tag_count)):
				if (index_to_tag[row] == list(tag_count.keys())[key][0]):
					prob_sum += list(tag_count.values())[key]
			trans[row, col] = float((list(tag_count.values())[count] + 1) / (prob_sum + len(index_to_tag)))
			count += 1
	return (trans)

def hmmemit_fun(train, index_to_word, index_to_tag):
	words = []
	for line in train:
		list_line = line.split()
		for word in list_line:
			words.append((word.split("_")[1], word.split("_")[0]))

	emit = np.zeros((len(index_to_tag), len(index_to_word)))
	for key in range(len(index_to_tag)):
		for word in range(len(index_to_word)):
			curr = ((index_to_tag[key], index_to_word[word]))
			emit[key, word] = float((sum(map(lambda x : x == curr, words)) + 1) / \
								((sum(map(lambda x : x[0] == curr[0], words)) + len(index_to_word))))

	return (emit)

with open(hmmprior, "w") as hmmprior:
	prior = hmmprior_fun(train, index_to_tag)
	prior_string = ""
	for i in prior:
		prior_string += (str(i)).strip() + "\n"
	print(prior_string[:-1], file=hmmprior)

with open(hmmemit, "w") as hmmemit:
	emit = hmmemit_fun(train, index_to_word, index_to_tag)
	emit_string = ""
	for i in emit:
		emit_string += (str(i)[1:-1]).strip() + "\n"
	print(emit_string[:-1], file=hmmemit)

with open(hmmtrans, "w") as hmmtrans:
	trans = hmmtrans_fun(train, index_to_tag)
	trans_string = ""
	for i in trans:
		trans_string += (str(i)[1:-1]).strip() + "\n"
	print(trans_string[:-1], file=hmmtrans)
