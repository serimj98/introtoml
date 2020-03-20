from __future__ import print_function
import sys
import math
import numpy as np

if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

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

def var_list(var):
	v_list = []
	for i in var:
		i = i.split()
		v_list.append(i)
	return v_list

test = var_list(remove_whitespace(readFile(test_input)))
index_to_word = remove_whitespace(readFile(index_to_word))
index_to_tag = remove_whitespace(readFile(index_to_tag))
hmmprior = remove_whitespace(readFile(hmmprior))
hmmemit = var_list(remove_whitespace(readFile(hmmemit)))
hmmtrans = var_list(remove_whitespace(readFile(hmmtrans)))

def logspace(t, j, lw, pi, A, B):
	lw_list = np.zeros(len(index_to_tag))
	if t == 0:
		return (np.log(float(pi[j])) + np.log(float(B[j][0])), j)
	# t > 1 
	for k in range(len(index_to_tag)):
		lw_list[k] = np.log(float(B[j][t])) + np.log(float(A[k][j])) \
						+ logspace(t-1, k, lw, pi, A, B)[0]
	return (np.max(lw_list), np.argmax(lw_list))

def viterbi(test, index_to_tag, pi, A, B):
	y_list = [0 for i in range(len(test))]
	for i in range(len(test)):
		T = len(test[i])
		J = len(index_to_tag)
		lw = np.zeros((T, J))
		p = np.zeros((T, J))

		for t in range(T):
			for j in range(J):
				if t == 0:
					lw[t][j] = np.log(float(pi[j])) + np.log(float(B[j][0]))
					p[t][j] = j
				else:
					lw[t][j] = logspace(t, j, lw, pi, A, B)[0]
					p[t][j] = logspace(t, j, lw, pi, A, B)[1]
		new_y = np.zeros(T)
		new_y[T-1] = np.argmax(lw[T-1])

		for t in range(T-1, 0, -1):
			new_y[t-1] = p[t][np.int(new_y[t])]
		y_list[i] = new_y

	final = []
	for i in range(len(test[0])):
		final.append(test[0][i].split("_")[0] + "_" + index_to_tag[int(y_list[0][i])])

	return y_list

viterbi_seq = viterbi(test, index_to_tag, hmmprior, hmmtrans, hmmemit)

def prediction(test, viterbi_seq):
	final = []
	for i in range(len(test)):
		fin_list = []
		for j in range(len(test[i])):
			fin_list.append(test[i][j].split("_")[0] + "_" + index_to_tag[int(viterbi_seq[i][j])])
		final.append(fin_list)
	return final

final_prediction = prediction(test, viterbi_seq)

def metrics(test, final_prediction):
	tot = 0
	cor = 0
	for i in range(len(test)):
		for j in range(len(test[i])):
			tot += 1
			if (test[i][j] == final_prediction[i][j]):
				cor += 1
	return cor/tot

final_metrics = metrics(test, final_prediction)

with open(predicted_file, "w") as predicted_file:
	predicted_str = ""
	for i in final_prediction:
		for j in i:
			predicted_str += str(j) + " "
		predicted_str += "\n"
	print(predicted_str, file=predicted_file)

with open(metric_file, "w") as metric_file:
	metric_str = "Accuracy: " + str(final_metrics)
	print(metric_str, file=metric_file)

