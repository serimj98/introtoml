from __future__ import print_function
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6] #number of times backpropogation loops through training data
    hidden_units = sys.argv[7] #number of hidden units
    init_flag = sys.argv[8] #random or zero initialization
    learning_rate = sys.argv[9] #learning rate for SDG

def readFile(path):
	with open(path) as csvfile:
		return list(csv.reader(csvfile, delimiter=','))

train = readFile(train_input)
test = readFile(test_input)

x_train, y_train = [], [] # x = data
for row in train:
	x_train.append(row[1:])
	y_train.append(row[0])

for i in range(len(x_train)):
	x_train[i] = list(map(int, x_train[i]))

y_train_label = [] # y_label = labels
for y_num in y_train:
	y_train_label_i = np.zeros(10)
	y_train_label_i[int(y_num)] = 1
	y_train_label.append(y_train_label_i)

x_test, y_test = [], [] # x = data
for row in test:
	x_test.append(row[1:])
	y_test.append(row[0])

for i in range(len(x_test)):
	x_test[i] = list(map(int, x_test[i]))

y_test_label = [] # y_label = labels
for y_num in y_test:
	y_test_label_i = np.zeros(10)
	y_test_label_i[int(y_num)] = 1
	y_test_label.append(y_test_label_i)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train_label = np.array(y_train_label)
y_test_label = np.array(y_test_label)

#initialize weights and biases (alpha + beta)
#alpha: D x M+1, where D = hidden units and M = length of input vectors x
#beta: K x D+1, where K = classes of output layer y_hat and D = hidden units
def weights(hidden_units, x, init_flag):
	a = int(hidden_units)
	b = len(x[0])+1
	alpha = np.zeros((a,b))
	beta = np.zeros([10, int(hidden_units)+1])
	if (int(init_flag) == 1): #random
		alpha[:,1:] = np.random.uniform(-0.1, 0.1, (int(hidden_units), len(x[0])))
		beta[:,1:] = np.random.uniform(-0.1, 0.1, (10, int(hidden_units)))
	return ((alpha, beta))

test_alpha = np.matrix([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1],[1,1,0,2,1,-2,2]])
test_beta = np.matrix([[1,1,2,-2,1],
			 [1,1,-1,1,2],
			 [1,3,1,-1,1]])

test_x = np.matrix([1,1,1,0,0,1,1]).T
test_y = np.matrix([0,1,0]).T

def linearforward(layer, weight): #a, b
	return np.dot(weight, layer)

a= linearforward(test_x, test_alpha)

def sigmoidforward(a): #z
	return 1/(1+np.exp(-a))

z=sigmoidforward(a)
z_bias = np.insert(z, 0, 1)
b=linearforward(z_bias.T, test_beta)
def softmaxforward(b): #y_hat
	return np.divide(np.exp(b), np.sum(np.exp(b)))

y_hat = softmaxforward(b)

def crossentropyforward(y, y_hat): #J = total loss
	return -np.dot(y.T, np.log(y_hat))

J = (crossentropyforward(test_y, y_hat))

def nnforward(x, y, alpha, beta):
	x = np.insert(x, 0, 1)
	a = linearforward(x, alpha)
	z = sigmoidforward(a)
	z_bias = np.insert(z, 0, 1)
	b = linearforward(z_bias, beta)
	y_hat = softmaxforward(b)
	J = crossentropyforward(y, y_hat)
	return (a, z, b, y_hat, J)

g_b = np.subtract(y_hat, test_y)
z_bias = np.insert(z, 0, 1)
g_beta = np.dot(g_b, z_bias)
b_star = np.delete(test_beta, 0, 1)
g_z = np.dot(b_star.T, g_b)
g_a = np.multiply(g_z, z)
g_a = np.multiply(g_a, (1-z))
g_alpha = np.dot(g_a, test_x.T)

def nnbackward(x, y, alpha, beta, o):
	a, z, b, y_hat, J = o
	x = np.insert(x, 0, 1)
	z = z.reshape(1, z.shape[0])
	z_bias = np.insert(z, 0, 1)
	z_bias = z_bias.reshape(1, z_bias.shape[0])
	g_b = np.subtract(y_hat, y)
	g_b_res = g_b.reshape(1, g_b.shape[0])
	g_beta = np.dot(g_b_res.T, z_bias)
	b_star = np.delete(beta, 0, 1)
	g_z = np.dot(b_star.T, g_b)
	g_a = np.multiply(g_z, z)
	g_a = np.multiply(g_a, (1-z))
	g_a_res = np.array(g_a).reshape(1, np.array(g_a).shape[1])
	x_res = np.array(x).reshape(1, np.array(x).shape[0])
	g_alpha = np.dot(g_a_res.T, x_res)
	return (g_alpha, g_beta)

def predict(x, y, alpha, beta):
	predict = []
	error = 0
	for i in range(len(x)):
		(a, z, b, y_hat, J) = nnforward(x[i], y[i], alpha, beta)
		# print("y[i]", y[i])
		# print("y_hat", y_hat)
		maximum_y_hat = np.argmax(y_hat)
		maximum_y = np.argmax(y[i])
		# print("maximum_y", maximum_y)
		predict.append(maximum_y_hat)
		# print("maximum_y_hat", maximum_y_hat)
		if (maximum_y_hat != maximum_y):
			error += 1
	error /= len(x)
	return (predict, error)

def meancrossentropy(x, y, alpha, beta):
	entropy = []
	for i in range(len(x)):
		(a, z, b, y_hat, J) = nnforward(x[i], y[i], alpha, beta)
		entropy.append(J)
	# print(entropy)
	return np.sum(np.array(entropy))/len(x)

def sdg(x_train, y_train, x_test, y_test, hidden_units, init_flag):
	alpha, beta = weights(hidden_units, x_train, init_flag)
	meancrossentropy_list_train = []
	meancrossentropy_list_test = []
	for e in range(int(num_epoch)):
		for i in range(len(x_train)):
			o = (a, z, b, y_hat, J) = nnforward(x_train[i], y_train[i], alpha, beta)
			g_alpha, g_beta = nnbackward(x_train[i], y_train[i], alpha, beta, o)
			alpha -= np.dot(float(learning_rate), g_alpha)
			beta -= np.dot(float(learning_rate), g_beta)
		meancrossentropy_list_train.append(meancrossentropy(x_train, y_train, alpha, beta))
		meancrossentropy_list_test.append(meancrossentropy(x_test, y_test, alpha, beta))
	return (meancrossentropy_list_train, meancrossentropy_list_test, alpha, beta)

train_mce_list, test_mce_list, alpha_train, beta_train = sdg(x_train, y_train_label, x_test, y_test_label, hidden_units, init_flag)


train_pred, train_err = predict(x_train, y_train_label, alpha_train, beta_train)
test_pred, test_err = predict(x_test, y_test_label, alpha_train, beta_train)

with open(train_out, "w") as train_out:
	train_string = ""
	for i in train_pred:
		train_string += str(i) + "\n"
	print(train_string, file=train_out)

with open(test_out, "w") as test_out:
	test_string = ""
	for i in test_pred:
	    test_string += str(i) + "\n"
	print(test_string, file=test_out)

with open(metrics_out, "w") as metrics_out:
	for i in range(int(num_epoch)):
		#train printing
		print("epoch="+str(i+1), end=' ', file=metrics_out)
		print("crossentropy(train): "+str(train_mce_list[i]), file=metrics_out)
		#test printing
		print("epoch="+str(i+1), end=' ', file=metrics_out)
		print("crossentropy(test): "+str(test_mce_list[i]), file=metrics_out)
	print("error(train): "+str(train_err), file=metrics_out)
	print("error(test): "+str(test_err), file=metrics_out)










