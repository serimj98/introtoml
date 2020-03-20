from __future__ import print_function
import sys
import csv
import math

if __name__ == '__main__':
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

def readTsvFile(path):
    with open(path) as tsvfile:
        return list(csv.reader(tsvfile, dialect='excel-tab'))

def readTxtFile(path):
    with open(path, "r") as f:
        return f.read()

train = readTsvFile(formatted_train_input)
validation = readTsvFile(formatted_validation_input)
test = readTsvFile(formatted_test_input)
dictionary = readTxtFile(dict_input)

dictionary_split = dictionary.splitlines()

def find_index(review):
	index_list = []
	for index in range(1, len(review)):
		index_list.append(review[index].split(':')[0])
	return set(index_list)

theta = [0] * (len(dictionary_split) + 1) #theta[0] = bias term

def create_index_list(data):
	index_list = []
	for review in range(len(data)):
		index_list.append(find_index(data[review]))
	return index_list

def cond_log_likelihood(data):
	total_sum = 0
	index_list = create_index_list(data)
	for review in range(len(data)):
		sum_theta = 0
		index_list = find_index(data[review])
		for index in range(1, len(theta)):
			if index in index_list:
				sum_theta += theta[index]
			sum_theta += theta[0]
		y = int(data[review][0])
		total_sum += -y*sum_theta + math.log(1 + math.pow(math.e, sum_theta))
	return -total_sum

def update_rule(data, theta):
	index_list = create_index_list(data)
	for review in range(len(data)):
		total_sum = 0
		sum_theta = 0
		y = int(data[review][0])
		for word in index_list[review]: #word index in review
			sum_theta += theta[int(word)+1]
		sum_theta += theta[0]
		total_sum = (y - ((math.exp(sum_theta))/(1 + math.exp(sum_theta))))
		for j_index in index_list[review]:
			theta[int(j_index)+1] -= 0.1*(-total_sum)
		theta[0] -= 0.1*(-total_sum)
	return theta

def epoch_calc(data, theta, epoch):
	new_theta = theta
	for i in range(int(num_epoch)):
		new_theta2 = update_rule(data, new_theta)
	return new_theta2

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def classify(data, theta):
	classify_list = []
	index_list = create_index_list(data)
	for review in range(len(data)):
		sum_theta = 0
		for word in index_list[review]:
			sum_theta += theta[int(word)+1]
		sum_theta += theta[0]
		if (sigmoid(sum_theta) >= .5):
			classify_list.append(1)
		else:
			classify_list.append(0)
	return classify_list


# print(create_index_list(train)[0])
# print(classify(train, epoch_calc(train, theta, num_epoch)))
# print(partial_derivative(train))
# print(update_rule(train, theta))

train_classify_list = []
for review in range(len(train)):
	train_classify_list.append(int(train[review][0]))

test_classify_list = []
for review in range(len(test)):
	test_classify_list.append(int(test[review][0]))

def error_rate(predict_list, actual_list):
	count = 0
	for i in range(len(predict_list)):
		if (predict_list[i] != actual_list[i]):
			count += 1
	return count / len(predict_list)


train_classify = classify(train, epoch_calc(train, theta, num_epoch))

test_classify = classify(test, theta)

print(test_classify)

with open(train_out, "w") as f:
	for i in range(len(train_classify)):
		f.writelines(str(train_classify[i]) + "\n")

with open(test_out, "w") as f:
	for i in range(len(test_classify)):
		f.writelines(str(test_classify[i]) + "\n")

with open(metrics_out, "w") as f:
	f.writelines("error(train): " + str(error_rate(train_classify, train_classify_list)) + "\n" + \
		"error(test): " + str(error_rate(test_classify, test_classify_list)))