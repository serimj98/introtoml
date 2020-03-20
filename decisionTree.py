from __future__ import print_function
import sys
import csv
import math
import copy

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

def readFile(path):
    with open(path) as tsvfile:
        return list(csv.reader(tsvfile, dialect='excel-tab'))

train = readFile(train_input)
train_copy1 = copy.deepcopy(train)
train_copy2 = copy.deepcopy(train)
train_copy3 = copy.deepcopy(train)
train_copy4 = copy.deepcopy(train)
train_copy5 = copy.deepcopy(train)

test = readFile(test_input)
test_copy1 = copy.deepcopy(test)
test_copy2 = copy.deepcopy(test)
test_copy3 = copy.deepcopy(test)
test_copy4 = copy.deepcopy(test)
test_copy5 = copy.deepcopy(test)

def data_info(data):

	col_list, data_list = [], []
	for col in range(len(data[0])):
		for instance in data:
			if (instance[col] not in col_list):
				col_list.append(instance[col])
		data_list.append(col_list)
		col_list = []
	
	return (data_list)

train_data_info_list = data_info(train) #[[attribute, category 1, category 2], ...]
test_data_info_list = data_info(test)

train.pop(0)
test.pop(0)

def entropy_calc(data, data_info_list):

	y_data = data_info_list[-1]
	
	y_list = [] #all instances in y column
	for instance in data:
		y_list.append(instance[-1])

	ycount1, ycount2 = 0, 0
	for i in range(len(y_list)): #counting number of particular instance in y columm
		if (y_list[i] == y_data[1]):
			ycount1 += 1
		if (y_list[i] == y_data[2]):
			ycount2 += 1

	if (ycount1 == 0 or ycount2 == 0):
		entropy = 0
	else:
		entropy = - (ycount1/len(data))*math.log2(ycount1/len(data)) - \
					(ycount2/len(data))*math.log2(ycount2/len(data))

	return (entropy)

def cond_entropy_calc(data, x_index, data_info_list):

	x_data = data_info_list[x_index]
	y_data = data_info_list[-1]

	x_list = [] #all instances in x column
	for instance in data:
		x_list.append(instance[x_index])

	xcount1, xcount2 = 0, 0
	x_list1, x_list2 = [], [] #all instances of x = 0, x = 1
	for i in range(len(x_list)): #counting number of particular instance in x column of x_index
		if (x_list[i] == x_data[1]):
			xcount1 += 1
			x_list1.append(data[i])
		if (x_list[i] == x_data[2]):
			xcount2 += 1
			x_list2.append(data[i])

	ycount1_list1, ycount2_list1 = 0, 0
	for instance in x_list1:
		if (instance[-1] == y_data[1]):
			ycount1_list1 += 1
		if (instance[-1] == y_data[2]):
			ycount2_list1 += 1

	ycount1_list2, ycount2_list2 = 0, 0
	for instance in x_list2:
		if (instance[-1] == y_data[1]):
			ycount1_list2 += 1
		if (instance[-1] == y_data[2]):
			ycount2_list2 += 1

	if (len(data) == 0):
		prob1 = 0
		prob2 = 0
	else:
		prob1 = xcount1/len(data)
		prob2 = xcount2/len(data)

	if (ycount1_list1 == 0 or ycount2_list1 == 0):
		entropy1 = 0
	else:
		entropy1 = -((ycount1_list1/xcount1)*math.log2((ycount1_list1/xcount1))) \
				   -((ycount2_list1/xcount1)*math.log2((ycount2_list1/xcount1)))

	if (ycount1_list2 == 0 or ycount2_list2 == 0):
		entropy2 = 0
	else:
		entropy2 = -((ycount1_list2/xcount2)*math.log2((ycount1_list2/xcount2))) \
				   -((ycount2_list2/xcount2)*math.log2((ycount2_list2/xcount2)))

	return (prob1*entropy1 + prob2*entropy2)

class Node:
	def __init__(self):
		self.left = None
		self.right = None
		self.split = None
		self.category = None
		self.decision = None
		self.data = None
		self.depth = 0

train_tree = Node()
test_tree = Node()

def train_stump(node, data, depth, data_info_list):

	node.data = data
	depth = int(depth)
	y_data = data_info_list[-1]

	#majority vote classifier

	y_list = [] #all instances in y column
	for instance in data:
		y_list.append(instance[-1])

	ycount1, ycount2 = 0, 0
	for instance in y_list:
		if (instance == y_data[1]):
			ycount1 += 1
		if (instance == y_data[2]):
			ycount2 += 1

	#base case
	if (depth == 0 or (len(data_info_list)-1) == 0 or ycount1 == 0 or ycount2 == 0):

		if (ycount1 >= ycount2):
			node.decision = y_data[1]
		else:
			node.decision = y_data[2]

		return (node)

	if (len(data_info_list)-1 == 1):

		node.left = Node()
		node.right = Node()
		node.left.depth = node.depth+1
		node.right.depth = node.depth+1

		x_data = data_info_list[0]
		node.left.split = x_data[0]
		node.right.split = x_data[0]
		data_info_list.pop(0)
		info1 = copy.deepcopy(data_info_list)
		info2 = copy.deepcopy(data_info_list)

		x_list = [] #all instances in x column with maximum mutual information
		for instance in data:
			x_list.append(instance[0])

		xcount1, xcount2 = 0, 0
		x_list1, x_list2 = [], [] #all instances of x = 0, x = 1
		for i in range(len(x_list)): #counting number of particular instance in x column of max_ind
			if (x_list[i] == x_data[1]):
				xcount1 += 1
				x_list1.append(data[i])
			if (x_list[i] == x_data[2]):
				xcount2 += 1
				x_list2.append(data[i])

		if (xcount1 >= xcount2):
			node.left.category = x_data[1]
			node.right.category = x_data[2]
		else:
			node.left.category = x_data[2]
			node.right.category = x_data[1]

		train_stump(node.left, x_list1, depth-1, info1)
		train_stump(node.right, x_list2, depth-1, info2)

		return (node)

	else:

		node.left = Node()
		node.right = Node()
		node.left.depth = node.depth+1
		node.right.depth = node.depth+1

		#calculate mutual information for all x columns
		mutual_info = []
		for x in range(len(data_info_list)-1):
			mutual_info.append(entropy_calc(data, data_info_list) - cond_entropy_calc(data, x, data_info_list))

		#find maximum mutual information index (ex. x1, x2..., xi)
		max_val = 0
		for i in range(len(mutual_info)):
			if (mutual_info[i] >= max_val):
				max_ind = i
				max_val = mutual_info[i]

		if (max_val == 0):

			#majority vote classifier
			y_list = [] #all instances in y column
			for instance in data:
				y_list.append(instance[-1])

			ycount1, ycount2 = 0, 0
			for instance in y_list:
				if (instance == y_data[1]):
					ycount1 += 1
				if (instance == y_data[2]):
					ycount2 += 1

			if (ycount1 >= ycount2):
				node.decision = y_data[1]
			else:
				node.decision = y_data[2]

			return (node)

		else:

			x_data = data_info_list[max_ind]
			node.left.split = x_data[0]
			node.right.split = x_data[0]
			data_info_list.pop(max_ind)
			info1 = copy.deepcopy(data_info_list)
			info2 = copy.deepcopy(data_info_list)

			#split based on attribute with maximum mutual information

			x_list = [] #all instances in x column with maximum mutual information
			for instance in data:
				x_list.append(instance[max_ind])

			xcount1, xcount2 = 0, 0
			x_list1, x_list2 = [], [] #all instances of x = 0, x = 1
			for i in range(len(x_list)): #counting number of particular instance in x column of max_ind
				if (x_list[i] == x_data[1]):
					xcount1 += 1
					x_list1.append(data[i])
				if (x_list[i] == x_data[2]):
					xcount2 += 1
					x_list2.append(data[i])

			if (xcount1 >= xcount2):
				node.left.category = x_data[1]
				node.right.category = x_data[2]
			else:
				node.left.category = x_data[2]
				node.right.category = x_data[1]

			x_new_list1, x_new_list2 = [], [] #x_list1 and x_list2 with the splitting attribute removed
			for instance in x_list1:
				instance.pop(max_ind)
				x_new_list1.append(instance) #get rid of x attribute chosen for x = 0
			for instance in x_list2:
				instance.pop(max_ind)
				x_new_list2.append(instance) #get rid of x attribute chosen for x = 1

			train_stump(node.left, x_new_list1, depth-1, info1)
			train_stump(node.right, x_new_list2, depth-1, info2)

	return (node)

def num_category(data, data_info_list):

	y_data = data_info_list[-1]
	y_list = [] #all instances in y column
	
	if (data == None):
		return("")

	else:
		for instance in data:
			y_list.append(instance[-1])

		ycount1, ycount2 = 0, 0
		for instance in y_list:
			if (instance == y_data[1]):
				ycount1 += 1
			if (instance == y_data[2]):
				ycount2 += 1

		return("[" + str(ycount1) + str(y_data[1]) + str("/") + str(ycount2) + str(y_data[2]) + "]")

#A function to do preorder tree traversal
def printPreorder(root, data_info_list, depth = 0):

	if root:

		if (depth == 0):
			print(num_category(root.data, data_info_list))

		elif (root.data == None):
			pass

		#First print the data of node
		else:
			print('| '*int(root.depth), str(root.split), '=', str(root.category), ":", \
				root.decision, num_category(root.data, data_info_list), "\n", end = "")
		depth += 1

		#Then recur on left child
		printPreorder(root.left, data_info_list, depth)

		#Finally recur on right child
		printPreorder(root.right, data_info_list, depth)

train_predict_list = [] #creating dataset without y category for train dataset

for instance in train_copy3:
	instance.pop(-1)
	train_predict_list.append(instance)

train_col_names = train_predict_list.pop(0)

test_predict_list = [] #creating dataset without y category for test dataset

for instance in test_copy3:
	instance.pop(-1)
	test_predict_list.append(instance)

test_col_names = test_predict_list.pop(0)

def predict_label(predict_elem, col_names, tree):

	if (tree.decision != None):
		return (tree.decision)

	else:
		attribute = tree.left.split
		category1 = tree.left.category
		category2 = tree.right.category

		for i in range(len(col_names)):
			if (col_names[i] == attribute):
				decision = predict_elem[i]
				if (decision == category1):
					return (predict_label(predict_elem, col_names, tree.left))
				if (decision == category2):
					return (predict_label(predict_elem, col_names, tree.right))

def error_rate(predict_list, actual_list):
	count = 0
	for i in range(len(predict_list)):
		if (predict_list[i] != actual_list[i]):
			count += 1
	return (count / len(predict_list))

train_y_list = [] #all instances in y column
for instance in train:
	train_y_list.append(instance[-1])

test_y_list = [] #all instances in y column
for instance in test:
	test_y_list.append(instance[-1])

train_tree_comp = train_stump(train_tree, train, max_depth, train_data_info_list)

printPreorder(train_tree_comp, train_data_info_list)

train_labels = []
for i in range(len(train_predict_list)):
	train_labels.append(predict_label(train_predict_list[i], train_col_names, train_tree_comp))

# test_tree_comp = train_stump(test_tree, test, max_depth, test_data_info_list)

# printPreorder(train_tree_comp, test_data_info_list)

test_labels = []
for i in range(len(test_predict_list)):
	test_labels.append(predict_label(test_predict_list[i], test_col_names, train_tree_comp))

with open(train_out, "w") as f:
	for i in range(len(train_labels)):
		f.writelines(train_labels[i] + "\n")

with open(test_out, "w") as f:
	for i in range(len(test_labels)):
		f.writelines(test_labels[i] + "\n")

with open(metrics_out, "w") as f:
	f.writelines("error(train): " + str(error_rate(train_labels, train_y_list)) + "\n" + \
		"error(test): " + str(error_rate(test_labels, test_y_list)))
