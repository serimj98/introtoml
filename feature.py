from __future__ import print_function
import sys
import csv

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

def readTsvFile(path):
    with open(path) as tsvfile:
        return list(csv.reader(tsvfile, dialect='excel-tab'))

def readTxtFile(path):
    with open(path, "r") as f:
        return f.read()

train = readTsvFile(train_input)
validation = readTsvFile(validation_input)
test = readTsvFile(test_input)
dictionary = readTxtFile(dict_input)

dictionary_split = dictionary.splitlines()

def dict_dict(dictionary):
	dict_words = {}
	for elem in range(len(dictionary)):
		word = dictionary[elem].split(" ")[0]
		index = dictionary[elem].split(" ")[1]
		dict_words[word] = index
	return dict_words

dict_new = dict_dict(dictionary_split)

def model_one(data, dictionary):
	output = []
	for review in range(len(data)):
		label = data[review][0]
		content = data[review][1].split()
		output.append(label)
		output.append('\t')
		temp_content = []
		for word in range(len(content)):
			if (content[word] not in temp_content):
				temp_content.append(content[word])
		for word in range(len(temp_content)):
			if (temp_content[word] in dictionary):
				output.append(dictionary[temp_content[word]])
				output.append(":1")
				output.append('\t')
		output.append('\n')
	return ''.join(output)

def count_dict(dictionary):
	dict_count = {}
	for elem in range(len(dictionary)):
		word = dictionary[elem].split()[0]
		dict_count[word] = 0
	return dict_count

def model_two(data, dictionary, dict_split):
	output = []
	for review in range(len(data)):
		dict_count = count_dict(dict_split)
		label = data[review][0]
		content = data[review][1].split()
		output.append(label)
		output.append('\t')
		for word in range(len(content)):
			if (content[word] in dictionary):
				dict_count[content[word]] += 1
		temp_content = []
		for word in range(len(content)):
			if (content[word] not in temp_content):
				temp_content.append(content[word])
		for word in range(len(temp_content)):
			if ((temp_content[word] in dictionary) and (dict_count[temp_content[word]] < 4)):
				output.append(dictionary[temp_content[word]])
				output.append(":1")
				output.append('\t')
		output.append('\n')
	return ''.join(output)

if (int(feature_flag) == 1):
	with open(formatted_train_out, "w") as f:
		f.write(model_one(train, dict_new))
	with open(formatted_validation_out, "w") as f:
		f.write(model_one(validation, dict_new))
	with open(formatted_test_out, "w") as f:
		f.write(model_one(test, dict_new))
if (int(feature_flag) == 2):
	with open(formatted_train_out, "w") as f:
		f.write(model_two(train, dict_new, dictionary_split))
	with open(formatted_validation_out, "w") as f:
		f.write(model_two(validation, dict_new, dictionary_split))
	with open(formatted_test_out, "w") as f:
		f.write(model_two(test, dict_new, dictionary_split))