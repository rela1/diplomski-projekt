import os
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	with open(sys.argv[1]) as f:
		lines = f.read().splitlines()
	train_acc = []
	c = []
	valid_acc = []
	for index, line in enumerate(lines):
		if line.startswith('Train'):
			c.append(float(lines[index + 1].split('=')[1].strip()))
			for i in range(2, 6):
				if lines[index + i].strip().startswith('accuracy'):
					train_acc.append(lines[index + i].split('=')[1].strip())
		if line.startswith('Valid'):
			for i in range(2, 6):
				if lines[index + i].strip().startswith('accuracy'):
					valid_acc.append(lines[index + i].split('=')[1].strip())
		if len(line.strip()) == 0:
			break
	assert len(c) == len(train_acc)
	assert len(train_acc) == len(valid_acc)
	plt.subplot('211')
	plt.title('Train accuracy')
	plt.xlabel('c')
	plt.ylabel('accuracy')
	plt.plot(c, train_acc)
	plt.subplot('212')
	plt.title('Validate accuracy')
	plt.xlabel('c')
	plt.ylabel('accuracy')
	plt.plot(c, valid_acc)
	plt.show()